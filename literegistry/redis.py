import asyncio
import json
import math
from pathlib import Path
import sys
import tempfile
from typing import Optional, Union, List
from urllib.parse import urlsplit, urlunsplit
import redis.asyncio as redis
from literegistry.kvstore import KeyValueStore
import socket
import subprocess
import shutil
import os
import fire
import logging
from literegistry.runtime import build_runtime
from literegistry.coop.endpoints import (
    endpoint_registry_filesystem_path,
    normalize_endpoint_registry,
    run as supervise_endpoint,
    wait_for_endpoint,
)
from literegistry.sqlite import sqlite_registry_path


logger = logging.getLogger(__name__)


def redact_redis_url(url: str) -> str:
    """Return a Redis URL safe for logs."""
    parsed = urlsplit(url)
    if parsed.password is None:
        return url
    host = parsed.hostname or ""
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    credentials = f"{parsed.username}:***@" if parsed.username else ":***@"
    port = f":{parsed.port}" if parsed.port is not None else ""
    return urlunsplit(
        (parsed.scheme, f"{credentials}{host}{port}", parsed.path, parsed.query, parsed.fragment)
    )


class RedisKVStore(KeyValueStore):
    """Redis-based key-value store"""
    #  http://klone-login01.hyak.local:8080/v1/models
    def __init__(
        self,
        url: str = "redis://klone-login01.hyak.local:6379",
        db: int = 0,
        *,
        raise_on_error: bool = False,
        log_connections: bool = True,
    ):
        """
        Initialize Redis KV store
        
        Args:
            url: Redis connection URL (e.g., "redis://localhost:6379", "redis://user:pass@host:port")
            db: Redis database number
        """
        self.url = url
        self.db = db
        self.raise_on_error = raise_on_error
        self.log_connections = log_connections
        self._redis = None

    async def _get_redis(self) -> redis.Redis:
        """Get Redis connection, creating it if necessary"""
        if self._redis is None:
            try:
                self._redis = redis.from_url(self.url, db=self.db, decode_responses=False)
                # Test the connection
                await self._redis.ping()
                if self.log_connections:
                    logger.info(
                        "Successfully connected to Redis at %s",
                        redact_redis_url(self.url),
                    )
            except Exception as e:
                if self.log_connections:
                    logger.warning(
                        "Failed to connect to Redis at %s: %s",
                        redact_redis_url(self.url),
                        e,
                    )
                raise
        return self._redis

    async def ping(self) -> bool:
        """Check the Redis connection, propagating connection failures."""
        redis_client = await self._get_redis()
        return bool(await redis_client.ping())

    async def get(self, key: str) -> Optional[bytes]:
        """Get value for a key from Redis"""
        redis_client = await self._get_redis()
        try:
            value = await redis_client.get(key)
            return value
        except Exception:
            if self.raise_on_error:
                raise
            return None

    async def set(
        self,
        key: str,
        value: Union[bytes, str],
        ttl_seconds: Optional[float] = None,
    ) -> bool:
        """Set value for a key in Redis."""
        if ttl_seconds is not None:
            ttl_seconds = float(ttl_seconds)
            if not math.isfinite(ttl_seconds) or ttl_seconds <= 0:
                raise ValueError(
                    "ttl_seconds must be a finite value greater than zero"
                )
        redis_client = await self._get_redis()
        try:
            if isinstance(value, str):
                value = value.encode("utf-8")
            options = {}
            if ttl_seconds is not None:
                options["px"] = max(1, math.ceil(ttl_seconds * 1000))
            await redis_client.set(key, value, **options)
            return True
        except Exception:
            if self.raise_on_error:
                raise
            return False

    async def delete(self, key: str) -> bool:
        """Delete a key from Redis"""
        redis_client = await self._get_redis()
        try:
            result = await redis_client.delete(key)
            return result > 0
        except Exception:
            if self.raise_on_error:
                raise
            return False

    async def exists(self, key: str) -> bool:
        """Check if key exists in Redis"""
        redis_client = await self._get_redis()
        try:
            result = await redis_client.exists(key)
            return result > 0
        except Exception:
            if self.raise_on_error:
                raise
            return False

    async def keys(self, prefix: Optional[str] = None) -> List[str]:
        """Get keys from Redis using a non-blocking scan."""
        redis_client = await self._get_redis()
        try:
            pattern = f"{prefix}*" if prefix is not None else "*"
            keys = []
            async for key in redis_client.scan_iter(match=pattern):
                keys.append(key.decode("utf-8") if isinstance(key, bytes) else key)
            return keys
        except Exception:
            if self.raise_on_error:
                raise
            return []

    async def close(self):
        """Close Redis connection"""
        if self._redis:
            await self._redis.aclose()
            self._redis = None


def start_redis_server(
    port=6379,
    redis_server_path=None,
    runtime="apptainer",
    foreground=False,
    log=None,
    image="redis_7-alpine.sif",
    image_source="docker://redis:7-alpine",
    pull_image=True,
    workdir=None,
    bind=None,
    env=None,
    apptainer_cleanenv=True,
    apptainer_executable="apptainer",
    apptainer_extra_args=None,
    advertise_host=None,
    coordination_dir=None,
    head_registry=None,
    coordination_ttl_seconds=60.0,
    coordination_refresh_interval=30.0,
    coordination_startup_timeout=600.0,
    coordination_healthcheck_timeout=2.0,
    persistence=True,
    data_dir=None,
    appendfsync="everysec",
):
    """
    Start a Redis server instance.
    
    Args:
        port: Port number for Redis server
        redis_server_path: Optional path to redis-server binary. If not provided,
                          will check REDIS_SERVER_PATH env var, then search PATH.
        runtime: Launch runtime ("local" or "apptainer")
        foreground: Run Redis attached to this process instead of starting it in
                    the background and returning.
        log: Optional file path to append the Redis URL to. Missing parent
             directories and the file are created.
        image: Apptainer image path when runtime="apptainer"
        image_source: Optional source used by "apptainer pull"
        pull_image: Pull image_source before launch when provided
        bind: Apptainer bind mount(s), e.g. /host:/container
        env: Apptainer environment entry or entries as KEY=VALUE
        advertise_host: Hostname placed in the published Redis URL.
        coordination_dir: Backward-compatible name for ``head_registry``.
        head_registry: Stable endpoint registry. Accepts ``file://``,
            ``sqlite://``, or ``redis://``. A raw absolute path remains a
            compatibility alias for ``file://``. When omitted, a unique
            filesystem registry is created under the temporary directory.
        coordination_ttl_seconds: Lifetime of the Redis endpoint record.
        coordination_refresh_interval: Seconds between health checks and refreshes.
        coordination_startup_timeout: Maximum wait for Redis to become healthy.
        coordination_healthcheck_timeout: Timeout for each Redis PING.
        persistence: Enable append-only-file persistence.
        data_dir: Redis persistence directory. Defaults to ``redis-data`` under
            the head registry when it is file- or SQLite-backed. For a
            Redis-backed head, a temporary directory is used unless supplied.
        appendfsync: Redis AOF fsync policy: always, everysec, or no.
    
    Returns:
        Redis URL string
    """
    launch_runtime = build_runtime(
        runtime=runtime,
        image=image,
        image_source=image_source,
        pull_image=pull_image,
        workdir=workdir,
        bind=bind,
        env=env,
        apptainer_nv=False,
        apptainer_cleanenv=apptainer_cleanenv,
        apptainer_executable=apptainer_executable,
        apptainer_extra_args=apptainer_extra_args,
    )

    if launch_runtime.name == "local":
        # Find redis-server binary
        if redis_server_path is None:
            # Check environment variable first (for custom installations)
            redis_server_path = os.environ.get('REDIS_SERVER_PATH')

            if redis_server_path:
                # Expand ~ in the path
                redis_server_path = os.path.expanduser(redis_server_path)
            else:
                # Fall back to searching in PATH
                redis_server_path = shutil.which('redis-server')

            if redis_server_path is None:
                raise RuntimeError(
                    "redis-server not found. Please either:\n"
                    "  1. Install redis-server and ensure it's in your PATH, or\n"
                    "  2. Set REDIS_SERVER_PATH environment variable to the binary path, or\n"
                    "  3. Pass redis_server_path parameter to this function"
                )
        else:
            # Expand ~ in provided path
            redis_server_path = os.path.expanduser(redis_server_path)
        server_command = [redis_server_path]
    else:
        launch_runtime.prepare()
        server_command = ["redis-server"]
    
    if coordination_ttl_seconds <= coordination_refresh_interval:
        raise ValueError(
            "coordination_ttl_seconds must be greater than "
            "coordination_refresh_interval"
        )
    if coordination_refresh_interval <= 0:
        raise ValueError("coordination_refresh_interval must be positive")
    if coordination_startup_timeout <= 0:
        raise ValueError("coordination_startup_timeout must be positive")
    if coordination_healthcheck_timeout <= 0:
        raise ValueError("coordination_healthcheck_timeout must be positive")

    resolved_host = (
        advertise_host
        or os.getenv("BEAKER_NODE_HOSTNAME")
        or socket.getfqdn()
    ).strip()
    if not resolved_host:
        raise ValueError("advertise_host must be non-empty")
    url = f"redis://{resolved_host}:{port}"

    if coordination_dir is not None and head_registry is not None:
        raise ValueError("supply only one of coordination_dir or head_registry")
    configured_head_registry = head_registry or coordination_dir
    if configured_head_registry is None:
        coordination_path = Path(
            tempfile.mkdtemp(prefix="literegistry-redis-coordination-")
        )
        coordination_location = normalize_endpoint_registry(coordination_path)
    else:
        if not str(configured_head_registry).strip():
            raise ValueError("head_registry must be non-empty when supplied")
        coordination_location = normalize_endpoint_registry(
            configured_head_registry
        )
        coordination_path = endpoint_registry_filesystem_path(
            coordination_location
        )
        if coordination_path is not None:
            coordination_path.mkdir(parents=True, exist_ok=True)

    if appendfsync not in {"always", "everysec", "no"}:
        raise ValueError("appendfsync must be one of: always, everysec, no")
    resolved_data_dir = None
    if persistence:
        if data_dir is not None:
            default_data_dir = Path(data_dir)
        elif coordination_path is not None:
            default_data_dir = coordination_path / "redis-data"
        elif coordination_location.startswith("sqlite:"):
            default_data_dir = Path(
                str(sqlite_registry_path(coordination_location)) + ".redis-data"
            )
        else:
            default_data_dir = Path(
                tempfile.mkdtemp(prefix="literegistry-redis-data-")
            )
            logger.warning(
                "Redis-backed head registry has no filesystem location for AOF "
                "data; using temporary directory %s. Pass data_dir on shared "
                "storage for resume persistence.",
                default_data_dir,
            )
        resolved_data_dir = Path(
            default_data_dir
        ).expanduser().absolute()
        resolved_data_dir.mkdir(parents=True, exist_ok=True)

    persistence_args = (
        [
            "--save", "",
            "--appendonly", "yes",
            "--appendfsync", appendfsync,
            "--dir", str(resolved_data_dir),
        ]
        if persistence
        else ["--save", "", "--appendonly", "no"]
    )
    command = launch_runtime.build_command([
        *server_command,
        *persistence_args,
        "--port", str(port),
        "--protected-mode", "no",
    ])

    print(f"LITEREGISTRY_HEAD_REGISTRY={coordination_location}", flush=True)
    print(f"LITEREGISTRY_COORDINATION_DIR={coordination_location}", flush=True)
    print("LITEREGISTRY_REDIS_ENDPOINT_NAME=redis", flush=True)
    if resolved_data_dir is not None:
        print(f"LITEREGISTRY_REDIS_DATA_DIR={resolved_data_dir}", flush=True)
    print(f"REDIS_URL={url}", flush=True)
    if log is not None:
        log_path = os.path.expanduser(log)
        log_dir = os.path.dirname(log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        with open(log_path, "a") as log_file:
            log_file.write(f"{url}\n")

    if foreground:
        print(f"Redis server running with URL: {url}", flush=True)
        supervise_endpoint(
            root=coordination_location,
            name="redis",
            uri=url,
            command_json=command,
            healthcheck="redis",
            startup_timeout=coordination_startup_timeout,
            healthcheck_timeout=coordination_healthcheck_timeout,
            ttl_seconds=coordination_ttl_seconds,
            refresh_interval=coordination_refresh_interval,
        )
    else:
        supervisor_command = [
            sys.executable,
            "-m",
            "literegistry.coop.endpoints",
            "run",
            f"--root={coordination_location}",
            "--name=redis",
            f"--uri={url}",
            "--healthcheck=redis",
            f"--startup_timeout={coordination_startup_timeout}",
            f"--healthcheck_timeout={coordination_healthcheck_timeout}",
            f"--ttl_seconds={coordination_ttl_seconds}",
            f"--refresh_interval={coordination_refresh_interval}",
            f"--command_json={json.dumps(command, separators=(',', ':'))}",
        ]
        supervisor = subprocess.Popen(supervisor_command, start_new_session=True)
        try:
            asyncio.run(
                wait_for_endpoint(
                    coordination_location,
                    "redis",
                    timeout=coordination_startup_timeout,
                    poll_interval=min(0.25, coordination_refresh_interval),
                    healthcheck="redis",
                    healthcheck_timeout=coordination_healthcheck_timeout,
                )
            )
        except BaseException:
            supervisor.terminate()
            raise

    return url

# Usage Example
async def main_async(
    port=6379,
    runtime="apptainer",
    foreground=False,
    log=None,
    image="redis_7-alpine.sif",
    image_source="docker://redis:7-alpine",
    pull_image=True,
    redis_server_path=None,
    workdir=None,
    bind=None,
    env=None,
    apptainer_cleanenv=True,
    apptainer_executable="apptainer",
    apptainer_extra_args=None,
    advertise_host=None,
    coordination_dir=None,
    head_registry=None,
    coordination_ttl_seconds=60.0,
    coordination_refresh_interval=30.0,
    coordination_startup_timeout=600.0,
    coordination_healthcheck_timeout=2.0,
    persistence=True,
    data_dir=None,
    appendfsync="everysec",
):
    # FileSystem Example
    #fs_store = FileSystemKVStore()
    #await fs_store.set("test1.txt", "Hello FS!")
    #await fs_store.set("test2.txt", "World FS!")
    #print(await fs_store.keys())  # ['test1.txt', 'test2.txt']
    url = await asyncio.to_thread(
        start_redis_server,
        port=port,
        redis_server_path=redis_server_path,
        runtime=runtime,
        foreground=foreground,
        log=log,
        image=image,
        image_source=image_source,
        pull_image=pull_image,
        workdir=workdir,
        bind=bind,
        env=env,
        apptainer_cleanenv=apptainer_cleanenv,
        apptainer_executable=apptainer_executable,
        apptainer_extra_args=apptainer_extra_args,
        advertise_host=advertise_host,
        coordination_dir=coordination_dir,
        head_registry=head_registry,
        coordination_ttl_seconds=coordination_ttl_seconds,
        coordination_refresh_interval=coordination_refresh_interval,
        coordination_startup_timeout=coordination_startup_timeout,
        coordination_healthcheck_timeout=coordination_healthcheck_timeout,
        persistence=persistence,
        data_dir=data_dir,
        appendfsync=appendfsync,
    )
    print(f"Redis server started with URL: {url}")
    
def main(
    port=6379,
    runtime="apptainer",
    foreground=False,
    log=None,
    image="redis_7-alpine.sif",
    image_source="docker://redis:7-alpine",
    pull_image=True,
    redis_server_path=None,
    workdir=None,
    bind=None,
    env=None,
    apptainer_cleanenv=True,
    apptainer_executable="apptainer",
    apptainer_extra_args=None,
    advertise_host=None,
    coordination_dir=None,
    head_registry=None,
    coordination_ttl_seconds=60.0,
    coordination_refresh_interval=30.0,
    coordination_startup_timeout=600.0,
    coordination_healthcheck_timeout=2.0,
    persistence=True,
    data_dir=None,
    appendfsync="everysec",
):
    asyncio.run(
        main_async(
            port=port,
            runtime=runtime,
            foreground=foreground,
            log=log,
            image=image,
            image_source=image_source,
            pull_image=pull_image,
            redis_server_path=redis_server_path,
            workdir=workdir,
            bind=bind,
            env=env,
            apptainer_cleanenv=apptainer_cleanenv,
            apptainer_executable=apptainer_executable,
            apptainer_extra_args=apptainer_extra_args,
            advertise_host=advertise_host,
            coordination_dir=coordination_dir,
            head_registry=head_registry,
            coordination_ttl_seconds=coordination_ttl_seconds,
            coordination_refresh_interval=coordination_refresh_interval,
            coordination_startup_timeout=coordination_startup_timeout,
            coordination_healthcheck_timeout=coordination_healthcheck_timeout,
            persistence=persistence,
            data_dir=data_dir,
            appendfsync=appendfsync,
        )
    )


if __name__ == "__main__":
   fire.Fire(main)
