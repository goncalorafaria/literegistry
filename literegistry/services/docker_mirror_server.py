"""Supervise and register a Docker Hub pull-through registry mirror.

The process owns three pieces of one service lifecycle:

* a Distribution ``registry`` child process;
* health checks against a real image manifest;
* a LiteRegistry registration whose URI is the Registry V2 endpoint itself.

An optional cache warmer runs in the background after the mirror is healthy.
The mirror remains available if warming fails; the failure is reported in the
heartbeat data instead of taking down serving.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import json
import logging
import os
from pathlib import Path
import signal
import socket
import sys
import time
from typing import Any, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import fire
import redis.asyncio as redis

from literegistry.registry import ServerRegistry


logger = logging.getLogger(__name__)
MANIFEST_ACCEPT = ", ".join(
    [
        "application/vnd.oci.image.index.v1+json",
        "application/vnd.docker.distribution.manifest.list.v2+json",
        "application/vnd.oci.image.manifest.v1+json",
        "application/vnd.docker.distribution.manifest.v2+json",
    ]
)


def dockerhub_repo_and_reference(image: str) -> tuple[str, str]:
    """Return the Registry V2 repository and reference for a Docker Hub image."""
    value = image.strip()
    if not value:
        raise ValueError("health_image must be non-empty")
    first, separator, remainder = value.partition("/")
    if separator and ("." in first or ":" in first or first == "localhost"):
        if first not in {"docker.io", "registry-1.docker.io", "index.docker.io"}:
            raise ValueError("health_image must refer to a Docker Hub image")
        value = remainder
    if "/" not in value:
        value = f"library/{value}"
    if "@" in value:
        return tuple(value.rsplit("@", 1))  # type: ignore[return-value]
    last_slash = value.rfind("/")
    last_colon = value.rfind(":")
    if last_colon > last_slash:
        return value[:last_colon], value[last_colon + 1 :]
    return value, "latest"


def build_distribution_config(
    *,
    host: str,
    port: int,
    storage_root: str,
    upstream_url: str,
    docker_hub_username: Optional[str] = None,
    docker_hub_token: Optional[str] = None,
) -> str:
    """Build a Distribution configuration without logging upstream secrets."""
    if bool(docker_hub_username) != bool(docker_hub_token):
        raise ValueError(
            "docker_hub_username and docker_hub_token must be supplied together"
        )
    lines = [
        "version: 0.1",
        "log:",
        "  fields:",
        "    service: literegistry-docker-mirror",
        "storage:",
        "  filesystem:",
        f"    rootdirectory: {json.dumps(storage_root)}",
        "http:",
        f"  addr: {json.dumps(f'{host}:{port}')}",
        "proxy:",
        f"  remoteurl: {json.dumps(upstream_url)}",
    ]
    if docker_hub_username and docker_hub_token:
        lines.extend(
            [
                f"  username: {json.dumps(docker_hub_username)}",
                f"  password: {json.dumps(docker_hub_token)}",
            ]
        )
    return "\n".join(lines) + "\n"


def write_distribution_config(path: str, contents: str) -> None:
    config_path = Path(path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(contents, encoding="utf-8")
    config_path.chmod(0o600)


def _probe_manifest(url: str, timeout: float) -> dict[str, Any]:
    request = Request(url, headers={"Accept": MANIFEST_ACCEPT})
    with urlopen(request, timeout=timeout) as response:
        if response.status != 200:
            raise RuntimeError(f"mirror returned HTTP {response.status}")
        payload = json.loads(response.read().decode("utf-8"))
        if not isinstance(payload, dict) or payload.get("schemaVersion") != 2:
            raise RuntimeError("mirror returned an invalid Registry V2 manifest")
        return {
            "status": "healthy",
            "checked_at": time.time(),
            "manifest_url": url,
            "digest": response.headers.get("Docker-Content-Digest", ""),
            "media_type": payload.get("mediaType", ""),
        }


async def probe_manifest(url: str, timeout: float) -> dict[str, Any]:
    """Probe a real manifest so corrupt-but-live mirrors are not advertised."""
    try:
        return await asyncio.to_thread(_probe_manifest, url, timeout)
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError, RuntimeError) as exc:
        return {
            "status": "unhealthy",
            "checked_at": time.time(),
            "manifest_url": url,
            "error": str(exc),
        }


class AsyncRedisStore:
    """KeyValueStore-compatible async Redis adapter for mirror registration."""

    def __init__(self, url: str) -> None:
        self.url = url.rstrip("\\")
        self.client = redis.from_url(self.url, decode_responses=False)

    async def ping(self) -> bool:
        return bool(await self.client.ping())

    async def get(self, key: str):
        return await self.client.get(key)

    async def set(self, key: str, value, ttl_seconds=None) -> bool:
        options: dict[str, int] = {}
        if ttl_seconds is not None:
            options["px"] = max(1, int(float(ttl_seconds) * 1000))
        return bool(await self.client.set(key, value, **options))

    async def delete(self, key: str) -> bool:
        return bool(await self.client.delete(key))

    async def exists(self, key: str) -> bool:
        return bool(await self.client.exists(key))

    async def keys(self, prefix: Optional[str] = None) -> list[str]:
        pattern = f"{prefix or ''}*"
        return [
            key.decode() if isinstance(key, bytes) else key
            async for key in self.client.scan_iter(match=pattern)
        ]

    async def close(self) -> None:
        await self.client.aclose()


@dataclass(frozen=True)
class DockerMirrorConfig:
    registry_url: str
    advertise_host: str
    advertise_port: int = 5000
    host: str = "0.0.0.0"
    port: int = 5000
    service: str = "docker-mirror"
    instance_id: str = "docker-mirror-1"
    registry_binary: str = "registry"
    distribution_config: str = "/run/literegistry/config.yml"
    storage_root: str = "/var/lib/registry"
    upstream_url: str = "https://registry-1.docker.io"
    docker_hub_username: Optional[str] = None
    docker_hub_token: Optional[str] = None
    heartbeat_interval: float = 10.0
    startup_timeout: float = 180.0
    health_timeout: float = 30.0
    health_failure_threshold: int = 3
    health_image: str = "docker.io/library/alpine:3.20"
    warm_dataset: Optional[str] = None
    warm_revision: Optional[str] = None
    warm_workers: int = 8
    warm_platform: str = "linux/amd64"
    warm_tool_configs: Optional[str] = None
    warm_images_file: Optional[str] = None
    warm_images: tuple[str, ...] = field(default_factory=tuple)

    def validate(self) -> None:
        if not self.registry_url:
            raise ValueError("registry_url must be non-empty")
        if not self.advertise_host:
            raise ValueError("advertise_host must be non-empty")
        for name, port in {
            "port": self.port,
            "advertise_port": self.advertise_port,
        }.items():
            if not 1 <= port <= 65535:
                raise ValueError(f"{name} must be between 1 and 65535")
        if self.heartbeat_interval <= 0 or self.startup_timeout <= 0:
            raise ValueError("heartbeat_interval and startup_timeout must be positive")
        if self.health_timeout <= 0 or self.health_failure_threshold < 1:
            raise ValueError("health probe settings must be positive")
        if self.warm_workers < 1:
            raise ValueError("warm_workers must be at least one")
        if bool(self.docker_hub_username) != bool(self.docker_hub_token):
            raise ValueError(
                "docker_hub_username and docker_hub_token must be supplied together"
            )

    @property
    def advertised_url(self) -> str:
        return f"http://{self.advertise_host}"

    @property
    def local_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    @property
    def health_repository(self) -> str:
        repository, _reference = dockerhub_repo_and_reference(self.health_image)
        return repository

    @property
    def health_manifest_path(self) -> str:
        _repository, reference = dockerhub_repo_and_reference(self.health_image)
        return f"/v2/{self.health_repository}/manifests/{reference}"

    @property
    def health_url(self) -> str:
        return f"{self.local_url}{self.health_manifest_path}"

    def health_url_for_digest(self, digest: str) -> str:
        return f"{self.local_url}/v2/{self.health_repository}/manifests/{digest}"

    @property
    def warming_requested(self) -> bool:
        return bool(
            self.warm_dataset
            or self.warm_tool_configs
            or self.warm_images_file
            or self.warm_images
        )


class MirrorRegistration:
    """Publish only healthy mirrors and attach health/warmup heartbeat data."""

    def __init__(
        self,
        config: DockerMirrorConfig,
        *,
        store: Optional[AsyncRedisStore] = None,
        registry: Optional[ServerRegistry] = None,
    ) -> None:
        self.config = config
        self.store = store or AsyncRedisStore(config.registry_url)
        self.registry = registry or ServerRegistry(store=self.store)
        self.registered = False

    def metadata(self) -> dict[str, Any]:
        return {
            "model_path": self.config.service,
            "backend": "docker-registry-v2-pull-through-cache",
            "instance_id": self.config.instance_id,
            "protocol": "docker-registry-v2",
            "upstream": self.config.upstream_url,
            "upstream_authenticated": bool(self.config.docker_hub_username),
            "health_endpoint": self.config.health_manifest_path,
            "warmup": {
                "requested": self.config.warming_requested,
                "dataset": self.config.warm_dataset,
                "images_file": self.config.warm_images_file,
                "platform": self.config.warm_platform,
                "workers": self.config.warm_workers,
            },
            "authentication": {"type": "none"},
        }

    async def connect(self) -> None:
        if not await self.store.ping():
            raise RuntimeError(f"Redis PING failed for {self.config.registry_url}")

    async def healthy(self, data: dict[str, Any]) -> None:
        if self.registered:
            await self.registry.heartbeat(
                self.config.advertised_url,
                self.config.advertise_port,
                data=data,
            )
            return
        await self.registry.register_server(
            self.config.advertised_url,
            self.config.advertise_port,
            self.metadata(),
        )
        await self.registry.heartbeat(
            self.config.advertised_url,
            self.config.advertise_port,
            data=data,
        )
        self.registered = True
        logger.info(
            "Registered docker mirror server_id=%s uri=%s:%s registry=%s",
            self.registry.server_id,
            self.config.advertised_url,
            self.config.advertise_port,
            self.config.registry_url,
        )

    async def unhealthy(self) -> None:
        if self.registered:
            await self.registry.deregister()
            self.registered = False
            logger.warning("Deregistered unhealthy docker mirror")

    async def close(self) -> None:
        try:
            await self.unhealthy()
        finally:
            await self.store.close()


class DockerMirrorSupervisor:
    def __init__(
        self,
        config: DockerMirrorConfig,
        *,
        registration: Optional[MirrorRegistration] = None,
    ) -> None:
        config.validate()
        self.config = config
        self.registration = registration or MirrorRegistration(config)
        self.stop_event = asyncio.Event()
        self.warmup: dict[str, Any] = {
            "status": "pending" if config.warming_requested else "disabled"
        }

    def registry_command(self) -> list[str]:
        return [
            self.config.registry_binary,
            "serve",
            self.config.distribution_config,
        ]

    def warmer_command(self) -> list[str]:
        command = [
            sys.executable,
            "-m",
            "literegistry.services.docker_mirror_warmup",
            "--mirror",
            f"127.0.0.1:{self.config.port}",
            "--workers",
            str(self.config.warm_workers),
            "--platform",
            self.config.warm_platform,
        ]
        if self.config.warm_dataset:
            command.extend(["--dataset", self.config.warm_dataset])
        if self.config.warm_revision:
            command.extend(["--revision", self.config.warm_revision])
        if self.config.warm_tool_configs:
            command.extend(["--tool-configs", self.config.warm_tool_configs])
        if self.config.warm_images_file:
            command.extend(["--images-file", self.config.warm_images_file])
        if self.config.warm_images:
            command.extend(["--image", json.dumps(self.config.warm_images)])
        return command

    def heartbeat_data(self, health: dict[str, Any]) -> dict[str, Any]:
        return {"health": health, "warmup": dict(self.warmup)}

    async def _wait_until_healthy(
        self, process: asyncio.subprocess.Process
    ) -> dict[str, Any]:
        deadline = asyncio.get_running_loop().time() + self.config.startup_timeout
        last_health: dict[str, Any] = {"status": "starting"}
        while asyncio.get_running_loop().time() < deadline:
            if process.returncode is not None:
                raise RuntimeError(
                    f"registry exited during startup with code {process.returncode}"
                )
            last_health = await probe_manifest(
                self.config.health_url,
                self.config.health_timeout,
            )
            if last_health["status"] == "healthy":
                return last_health
            await asyncio.sleep(1)
        raise TimeoutError(
            f"registry did not pass manifest health within {self.config.startup_timeout}s: "
            f"{last_health.get('error', 'unknown error')}"
        )

    async def _run_warmer(self) -> None:
        self.warmup = {"status": "running", "started_at": time.time()}
        process = await asyncio.create_subprocess_exec(*self.warmer_command())
        try:
            returncode = await process.wait()
        except asyncio.CancelledError:
            if process.returncode is None:
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), timeout=10)
                except asyncio.TimeoutError:
                    process.kill()
                    await process.wait()
            raise
        self.warmup = {
            "status": "succeeded" if returncode == 0 else "failed",
            "finished_at": time.time(),
            "exit_code": returncode,
        }
        if returncode:
            logger.error("Docker mirror warmup exited with code %s", returncode)
        else:
            logger.info("Docker mirror warmup completed")

    async def _monitor(self, probe_url: str) -> None:
        failures = 0
        while not self.stop_event.is_set():
            health = await probe_manifest(
                probe_url,
                self.config.health_timeout,
            )
            if health["status"] == "healthy":
                failures = 0
                await self.registration.healthy(self.heartbeat_data(health))
            else:
                failures += 1
                logger.warning(
                    "Docker mirror manifest probe failed (%s/%s): %s",
                    failures,
                    self.config.health_failure_threshold,
                    health.get("error"),
                )
                if failures >= self.config.health_failure_threshold:
                    await self.registration.unhealthy()
            try:
                await asyncio.wait_for(
                    self.stop_event.wait(), timeout=self.config.heartbeat_interval
                )
            except asyncio.TimeoutError:
                pass

    async def _terminate_registry(self, process: asyncio.subprocess.Process) -> None:
        if process.returncode is not None:
            return
        process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=20)
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()

    async def run(self) -> None:
        Path(self.config.storage_root).mkdir(parents=True, exist_ok=True)
        write_distribution_config(
            self.config.distribution_config,
            build_distribution_config(
                host=self.config.host,
                port=self.config.port,
                storage_root=self.config.storage_root,
                upstream_url=self.config.upstream_url,
                docker_hub_username=self.config.docker_hub_username,
                docker_hub_token=self.config.docker_hub_token,
            ),
        )
        await self.registration.connect()
        process = await asyncio.create_subprocess_exec(*self.registry_command())
        warm_task: Optional[asyncio.Task] = None
        monitor_task: Optional[asyncio.Task] = None
        process_task: Optional[asyncio.Task] = None
        stop_task: Optional[asyncio.Task] = None
        try:
            health = await self._wait_until_healthy(process)
            await self.registration.healthy(self.heartbeat_data(health))
            if self.config.warming_requested:
                warm_task = asyncio.create_task(self._run_warmer())
            # The startup tag request verifies upstream pull-through. Steady-state
            # probes use its immutable cached digest, avoiding needless Docker Hub
            # tag revalidation on every heartbeat.
            probe_url = self.config.health_url
            if health.get("digest"):
                probe_url = self.config.health_url_for_digest(health["digest"])
            monitor_task = asyncio.create_task(self._monitor(probe_url))
            process_task = asyncio.create_task(process.wait())
            stop_task = asyncio.create_task(self.stop_event.wait())
            done, _pending = await asyncio.wait(
                {process_task, stop_task}, return_when=asyncio.FIRST_COMPLETED
            )
            if process_task in done and not self.stop_event.is_set():
                raise RuntimeError(
                    f"registry exited unexpectedly with code {process_task.result()}"
                )
        finally:
            self.stop_event.set()
            for task in (monitor_task, warm_task, process_task, stop_task):
                if task is not None and not task.done():
                    task.cancel()
            await asyncio.gather(
                *(task for task in (monitor_task, warm_task, process_task, stop_task) if task),
                return_exceptions=True,
            )
            await self.registration.close()
            await self._terminate_registry(process)


def _split_csv(value: Optional[str]) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(item.strip() for item in value.split(",") if item.strip())


def main(
    registry: str = os.environ.get(
        "LITEREGISTRY_URL", "redis://127.0.0.1:6379"
    ),
    host: str = os.environ.get("DOCKER_MIRROR_HOST", "0.0.0.0"),
    port: int = int(os.environ.get("DOCKER_MIRROR_PORT", "5000")),
    advertise_host: str = os.environ.get(
        "DOCKER_MIRROR_ADVERTISE_HOST", socket.getfqdn()
    ),
    advertise_port: int = int(
        os.environ.get("DOCKER_MIRROR_ADVERTISE_PORT", "5000")
    ),
    service: str = os.environ.get("DOCKER_MIRROR_SERVICE", "docker-mirror"),
    instance_id: str = os.environ.get(
        "DOCKER_MIRROR_INSTANCE_ID", "docker-mirror-1"
    ),
    registry_binary: str = os.environ.get("DISTRIBUTION_BINARY", "registry"),
    distribution_config: str = os.environ.get(
        "DISTRIBUTION_CONFIG", "/run/literegistry/config.yml"
    ),
    storage_root: str = os.environ.get(
        "DOCKER_MIRROR_STORAGE_ROOT", "/var/lib/registry"
    ),
    upstream_url: str = os.environ.get(
        "DOCKER_MIRROR_UPSTREAM", "https://registry-1.docker.io"
    ),
    docker_hub_username: Optional[str] = os.environ.get("DOCKER_HUB_USERNAME"),
    docker_hub_token: Optional[str] = os.environ.get("DOCKER_HUB_TOKEN")
    or os.environ.get("DOCKER_PAT")
    or os.environ.get("PERSONAL_ACCESS_TOKEN"),
    heartbeat_interval: float = float(
        os.environ.get("DOCKER_MIRROR_HEARTBEAT_INTERVAL", "10")
    ),
    startup_timeout: float = float(
        os.environ.get("DOCKER_MIRROR_STARTUP_TIMEOUT", "180")
    ),
    health_timeout: float = float(
        os.environ.get("DOCKER_MIRROR_HEALTH_TIMEOUT", "30")
    ),
    health_failure_threshold: int = int(
        os.environ.get("DOCKER_MIRROR_HEALTH_FAILURE_THRESHOLD", "3")
    ),
    health_image: str = os.environ.get(
        "DOCKER_MIRROR_HEALTH_IMAGE", "docker.io/library/alpine:3.20"
    ),
    warm_dataset: Optional[str] = os.environ.get("DOCKER_MIRROR_WARM_DATASET"),
    warm_revision: Optional[str] = os.environ.get("DOCKER_MIRROR_WARM_REVISION"),
    warm_workers: int = int(os.environ.get("DOCKER_MIRROR_WARM_WORKERS", "8")),
    warm_platform: str = os.environ.get(
        "DOCKER_MIRROR_WARM_PLATFORM", "linux/amd64"
    ),
    warm_tool_configs: Optional[str] = os.environ.get(
        "DOCKER_MIRROR_WARM_TOOL_CONFIGS"
    ),
    warm_images_file: Optional[str] = os.environ.get(
        "DOCKER_MIRROR_WARM_IMAGES_FILE"
    ),
    warm_images: Optional[str] = os.environ.get("DOCKER_MIRROR_WARM_IMAGES"),
    allow_non_loopback: bool = False,
) -> None:
    """Run one Docker Hub mirror, register health, and optionally warm it."""
    if host not in {"127.0.0.1", "localhost", "::1"} and not allow_non_loopback:
        raise ValueError(
            "the unauthenticated mirror must bind to loopback unless "
            "allow_non_loopback is explicitly enabled"
        )
    logging.basicConfig(level=logging.INFO)
    config = DockerMirrorConfig(
        registry_url=registry.rstrip("\\"),
        advertise_host=advertise_host,
        advertise_port=advertise_port,
        host=host,
        port=port,
        service=service,
        instance_id=instance_id,
        registry_binary=registry_binary,
        distribution_config=distribution_config,
        storage_root=storage_root,
        upstream_url=upstream_url,
        docker_hub_username=docker_hub_username,
        docker_hub_token=docker_hub_token,
        heartbeat_interval=heartbeat_interval,
        startup_timeout=startup_timeout,
        health_timeout=health_timeout,
        health_failure_threshold=health_failure_threshold,
        health_image=health_image,
        warm_dataset=warm_dataset,
        warm_revision=warm_revision,
        warm_workers=warm_workers,
        warm_platform=warm_platform,
        warm_tool_configs=warm_tool_configs,
        warm_images_file=warm_images_file,
        warm_images=_split_csv(warm_images),
    )
    supervisor = DockerMirrorSupervisor(config)

    async def run() -> None:
        loop = asyncio.get_running_loop()
        for signum in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(signum, supervisor.stop_event.set)
            except NotImplementedError:
                pass
        await supervisor.run()

    asyncio.run(run())


if __name__ == "__main__":
    fire.Fire(main)
