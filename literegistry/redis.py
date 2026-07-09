import abc
import asyncio
from typing import Optional, Union, List
import redis.asyncio as redis
from literegistry.kvstore import KeyValueStore
import socket
import time 
import subprocess
import shutil
import os
import fire
from literegistry.runtime import build_runtime


class RedisKVStore(KeyValueStore):
    """Redis-based key-value store"""
    #  http://klone-login01.hyak.local:8080/v1/models
    def __init__(self, url: str = "redis://klone-login01.hyak.local:6379", db: int = 0):
        """
        Initialize Redis KV store
        
        Args:
            url: Redis connection URL (e.g., "redis://localhost:6379", "redis://user:pass@host:port")
            db: Redis database number
        """
        self.url = url
        self.db = db
        self._redis = None

    async def _get_redis(self) -> redis.Redis:
        """Get Redis connection, creating it if necessary"""
        if self._redis is None:
            try:
                self._redis = redis.from_url(self.url, db=self.db, decode_responses=False)
                # Test the connection
                await self._redis.ping()
                print(f"Successfully connected to Redis at {self.url}")
            except Exception as e:
                print(f"Failed to connect to Redis at {self.url}: {e}")
                raise
        return self._redis

    async def get(self, key: str) -> Optional[bytes]:
        """Get value for a key from Redis"""
        redis_client = await self._get_redis()
        try:
            value = await redis_client.get(key)
            return value
        except Exception:
            return None

    async def set(self, key: str, value: Union[bytes, str]) -> bool:
        """Set value for a key in Redis"""
        redis_client = await self._get_redis()
        try:
            if isinstance(value, str):
                value = value.encode("utf-8")
            await redis_client.set(key, value)
            return True
        except Exception:
            return False

    async def delete(self, key: str) -> bool:
        """Delete a key from Redis"""
        redis_client = await self._get_redis()
        try:
            result = await redis_client.delete(key)
            return result > 0
        except Exception:
            return False

    async def exists(self, key: str) -> bool:
        """Check if key exists in Redis"""
        redis_client = await self._get_redis()
        try:
            result = await redis_client.exists(key)
            return result > 0
        except Exception:
            return False

    async def keys(self) -> List[str]:
        """Get a list of all keys in Redis"""
        redis_client = await self._get_redis()
        try:
            keys = await redis_client.keys("*")
            return [key.decode('utf-8') if isinstance(key, bytes) else key for key in keys]
        except Exception:
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
    
    # get hostname
    hostname = socket.gethostname()
    url = socket.getfqdn()

    url = f"redis://{url}:{port}"

    command = launch_runtime.build_command(
        [
            *server_command,
            "--save", "",
            "--appendonly", "no",
            "--port", str(port),
            "--protected-mode", "no",
        ]
    )

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
        subprocess.run(command, check=True)
    else:
        subprocess.Popen(command)

        # Give it a moment to start
        time.sleep(2)

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
):
    # FileSystem Example
    #fs_store = FileSystemKVStore()
    #await fs_store.set("test1.txt", "Hello FS!")
    #await fs_store.set("test2.txt", "World FS!")
    #print(await fs_store.keys())  # ['test1.txt', 'test2.txt']
    url = start_redis_server(
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
        )
    )


if __name__ == "__main__":
   fire.Fire(main)
