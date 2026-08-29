"""The canonical Redis-registered Podman affinity HTTP server."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import logging
import os
import socket
from typing import Any, Optional

import fire
import redis.asyncio as redis
import uvicorn

from literegistry.services.podman import (
    PodmanAffinityConfig,
    PodmanAffinityService,
    PodmanSessionBackend,
    create_app,
)
from literegistry.registry import ServerRegistry


logger = logging.getLogger(__name__)
DEFAULT_REGISTRY_URL = "redis://127.0.0.1:6379"


class AsyncRedisStore:
    """Small KeyValueStore-compatible adapter used by ServerRegistry."""

    def __init__(self, url: str) -> None:
        self.url = url.rstrip("\\")
        self.client = redis.from_url(self.url, decode_responses=False)

    async def ping(self) -> bool:
        return bool(await self.client.ping())

    async def get(self, key: str):
        return await self.client.get(key)

    async def set(self, key: str, value, ttl_seconds=None) -> bool:
        options = {}
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


class RedisRegistration:
    """Register, heartbeat, and deregister one Podman HTTP replica."""

    def __init__(
        self,
        *,
        registry_url: str,
        advertise_host: str,
        advertise_port: int,
        instance_id: str,
        image: str,
        registry_mirror: Optional[str] = None,
        heartbeat_interval: float = 10.0,
        store: Optional[AsyncRedisStore] = None,
        registry: Optional[ServerRegistry] = None,
    ) -> None:
        self.store = store or AsyncRedisStore(registry_url)
        self.registry = registry or ServerRegistry(store=self.store)
        self.registry_url = registry_url.rstrip("\\")
        self.url = f"http://{advertise_host}"
        self.port = advertise_port
        self.instance_id = instance_id
        self.image = image
        self.registry_mirror = registry_mirror
        self.heartbeat_interval = heartbeat_interval
        self._heartbeat_task: Optional[asyncio.Task] = None

    def metadata(self) -> dict[str, Any]:
        return {
            "model_path": "podman",
            "backend": "podman-affinity",
            "instance_id": self.instance_id,
            "image": self.image,
            "registry_mirror": self.registry_mirror,
            "affinity": {
                "enabled": True,
                "handshake_endpoint": "handshake",
                "command_endpoint": "podman",
                "close_endpoint": "close",
                "id_field": "affinity_id",
            },
            "authentication": {"type": "none"},
        }

    async def start(self) -> None:
        if not await self.store.ping():
            raise RuntimeError(f"Redis PING failed for {self.registry_url}")
        await self.registry.register_server(self.url, self.port, self.metadata())
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        logger.info(
            "Registered server_id=%s uri=%s:%s registry=%s",
            self.registry.server_id,
            self.url,
            self.port,
            self.registry_url,
        )

    async def _heartbeat_loop(self) -> None:
        while True:
            await asyncio.sleep(self.heartbeat_interval)
            try:
                await self.registry.heartbeat(self.url, self.port)
            except Exception:
                logger.exception("Redis registry heartbeat failed")

    async def stop(self) -> None:
        if self._heartbeat_task is not None:
            self._heartbeat_task.cancel()
            await asyncio.gather(self._heartbeat_task, return_exceptions=True)
            self._heartbeat_task = None
        try:
            await self.registry.deregister()
        finally:
            await self.store.close()


def build_registered_app(
    config: PodmanAffinityConfig,
    registration: RedisRegistration,
):
    backend = PodmanSessionBackend(config)
    service = PodmanAffinityService(backend)

    @asynccontextmanager
    async def lifespan(app):
        await backend.cleanup()
        await registration.start()
        try:
            yield
        finally:
            await registration.stop()
            await backend.cleanup()

    app = create_app(service, None, lifespan=lifespan)
    app.state.redis_registration = registration
    return app


def main(
    host: str = os.environ.get("PODMAN_AFFINITY_HOST", "127.0.0.1"),
    port: int = int(os.environ.get("PODMAN_AFFINITY_PORT", "8091")),
    advertise_host: str = os.environ.get(
        "PODMAN_AFFINITY_ADVERTISE_HOST", socket.getfqdn()
    ),
    advertise_port: int = int(
        os.environ.get("PODMAN_AFFINITY_ADVERTISE_PORT", "8091")
    ),
    registry: str = os.environ.get("PODMAN_AFFINITY_REGISTRY", DEFAULT_REGISTRY_URL),
    image: str = os.environ.get(
        "PODMAN_AFFINITY_IMAGE", "docker.io/library/ubuntu:24.04"
    ),
    network: str = os.environ.get("PODMAN_AFFINITY_NETWORK", "none"),
    instance_id: str = os.environ.get(
        "PODMAN_AFFINITY_INSTANCE_ID", "podman-affinity-1"
    ),
    storage_driver: str = os.environ.get("PODMAN_STORAGE_DRIVER", "vfs"),
    registry_mirror: Optional[str] = os.environ.get("PODMAN_REGISTRY_MIRROR"),
    heartbeat_interval: float = float(
        os.environ.get("PODMAN_AFFINITY_HEARTBEAT_INTERVAL", "10")
    ),
    allow_non_loopback: bool = False,
) -> None:
    """Run the one-worker Podman HTTP server and register it in Redis.

    Args:
        host: Local HTTP bind address. Must remain loopback without authentication.
        port: Local HTTP listen port.
        advertise_host: Host stored in the Redis server record.
        advertise_port: Port stored in the Redis server record.
        registry: Redis registry URL.
        image: OCI image used for each affinity container.
        network: Podman network mode assigned to affinity containers.
        instance_id: Owner label and registry identity for this server.
        storage_driver: Podman storage driver in the outer Docker container.
        registry_mirror: Optional HTTP(S) gateway root used as the native
            docker.io pull-through mirror. Podman falls back to Docker Hub.
        heartbeat_interval: Seconds between Redis heartbeats.
        allow_non_loopback: Explicitly permit a managed cluster bind. This is
            required for Beaker host networking and must not be used casually.
    """
    if host not in {"127.0.0.1", "localhost", "::1"} and not allow_non_loopback:
        raise ValueError(
            "the unauthenticated Podman HTTP server must bind to loopback unless "
            "allow_non_loopback is explicitly enabled"
        )

    logging.basicConfig(level=logging.INFO)
    config = PodmanAffinityConfig(
        storage_driver=storage_driver,
        session_image=image,
        session_network=network,
        instance_id=instance_id,
        registry_mirror=registry_mirror,
    )
    registration = RedisRegistration(
        registry_url=registry,
        advertise_host=advertise_host,
        advertise_port=advertise_port,
        instance_id=instance_id,
        image=image,
        registry_mirror=registry_mirror,
        heartbeat_interval=heartbeat_interval,
    )
    uvicorn.run(
        build_registered_app(config, registration),
        host=host,
        port=port,
        workers=1,
    )


if __name__ == "__main__":
    fire.Fire(main)
