"""The canonical Redis-registered Podman affinity HTTP server."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import logging
import os
import socket
from typing import Any, Optional

import fire
import uvicorn

from literegistry import get_kvstore
from literegistry.kvstore import KeyValueStore
from literegistry.services.podman import (
    PodmanAffinityConfig,
    PodmanAffinityService,
    PodmanSessionBackend,
    create_app,
)
from literegistry.registry import ServerRegistry


logger = logging.getLogger(__name__)
DEFAULT_REGISTRY_URL = "redis://127.0.0.1:6379"


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
        session_limits: Optional[dict] = None,
        heartbeat_interval: float = 10.0,
        store: Optional[KeyValueStore] = None,
        registry: Optional[ServerRegistry] = None,
    ) -> None:
        self.store = store or get_kvstore(registry_url, raise_on_error=True)
        self.registry = registry or ServerRegistry(store=self.store)
        self.registry_url = registry_url.rstrip("\\")
        self.url = f"http://{advertise_host}"
        self.port = advertise_port
        self.instance_id = instance_id
        self.image = image
        self.registry_mirror = registry_mirror
        self.session_limits = session_limits or {}
        self.heartbeat_interval = heartbeat_interval
        self._heartbeat_task: Optional[asyncio.Task] = None

    def metadata(self) -> dict[str, Any]:
        return {
            "model_path": "podman",
            "backend": "podman-affinity",
            "instance_id": self.instance_id,
            "image": self.image,
            "registry_mirror": self.registry_mirror,
            "session_limits": self.session_limits,
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
        ping = getattr(self.store, "ping", None)
        if ping is not None and not await ping():
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
        janitor_task = asyncio.create_task(backend.janitor_loop())
        watchdog_task = asyncio.create_task(backend.resource_watchdog_loop())
        try:
            yield
        finally:
            janitor_task.cancel()
            watchdog_task.cancel()
            await asyncio.gather(
                janitor_task, watchdog_task, return_exceptions=True
            )
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
    max_sessions: Optional[int] = (
        int(os.environ["PODMAN_AFFINITY_MAX_SESSIONS"])
        if os.environ.get("PODMAN_AFFINITY_MAX_SESSIONS")
        else None
    ),
    session_memory: Optional[str] = os.environ.get("PODMAN_AFFINITY_SESSION_MEMORY"),
    session_pids_limit: Optional[int] = (
        int(os.environ["PODMAN_AFFINITY_SESSION_PIDS_LIMIT"])
        if os.environ.get("PODMAN_AFFINITY_SESSION_PIDS_LIMIT")
        else None
    ),
    session_idle_timeout: Optional[float] = (
        float(os.environ["PODMAN_AFFINITY_SESSION_IDLE_TIMEOUT"])
        if os.environ.get("PODMAN_AFFINITY_SESSION_IDLE_TIMEOUT")
        else None
    ),
    janitor_interval: float = float(
        os.environ.get("PODMAN_AFFINITY_JANITOR_INTERVAL", "300")
    ),
    resource_watchdog_interval: Optional[float] = (
        float(os.environ["PODMAN_AFFINITY_RESOURCE_WATCHDOG_INTERVAL"])
        if os.environ.get("PODMAN_AFFINITY_RESOURCE_WATCHDOG_INTERVAL")
        else 5.0
    ),
    image_prune_until: Optional[str] = os.environ.get("PODMAN_AFFINITY_IMAGE_PRUNE_UNTIL"),
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
        max_sessions: Maximum simultaneous session containers on this replica.
            Concurrent handshakes reserve capacity before starting Podman.
        session_memory: Optional per-container memory limit (e.g. ``4g``).
            Applied with an equal ``--memory-swap`` so a runaway session is
            OOM-killed instead of exhausting the replica host.
        session_pids_limit: Optional per-container pids limit (fork-bomb guard).
        session_idle_timeout: Seconds of inactivity after which a session
            container is reaped by the janitor. Unset disables reaping;
            clients that never call close then leak containers until restart.
        janitor_interval: Seconds between janitor sweeps.
        resource_watchdog_interval: Seconds between userspace memory/PID
            checks. Set to ``None`` to rely only on native cgroup enforcement.
        image_prune_until: Optional ``podman image prune --filter until=``
            value (e.g. ``24h``) applied on each janitor sweep.
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
        max_sessions=max_sessions,
        session_memory=session_memory,
        session_pids_limit=session_pids_limit,
        session_idle_timeout=session_idle_timeout,
        janitor_interval=janitor_interval,
        resource_watchdog_interval=resource_watchdog_interval,
        image_prune_until=image_prune_until,
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
        session_limits={
            "max_sessions": max_sessions,
            "memory": session_memory,
            "pids_limit": session_pids_limit,
            "idle_timeout": session_idle_timeout,
            "janitor_interval": janitor_interval,
            "resource_watchdog_interval": resource_watchdog_interval,
            "image_prune_until": image_prune_until,
        },
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
