"""Stateful Podman execution client."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import aiohttp

from ._transport import ToolClient


DEFAULT_PODMAN_GATEWAY_URL = "http://127.0.0.1:1212"
DEFAULT_PODMAN_SESSION_IMAGE = "docker.io/library/ubuntu:24.04"


class PodmanExecutionClient(ToolClient):
    """Own one gateway-affined Podman container for a rollout."""

    def __init__(
        self,
        gateway_url: str = DEFAULT_PODMAN_GATEWAY_URL,
        *,
        image: str = DEFAULT_PODMAN_SESSION_IMAGE,
        client_id: str | None = None,
        service: str = "podman",
        timeout: float = 70,
        handshake_timeout: float = 300,
        max_retries: int = 3,
        workdir: str = "/workspace",
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        gateway_url = gateway_url.rstrip("/")
        if not gateway_url:
            raise ValueError("gateway_url must be non-empty")
        if not image:
            raise ValueError("image must be non-empty")
        if not service:
            raise ValueError("service must be non-empty")
        if timeout <= 0 or handshake_timeout <= 0:
            raise ValueError("timeouts must be positive")
        if not workdir:
            raise ValueError("workdir must be non-empty")
        super().__init__(
            f"{gateway_url}/affinity/podman",
            timeout=timeout,
            max_retries=max_retries,
            http_session=http_session,
        )
        self.gateway_url = gateway_url
        self.image = image
        self.client_id = client_id
        self.service = service
        self.handshake_timeout = handshake_timeout
        self.workdir = workdir
        self._affinity_id: str | None = None
        self._container_id: str | None = None
        self._session_lock = asyncio.Lock()

    @property
    def request_name(self) -> str:
        return "Podman execution"

    @property
    def affinity_id(self) -> str | None:
        return self._affinity_id

    @property
    def container_id(self) -> str | None:
        return self._container_id

    @property
    def started(self) -> bool:
        return self._affinity_id is not None

    def _prepare_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {**payload, "service": self.service}

    async def start(
        self,
        *,
        image: str | None = None,
        client_id: str | None = None,
    ) -> dict[str, Any]:
        async with self._session_lock:
            if self._affinity_id is not None:
                raise RuntimeError("Podman session is already started")
            payload: dict[str, Any] = {
                "image": self.image if image is None else image,
            }
            selected_client_id = self.client_id if client_id is None else client_id
            if selected_client_id is not None:
                payload["client_id"] = selected_client_id
            result = await self._post_with_retries_async(
                payload,
                server_url=f"{self.gateway_url}/affinity/handshake",
                timeout=self.handshake_timeout,
            )
            affinity_id = result.get("affinity_id")
            container_id = result.get("container_id")
            if not isinstance(affinity_id, str) or not affinity_id:
                raise RuntimeError("Podman handshake returned no affinity_id")
            if not isinstance(container_id, str) or not container_id:
                raise RuntimeError("Podman handshake returned no container_id")
            if affinity_id != container_id:
                raise RuntimeError("Podman handshake returned conflicting session IDs")
            self._affinity_id = affinity_id
            self._container_id = container_id
            return result

    async def execute(
        self,
        *,
        command: str,
        stdin: str = "",
        timeout: float = 10,
        workdir: str | None = None,
    ) -> dict[str, Any]:
        if not command:
            raise ValueError("command must be non-empty")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        async with self._session_lock:
            if self._affinity_id is None:
                raise RuntimeError("Podman session is not started; call start() first")
            return await self._post_with_retries_async(
                {
                    "affinity_id": self._affinity_id,
                    "command": command,
                    "stdin": stdin,
                    "timeout": timeout,
                    "workdir": self.workdir if workdir is None else workdir,
                }
            )

    async def close(self) -> dict[str, Any] | None:
        async with self._session_lock:
            if self._affinity_id is None:
                return None
            result = await self._post_with_retries_async(
                {"affinity_id": self._affinity_id},
                server_url=f"{self.gateway_url}/affinity/close",
            )
            if result.get("removed") is not True:
                raise RuntimeError("Podman close did not confirm container removal")
            self._affinity_id = None
            self._container_id = None
            return result

    async def __aenter__(self) -> "PodmanExecutionClient":
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> bool:
        try:
            await self.close()
        except Exception:
            if exc is None:
                raise
            logging.exception("Failed to close Podman session after an exception")
        return False

    def __str__(self) -> str:
        return f"PodmanExecutionClient(gateway_url={self.gateway_url})"


__all__ = [
    "DEFAULT_PODMAN_GATEWAY_URL",
    "DEFAULT_PODMAN_SESSION_IMAGE",
    "PodmanExecutionClient",
]
