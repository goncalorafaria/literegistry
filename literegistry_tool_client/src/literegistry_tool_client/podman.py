"""Stateful Podman execution client."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
import json
import logging
from typing import Any
from uuid import uuid4

import aiohttp

from ._transport import ToolClient


DEFAULT_PODMAN_GATEWAY_URL = "http://127.0.0.1:1212"
DEFAULT_PODMAN_SESSION_IMAGE = "docker.io/library/ubuntu:24.04"


class PodmanStreamError(RuntimeError):
    """A streaming request failed; output expiry never replaces the session."""

    def __init__(self, status_code: int, response: Any) -> None:
        self.status_code = status_code
        self.response = response
        detail = response.get("detail", response) if isinstance(response, dict) else {}
        self.code = detail.get("code") if isinstance(detail, dict) else None
        super().__init__(f"Podman streaming request failed (HTTP {status_code}): {response}")


class PodmanCommand:
    """One command's output reader. Routing IDs and cursors stay inside it.

    Use one consumer per command. wait() drains unread output and returns final
    status; it does not retain a second copy of output already read or streamed.
    """

    def __init__(self, client: PodmanExecutionClient, affinity_id: str, command_id: str) -> None:
        self._client = client
        self._affinity_id = affinity_id
        self._command_id = command_id
        self._cursor = 0
        self._lock = asyncio.Lock()
        self._final: dict[str, Any] | None = None
        self._acknowledged = False

    async def _poll(self, wait_ms: int) -> dict[str, Any]:
        if self._client.affinity_id != self._affinity_id:
            raise RuntimeError("The command's Podman session has been closed")
        batch = await self._client._stream_request("GET", "output", {
            "affinity_id": self._affinity_id, "command_id": self._command_id,
            "cursor": self._cursor, "wait_ms": wait_ms,
        })
        next_cursor = batch.get("next_cursor")
        if type(next_cursor) is not int or next_cursor < self._cursor:
            raise RuntimeError("Podman output returned an invalid cursor")
        if not isinstance(batch.get("chunks"), list):
            raise RuntimeError("Podman output returned invalid chunks")
        if type(batch.get("done")) is not bool or type(batch.get("has_more")) is not bool:
            raise RuntimeError("Podman output returned invalid completion status")
        for chunk in batch["chunks"]:
            if (not isinstance(chunk, dict) or chunk.get("stream") not in {"stdout", "stderr"}
                    or not isinstance(chunk.get("data"), str)):
                raise RuntimeError("Podman output returned an invalid chunk")
        self._cursor = next_cursor
        if batch["done"] and not batch["has_more"]:
            self._final = {key: batch.get(key) for key in
                           ("exit_code", "success", "timed_out", "error")}
        return batch

    async def read(self, *, wait_ms: int = 1000) -> dict[str, Any]:
        """Read the next batch as stdout/stderr text and command status."""
        async with self._lock:
            batch = await self._poll(wait_ms)
            return {
                "stdout": "".join(c["data"] for c in batch["chunks"] if c["stream"] == "stdout"),
                "stderr": "".join(c["data"] for c in batch["chunks"] if c["stream"] == "stderr"),
                **{key: batch.get(key) for key in
                   ("running", "done", "has_more", "exit_code", "success", "timed_out", "error")},
            }

    async def stream(self, *, wait_ms: int = 1000) -> AsyncIterator[dict[str, str]]:
        """Yield {stream, data} chunks and acknowledge them on the next read."""
        while self._final is None:
            async with self._lock:
                batch = await self._poll(wait_ms)
            for chunk in batch["chunks"]:
                yield {"stream": chunk["stream"], "data": chunk["data"]}
        await self.wait(wait_ms=wait_ms)

    async def wait(self, *, wait_ms: int = 1000) -> dict[str, Any]:
        """Drain unread output and return cached completion status on later calls."""
        async with self._lock:
            while self._final is None:
                await self._poll(wait_ms)
            if not self._acknowledged:
                try:
                    await self._poll(0)
                except PodmanStreamError as exc:
                    # The final status was already read. Expiry also frees the
                    # last batch, so acknowledgement need not mask success.
                    if exc.status_code != 410 or exc.code not in {"command_expired", "output_expired"}:
                        raise
                self._acknowledged = True
            return dict(self._final)


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

    async def _stream_request(
        self, method: str, endpoint: str, payload: dict[str, Any],
    ) -> dict[str, Any]:
        """One attempt; keep the handle's cursor unchanged on HTTP failure."""
        url = f"{self.gateway_url}/affinity/podman/{endpoint}"
        body = self._prepare_payload(payload)
        request_timeout = aiohttp.ClientTimeout(total=self.timeout)

        async def send(session: aiohttp.ClientSession) -> dict[str, Any]:
            kwargs = {"params": body} if method == "GET" else {"json": body}
            async with session.request(method, url, timeout=request_timeout, **kwargs) as response:
                raw = await response.text()
                try:
                    result = json.loads(raw)
                except json.JSONDecodeError:
                    result = raw[:1000]
                if response.status >= 400:
                    raise PodmanStreamError(response.status, result)
                if not isinstance(result, dict):
                    raise RuntimeError("Podman streaming response was not a JSON object")
                return result

        if self._http_session is not None:
            return await send(self._http_session)
        async with aiohttp.ClientSession(trust_env=False) as session:
            return await send(session)

    async def start_command(
        self, *, command: str, stdin: str = "", timeout: float = 10,
        workdir: str | None = None, merge_output: bool = False,
    ) -> PodmanCommand:
        """Start a command and return its output reader; IDs are managed here."""
        if not isinstance(command, str) or not command.strip():
            raise ValueError("command must be non-empty")
        if not isinstance(stdin, str):
            raise TypeError("stdin must be a string")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        async with self._session_lock:
            if self._affinity_id is None:
                raise RuntimeError("Podman session is not started; call start() first")
            affinity_id = self._affinity_id
            result = await self._stream_request("POST", "start", {
                "affinity_id": affinity_id, "request_id": uuid4().hex,
                "command": command, "stdin": stdin, "timeout": timeout,
                "workdir": self.workdir if workdir is None else workdir,
                "merge_output": merge_output,
            })
            command_id = result.get("command_id")
            if not isinstance(command_id, str) or not command_id:
                raise RuntimeError("Podman start returned no command handle")
            return PodmanCommand(self, affinity_id, command_id)

    async def stream(self, *, command: str, **options: Any) -> AsyncIterator[dict[str, str]]:
        """Start a command and yield {stream, data} chunks without protocol IDs."""
        job = await self.start_command(command=command, **options)
        async for chunk in job.stream():
            yield chunk

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
    "PodmanCommand",
    "PodmanStreamError",
]
