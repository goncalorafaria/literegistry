"""Standalone asynchronous client for LiteRegistry Podman gateways."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
import json
from collections.abc import AsyncIterator, Mapping
from typing import Any
from urllib.parse import urlencode, urlsplit
from uuid import uuid4

import aiohttp


JsonObject = dict[str, Any]


class PodmanGatewayError(RuntimeError):
    """A gateway request failed or returned an invalid response."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        response: Any = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.response = response


def _error_detail(response: Any) -> JsonObject:
    if not isinstance(response, dict):
        return {}
    detail = response.get("detail")
    return detail if isinstance(detail, dict) else response


class PodmanContainerLostError(PodmanGatewayError):
    """A container is permanently unavailable; its episode cannot continue."""

    def __init__(self, *, status_code: int, response: Any) -> None:
        detail = _error_detail(response)
        self.code = detail.get("code", "sandbox_lost")
        self.reason = detail.get("reason") or (
            "affinity_owner_lost" if self.code == "affinity_owner_lost" else "unknown"
        )
        self.container_id = detail.get("container_id")
        self.limit = detail.get("limit")
        self.observed = detail.get("observed")
        self.unit = detail.get("unit")
        self.enforcement = detail.get("enforcement")
        super().__init__(
            f"the Podman container died or was lost ({self.reason}) and cannot be recovered; "
            "end this episode before creating a new session",
            status_code=status_code,
            response=response,
        )
        self.recoverable = False


class PodmanCommandError(PodmanGatewayError):
    """A command completed with a non-zero exit code."""

    def __init__(self, result: "CommandResult") -> None:
        message = f"container command exited with code {result.exit_code}"
        if result.stderr.strip():
            message += f": {result.stderr.strip()}"
        super().__init__(message)
        self.result = result


@dataclass(frozen=True)
class CommandResult:
    """Captured result of one command executed inside a Podman container."""

    container_id: str
    affinity_id: str
    stdout: str
    stderr: str
    success: bool
    exit_code: int
    execution_time: float
    timed_out: bool
    stdout_truncated: bool = False
    stderr_truncated: bool = False

    @classmethod
    def from_payload(cls, payload: JsonObject) -> "CommandResult":
        try:
            container_id = payload["container_id"]
            affinity_id = payload["affinity_id"]
            stdout = payload.get("stdout", "")
            stderr = payload.get("stderr", "")
            success = payload["success"]
            timed_out = payload.get("timed_out", False)
            stdout_truncated = payload.get("stdout_truncated", False)
            stderr_truncated = payload.get("stderr_truncated", False)
            if not isinstance(container_id, str) or not container_id:
                raise TypeError("invalid container_id")
            if not isinstance(affinity_id, str) or not affinity_id:
                raise TypeError("invalid affinity_id")
            if not isinstance(stdout, str) or not isinstance(stderr, str):
                raise TypeError("invalid output")
            if not all(
                isinstance(value, bool)
                for value in (success, timed_out, stdout_truncated, stderr_truncated)
            ):
                raise TypeError("invalid status")
            return cls(
                container_id=container_id,
                affinity_id=affinity_id,
                stdout=stdout,
                stderr=stderr,
                success=success,
                exit_code=int(payload["exit_code"]),
                execution_time=float(payload["execution_time"]),
                timed_out=timed_out,
                stdout_truncated=stdout_truncated,
                stderr_truncated=stderr_truncated,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise PodmanGatewayError(
                "gateway returned an invalid Podman command response",
                response=payload,
            ) from exc

    def check_returncode(self) -> "CommandResult":
        """Return this result, or raise :class:`PodmanCommandError`."""
        if not self.success:
            raise PodmanCommandError(self)
        return self


class PodmanClient:
    """Shareable client for creating independent Podman sessions.

    The client owns only an HTTP connection pool. Container identity lives on
    each :class:`PodmanSession`, so one client can safely serve many concurrent
    trajectories.
    """

    def __init__(
        self,
        gateway_url: str,
        *,
        service: str = "podman",
        workdir: str = "/workspace",
        request_timeout: float = 310.0,
        headers: Mapping[str, str] | None = None,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        if not isinstance(gateway_url, str):
            raise TypeError("gateway_url must be a string")
        gateway_url = gateway_url.rstrip("/")
        parsed = urlsplit(gateway_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("gateway_url must be an absolute HTTP(S) URL")
        if not isinstance(service, str) or not service.strip():
            raise ValueError("service must be a non-empty string")
        if not isinstance(workdir, str) or not workdir.strip():
            raise ValueError("workdir must be a non-empty string")
        if request_timeout <= 0:
            raise ValueError("request_timeout must be greater than zero")

        self.gateway_url = gateway_url
        self.service = service.strip()
        self.workdir = workdir
        self.request_timeout = request_timeout
        self.headers = dict(headers or {})
        self._http_session = http_session
        self._owns_http_session = http_session is None
        self._open_lock = asyncio.Lock()

    @property
    def is_open(self) -> bool:
        """Whether this client currently has a usable HTTP session."""
        return self._http_session is not None and not self._http_session.closed

    @property
    def mirror_url(self) -> str:
        """Docker Registry mirror base URL exposed by the same gateway."""
        return self.gateway_url

    async def open(self) -> "PodmanClient":
        """Open the shared HTTP connection pool and return this client."""
        async with self._open_lock:
            if self.is_open:
                return self
            if not self._owns_http_session and self._http_session is not None:
                raise PodmanGatewayError("the supplied aiohttp session is closed")
            self._http_session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.request_timeout),
                trust_env=False,
            )
            self._owns_http_session = True
        return self

    async def aclose(self) -> None:
        """Close the owned HTTP pool without deleting active containers."""
        async with self._open_lock:
            session = self._http_session
            if session is not None and self._owns_http_session and not session.closed:
                await session.close()
            self._http_session = None

    async def __aenter__(self) -> "PodmanClient":
        return await self.open()

    async def __aexit__(self, exc_type, exc_value, traceback) -> bool:
        await self.aclose()
        return False

    @staticmethod
    def _decode_json(raw: str) -> Any:
        if not raw:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return raw

    async def _request(
        self,
        method: str,
        endpoint: str,
        payload: Any = None,
    ) -> JsonObject:
        await self.open()
        assert self._http_session is not None
        url = f"{self.gateway_url}/{endpoint.lstrip('/')}"
        headers = {"accept": "application/json", **self.headers}
        if payload is not None:
            headers["content-type"] = "application/json"
        try:
            async with self._http_session.request(
                method,
                url,
                headers=headers,
                json=payload,
            ) as response:
                result = self._decode_json(await response.text())
                if response.status >= 400:
                    detail = _error_detail(result)
                    confirmed_loss = (
                        response.status == 410
                        and detail.get("code") in ("affinity_owner_lost", "sandbox_lost")
                        and detail.get("recoverable") is False
                    )
                    legacy_missing = (
                        response.status == 404
                        and detail.get("error") == "container_not_found"
                    )
                    if confirmed_loss or legacy_missing:
                        raise PodmanContainerLostError(
                            status_code=response.status,
                            response=result,
                        )
                    raise PodmanGatewayError(
                        f"gateway returned HTTP {response.status} for "
                        f"{endpoint}: {result!r}",
                        status_code=response.status,
                        response=result,
                    )
        except PodmanGatewayError:
            raise
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            raise PodmanGatewayError(
                f"could not reach gateway at {self.gateway_url}: {exc}"
            ) from exc
        if not isinstance(result, dict):
            raise PodmanGatewayError(
                f"gateway returned a non-object response for {endpoint}",
                response=result,
            )
        return result

    async def _post(self, endpoint: str, payload: JsonObject) -> JsonObject:
        return await self._request("POST", endpoint, payload)

    async def health(self) -> JsonObject:
        """Return the LiteRegistry gateway health response."""
        return await self._request("GET", "health")

    async def mirror_health(self) -> JsonObject:
        """Probe the Docker Registry V2 endpoint exposed by the gateway."""
        return await self._request("GET", "v2/")

    async def handshake(
        self,
        *,
        image: str | None = None,
        client_id: str | None = None,
    ) -> "PodmanSession":
        """Create a container and return its independent affinity session."""
        payload: JsonObject = {"service": self.service}
        if image is not None:
            if not isinstance(image, str) or not image.strip():
                raise ValueError("image must be a non-empty string")
            payload["image"] = image.strip()
        if client_id is not None:
            if not isinstance(client_id, str) or not client_id.strip():
                raise ValueError("client_id must be a non-empty string")
            payload["client_id"] = client_id.strip()

        response = await self._post("affinity/handshake", payload)
        try:
            values = (
                response["affinity_id"],
                response["container_id"],
                response["instance_id"],
                response["image"],
            )
        except KeyError as exc:
            raise PodmanGatewayError(
                "gateway returned an invalid handshake response",
                response=response,
            ) from exc
        if not all(isinstance(value, str) and value for value in values):
            raise PodmanGatewayError(
                "gateway returned invalid handshake identifiers",
                response=response,
            )
        affinity_id, container_id, instance_id, selected_image = values
        return PodmanSession(
            client=self,
            affinity_id=affinity_id,
            container_id=container_id,
            instance_id=instance_id,
            image=selected_image,
        )

    async def execute(
        self,
        affinity_id: str,
        command: str,
        *,
        stdin: str = "",
        timeout: float = 10.0,
        workdir: str | None = None,
    ) -> CommandResult:
        """Execute a command in the container selected by ``affinity_id``."""
        if not isinstance(affinity_id, str) or not affinity_id.strip():
            raise ValueError("affinity_id must be a non-empty string")
        if not isinstance(command, str) or not command.strip():
            raise ValueError("command must be a non-empty string")
        if not isinstance(stdin, str):
            raise TypeError("stdin must be a string")
        if timeout <= 0:
            raise ValueError("timeout must be greater than zero")
        selected_workdir = self.workdir if workdir is None else workdir
        if not isinstance(selected_workdir, str) or not selected_workdir.strip():
            raise ValueError("workdir must be a non-empty string")

        response = await self._post(
            "affinity/podman",
            {
                "service": self.service,
                "affinity_id": affinity_id.strip(),
                "command": command,
                "stdin": stdin,
                "timeout": timeout,
                "workdir": selected_workdir,
            },
        )
        return CommandResult.from_payload(response)

    async def start_command(
        self, affinity_id: str, command: str, *, stdin: str = "",
        timeout: float = 10.0, workdir: str | None = None,
        merge_output: bool = False, request_id: str | None = None,
    ) -> "PodmanCommand":
        """Start once and return a handle for incremental output polling.

        Supply a stable request_id to safely retry an uncertain start within
        the worker's retention window. Commands on a busy session return 409.
        """
        if not isinstance(affinity_id, str) or not affinity_id.strip():
            raise ValueError("affinity_id must be a non-empty string")
        if not isinstance(command, str) or not command.strip():
            raise ValueError("command must be a non-empty string")
        response = await self._post("affinity/podman/start", {
            "service": self.service, "affinity_id": affinity_id,
            "command": command, "stdin": stdin, "timeout": timeout,
            "workdir": self.workdir if workdir is None else workdir,
            "merge_output": merge_output,
            "request_id": uuid4().hex if request_id is None else request_id,
        })
        command_id = response.get("command_id")
        if not isinstance(command_id, str) or not command_id:
            raise PodmanGatewayError("invalid command start response", response=response)
        return PodmanCommand(client=self, affinity_id=affinity_id, command_id=command_id)

    async def command_output(
        self, affinity_id: str, command_id: str, *, cursor: int = 0,
        wait_ms: int = 1000, max_chunks: int = 64,
    ) -> JsonObject:
        """Read output; cursor acknowledges and frees all previous chunks."""
        query = urlencode({"service": self.service, "affinity_id": affinity_id,
                           "command_id": command_id, "cursor": cursor,
                           "wait_ms": wait_ms, "max_chunks": max_chunks})
        return await self._request("GET", f"affinity/podman/output?{query}")

    async def close(self, affinity_id: str) -> JsonObject:
        """Delete one container and its affinity binding."""
        if not isinstance(affinity_id, str) or not affinity_id.strip():
            raise ValueError("affinity_id must be a non-empty string")
        return await self._post(
            "affinity/close",
            {"service": self.service, "affinity_id": affinity_id.strip()},
        )

    @asynccontextmanager
    async def session(
        self,
        *,
        image: str | None = None,
        client_id: str | None = None,
    ) -> AsyncIterator["PodmanSession"]:
        """Handshake and guarantee container close around an async block."""
        podman_session = await self.handshake(image=image, client_id=client_id)
        try:
            yield podman_session
        finally:
            await podman_session.close()


class PodmanCommand:
    """A running command and its single-consumer acknowledgement cursor."""

    def __init__(self, *, client: PodmanClient, affinity_id: str, command_id: str) -> None:
        self.client = client
        self.affinity_id = affinity_id
        self.command_id = command_id
        self.cursor = 0
        self._poll_lock = asyncio.Lock()
        self._final: JsonObject | None = None
        self._acknowledged = False

    async def poll(self, *, wait_ms: int = 1000, max_chunks: int = 64) -> JsonObject:
        """Return the next batch, advancing only after a successful response."""
        async with self._poll_lock:
            response = await self.client.command_output(
                self.affinity_id, self.command_id, cursor=self.cursor,
                wait_ms=wait_ms, max_chunks=max_chunks,
            )
            next_cursor = response.get("next_cursor")
            if type(next_cursor) is not int or next_cursor < self.cursor:
                raise PodmanGatewayError("invalid command output cursor", response=response)
            if (not isinstance(response.get("chunks"), list)
                    or type(response.get("done")) is not bool
                    or type(response.get("has_more")) is not bool):
                raise PodmanGatewayError("invalid command output batch", response=response)
            for chunk in response["chunks"]:
                if (not isinstance(chunk, dict) or chunk.get("stream") not in {"stdout", "stderr"}
                        or not isinstance(chunk.get("data"), str)):
                    raise PodmanGatewayError("invalid command output chunk", response=response)
            self.cursor = next_cursor
            if response["done"] and not response["has_more"]:
                self._final = {key: response.get(key) for key in
                               ("exit_code", "success", "timed_out", "error")}
            return response

    async def read(self, *, wait_ms: int = 1000) -> JsonObject:
        """Read a batch of stdout/stderr and status, without protocol IDs."""
        batch = await self.poll(wait_ms=wait_ms)
        return {
            "stdout": "".join(c["data"] for c in batch["chunks"] if c["stream"] == "stdout"),
            "stderr": "".join(c["data"] for c in batch["chunks"] if c["stream"] == "stderr"),
            **{key: batch.get(key) for key in
               ("running", "done", "has_more", "exit_code", "success", "timed_out", "error")},
        }

    async def stream(self, *, wait_ms: int = 1000) -> AsyncIterator[dict[str, str]]:
        """Yield {stream, data} chunks and manage cursors and acknowledgement."""
        if self._final is None:
            async for batch in self.output(wait_ms=wait_ms):
                for chunk in batch["chunks"]:
                    yield {"stream": chunk["stream"], "data": chunk["data"]}
        else:
            await self.wait(wait_ms=wait_ms)

    async def wait(self, *, wait_ms: int = 1000) -> JsonObject:
        """Drain unread output and return final status, without accumulating logs."""
        while self._final is None:
            await self.poll(wait_ms=wait_ms)
        if not self._acknowledged:
            try:
                await self.poll(wait_ms=0)
            except PodmanGatewayError as exc:
                if exc.status_code != 410 or _error_detail(exc.response).get("code") not in {
                    "command_expired", "output_expired",
                }:
                    raise
            self._acknowledged = True
        return dict(self._final)

    async def output(self, *, wait_ms: int = 1000) -> AsyncIterator[JsonObject]:
        """Yield batches through the final status and acknowledge the last batch."""
        while True:
            batch = await self.poll(wait_ms=wait_ms)
            yield batch
            if batch["done"] and not batch["has_more"]:
                await self.wait(wait_ms=0)
                return


class PodmanSession:
    """One container returned by a gateway handshake."""

    def __init__(
        self,
        *,
        client: PodmanClient,
        affinity_id: str,
        container_id: str,
        instance_id: str,
        image: str,
    ) -> None:
        self.client = client
        self.affinity_id = affinity_id
        self.container_id = container_id
        self.instance_id = instance_id
        self.image = image
        self._closed = False
        self._loss: PodmanContainerLostError | None = None
        self._close_lock = asyncio.Lock()

    @property
    def closed(self) -> bool:
        return self._closed

    async def start_command(
        self, command: str, *, stdin: str = "", timeout: float = 10.0,
        workdir: str | None = None, merge_output: bool = False,
        request_id: str | None = None,
    ) -> PodmanCommand:
        if self._loss is not None:
            raise self._loss
        if self.closed:
            raise PodmanGatewayError("cannot execute on a closed Podman session")
        try:
            return await self.client.start_command(
                self.affinity_id, command, stdin=stdin, timeout=timeout,
                workdir=workdir, merge_output=merge_output, request_id=request_id,
            )
        except PodmanContainerLostError as exc:
            self._loss = exc
            self._closed = True
            raise

    async def stream(self, command: str, **options: Any) -> AsyncIterator[dict[str, str]]:
        """Start a command and yield stdout/stderr chunks with IDs managed here."""
        job = await self.start_command(command, **options)
        async for chunk in job.stream():
            yield chunk

    async def execute(
        self,
        command: str,
        *,
        stdin: str = "",
        timeout: float = 10.0,
        workdir: str | None = None,
        check: bool = False,
    ) -> CommandResult:
        """Execute one command in this session's container."""
        if self._loss is not None:
            raise self._loss
        if self.closed:
            raise PodmanGatewayError("cannot execute on a closed Podman session")
        try:
            result = await self.client.execute(
                self.affinity_id,
                command,
                stdin=stdin,
                timeout=timeout,
                workdir=workdir,
            )
        except PodmanContainerLostError as exc:
            self._loss = exc
            self._closed = True
            raise
        if self._loss is not None:
            raise self._loss
        return result.check_returncode() if check else result

    async def close(self) -> JsonObject | None:
        """Delete this container once; repeated successful closes are no-ops."""
        async with self._close_lock:
            if self._closed:
                return None
            try:
                response = await self.client.close(self.affinity_id)
            except PodmanContainerLostError as exc:
                # Already gone: cleanup must not hide the episode's original error.
                self._loss = exc
                self._closed = True
                return None
            self._closed = True
            return response

    async def __aenter__(self) -> "PodmanSession":
        return self

    async def __aexit__(self, exc_type, exc_value, traceback) -> bool:
        try:
            await self.close()
        except PodmanGatewayError:
            if exc_type is None:
                raise
        return False
