"""Standalone asynchronous client for LiteRegistry Podman gateways."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
import json
from collections.abc import AsyncIterator, Mapping
from typing import Any
from urllib.parse import urlsplit

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


class PodmanContainerLostError(PodmanGatewayError):
    """The affinity owner and its container are permanently unavailable."""

    def __init__(self, *, status_code: int, response: Any) -> None:
        super().__init__(
            "the Podman container died with its affinity server and cannot be "
            "recovered; create a new session and replay any required state",
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
                    if (
                        response.status == 410
                        and isinstance(result, dict)
                        and result.get("code") == "affinity_owner_lost"
                        and result.get("recoverable") is False
                    ):
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
        self._close_lock = asyncio.Lock()

    @property
    def closed(self) -> bool:
        return self._closed

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
        if self.closed:
            raise PodmanGatewayError("cannot execute on a closed Podman session")
        result = await self.client.execute(
            self.affinity_id,
            command,
            stdin=stdin,
            timeout=timeout,
            workdir=workdir,
        )
        return result.check_returncode() if check else result

    async def close(self) -> JsonObject | None:
        """Delete this container once; repeated successful closes are no-ops."""
        async with self._close_lock:
            if self._closed:
                return None
            response = await self.client.close(self.affinity_id)
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
