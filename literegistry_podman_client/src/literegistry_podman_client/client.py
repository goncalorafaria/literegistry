"""Standalone asynchronous client for LiteRegistry Podman gateways."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
import json
import math
import random
import logging
import time
from collections.abc import AsyncIterator, Mapping
from typing import Any
from urllib.parse import urlsplit

import aiohttp

from .discovery import GatewayDiscovery


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
        self.request_not_sent = False


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
        gateway_url: str | None = None,
        *,
        gateway_discovery: GatewayDiscovery | None = None,
        service: str = "podman",
        workdir: str = "/workspace",
        request_timeout: float = 310.0,
        recovery_timeout: float = 120.0,
        keepalive_timeout: float = 2.0,
        event_loop_stall_timeout: float = 2.0,
        headers: Mapping[str, str] | None = None,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        if gateway_url is None and gateway_discovery is not None:
            gateway_url = ""
        elif not isinstance(gateway_url, str):
            raise TypeError("gateway_url must be a string")
        gateway_url = gateway_url.rstrip("/")
        parsed = urlsplit(gateway_url)
        if gateway_url or gateway_discovery is None:
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                raise ValueError("gateway_url must be an absolute HTTP(S) URL")
        if not isinstance(service, str) or not service.strip():
            raise ValueError("service must be a non-empty string")
        if not isinstance(workdir, str) or not workdir.strip():
            raise ValueError("workdir must be a non-empty string")
        if request_timeout <= 0:
            raise ValueError("request_timeout must be greater than zero")

        for name, value in (
            ("keepalive_timeout", keepalive_timeout),
            ("event_loop_stall_timeout", event_loop_stall_timeout),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and greater than zero")

        if not math.isfinite(recovery_timeout) or recovery_timeout < 0:
            raise ValueError("recovery_timeout must be finite and non-negative")
        self.recovery_timeout = recovery_timeout
        self.recovery_stats = {"retries": 0, "recovered": 0, "exhausted": 0}
        self.keepalive_timeout = keepalive_timeout
        self.event_loop_stall_timeout = event_loop_stall_timeout
        self._loop_tick = time.monotonic()
        self._loop_timer: asyncio.TimerHandle | None = None
        self._pool_stale = False
        self._active_requests: dict[aiohttp.ClientSession, int] = {}
        self._retired_sessions: set[aiohttp.ClientSession] = set()
        self.gateway_url = gateway_url
        self._gateway_discovery = gateway_discovery
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
        if not self.gateway_url:
            raise PodmanGatewayError(
                "open the client before using its discovered mirror URL"
            )
        return self.gateway_url

    def _observe_loop(self) -> None:
        # A response may sit in the socket while the event loop is blocked.
        # aiohttp then timestamps its release as "now", even though the peer
        # has already expired keepalive. Track loop stalls independently.
        now = time.monotonic()
        if now - self._loop_tick > self.event_loop_stall_timeout:
            self._pool_stale = True
        self._loop_tick = now
        self._loop_timer = asyncio.get_running_loop().call_later(
            min(0.5, self.event_loop_stall_timeout / 4), self._observe_loop
        )

    async def open(self) -> "PodmanClient":
        """Reuse healthy connections; replace owned pools after a loop stall."""
        if self._gateway_discovery is not None:
            try:
                self.gateway_url, _ = await self._gateway_discovery.resolve()
            except (OSError, asyncio.TimeoutError, LookupError) as exc:
                error = PodmanGatewayError(
                    f"gateway discovery failed before sending: {exc}"
                )
                error.request_not_sent = True
                raise error from exc
        async with self._open_lock:
            if self._owns_http_session:
                if time.monotonic() - self._loop_tick > self.event_loop_stall_timeout:
                    self._pool_stale = True
                if self._pool_stale and self._http_session is not None:
                    old = self._http_session
                    self._http_session = None
                    if self._active_requests.get(old, 0):
                        self._retired_sessions.add(old)
                    else:
                        await old.close()
                self._pool_stale = False
                # Check for a stall before advancing the observation clock.
                self._loop_tick = time.monotonic()
                if self._loop_timer is None:
                    self._observe_loop()
            if self.is_open:
                return self
            if not self._owns_http_session and self._http_session is not None:
                raise PodmanGatewayError("the supplied aiohttp session is closed")
            self._http_session = aiohttp.ClientSession(
                connector=aiohttp.TCPConnector(
                    keepalive_timeout=self.keepalive_timeout
                ),
                timeout=aiohttp.ClientTimeout(total=self.request_timeout),
                trust_env=False,
            )
            self._owns_http_session = True
        return self

    async def _retire_pool(self, session: aiohttp.ClientSession) -> None:
        async with self._open_lock:
            if not self._owns_http_session:
                return  # A caller-supplied pool remains the caller's responsibility.
            if self._http_session is session:
                self._http_session = None
            # Other requests on this pool may still be executing commands.
            # Drain them instead of closing their sockets underneath them.
            self._retired_sessions.add(session)

    async def aclose(self) -> None:
        """Close owned HTTP pools without deleting active containers."""
        async with self._open_lock:
            if self._loop_timer is not None:
                self._loop_timer.cancel()
                self._loop_timer = None
            sessions = set(self._retired_sessions)
            if self._owns_http_session and self._http_session is not None:
                sessions.add(self._http_session)
            self._http_session = None
            self._retired_sessions.clear()
            for session in sessions:
                await session.close()

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
        """Wait for recovery only when replay cannot duplicate an operation.

        Discovery and connection-establishment failures occur before sending.
        Disconnects/timeouts after sending remain terminal for mutations.
        """
        deadline = None
        attempt = 0
        while True:
            try:
                if deadline is None:
                    return await self._request_once(method, endpoint, payload)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self.recovery_stats["exhausted"] += 1
                    raise last_error
                # A timeout here can have an unknown execution outcome; do not
                # turn it into another retry.
                result = await asyncio.wait_for(
                    self._request_once(method, endpoint, payload), timeout=remaining
                )
                self.recovery_stats["recovered"] += 1
                return result
            except PodmanGatewayError as error:
                safe = getattr(error, "request_not_sent", False)
                readonly = method == "GET" and error.status_code is None
                if not (safe or readonly) or self.recovery_timeout == 0:
                    raise
                last_error = error
                if deadline is None:
                    deadline = time.monotonic() + self.recovery_timeout
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise
                attempt += 1
                self.recovery_stats["retries"] += 1
                delay = min(
                    remaining,
                    random.uniform(0.5, 1.0) * min(5.0, 0.5 * 2 ** min(attempt - 1, 4)),
                )
                logging.getLogger(__name__).warning(
                    "gateway_recovery endpoint=%s attempt=%s delay=%.2f request_not_sent=%s",
                    endpoint,
                    attempt,
                    delay,
                    safe,
                )
                await asyncio.sleep(delay)

    async def _request_once(
        self,
        method: str,
        endpoint: str,
        payload: Any = None,
    ) -> JsonObject:
        await self.open()
        assert self._http_session is not None
        gateway_url = self.gateway_url
        generation = None
        if self._gateway_discovery is not None:
            try:
                gateway_url, generation = await self._gateway_discovery.resolve()
            except (OSError, asyncio.TimeoutError, LookupError) as exc:
                error = PodmanGatewayError(
                    f"gateway discovery failed before sending: {exc}"
                )
                error.request_not_sent = True
                raise error from exc
            self.gateway_url = gateway_url
        url = f"{gateway_url}/{endpoint.lstrip('/')}"
        headers = {"accept": "application/json", **self.headers}
        if payload is not None:
            headers["content-type"] = "application/json"
        # Discovery may have yielded while another caller retired its pool.
        await self.open()
        session = self._http_session
        assert session is not None
        self._active_requests[session] = self._active_requests.get(session, 0) + 1
        try:
            async with session.request(
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
                        and detail.get("code")
                        in ("affinity_owner_lost", "sandbox_lost")
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
            await self._retire_pool(session)
            if self._gateway_discovery is not None and generation is not None:
                self._gateway_discovery.invalidate(gateway_url, generation)
            # The command may already have executed. Never retry it implicitly.
            error = PodmanGatewayError(
                f"could not reach gateway at {gateway_url}: {exc}"
            )
            error.request_not_sent = isinstance(exc, aiohttp.ClientConnectorError)
            raise error from exc
        finally:
            remaining = self._active_requests[session] - 1
            if remaining:
                self._active_requests[session] = remaining
            else:
                self._active_requests.pop(session, None)
                if session in self._retired_sessions:
                    self._retired_sessions.discard(session)
                    await session.close()
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

    async def renew(self, affinity_id: str) -> JsonObject:
        """Renew the existing gateway mapping; never create or reassign a session."""
        if not isinstance(affinity_id, str) or not affinity_id.strip():
            raise ValueError("affinity_id must be a non-empty string")
        return await self._post(
            "affinity/renew",
            {
                "service": self.service,
                "affinity_id": affinity_id.strip(),
            },
        )

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
        self._loss: PodmanContainerLostError | None = None
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
