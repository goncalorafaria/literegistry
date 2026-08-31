#!/usr/bin/env python3
"""Asynchronous client for Podman sessions through LiteRegistry Gateway.

The client is shareable: it has no global "current container". Every handshake
returns an independent PodmanSession with an explicit affinity ID. All network
and lifecycle operations use aiohttp and are awaitable.
"""

from __future__ import annotations

import argparse
import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
import json
import os
from typing import Any, AsyncIterator, Mapping, Optional
from urllib.parse import urlsplit

import aiohttp


JsonObject = dict[str, Any]


class PodmanGatewayError(RuntimeError):
    """The gateway could not complete a request."""

    def __init__(
        self,
        message: str,
        *,
        status_code: Optional[int] = None,
        response: Any = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.response = response


class PodmanCommandError(PodmanGatewayError):
    """A container command returned a non-zero exit code."""

    def __init__(self, result: "CommandResult") -> None:
        message = f"container command exited with code {result.exit_code}"
        if result.stderr.strip():
            message += f": {result.stderr.strip()}"
        super().__init__(message)
        self.result = result


@dataclass(frozen=True)
class CommandResult:
    container_id: str
    affinity_id: str
    stdout: str
    stderr: str
    success: bool
    exit_code: int
    execution_time: float
    timed_out: bool

    @classmethod
    def from_payload(cls, payload: JsonObject) -> "CommandResult":
        try:
            container_id = payload["container_id"]
            affinity_id = payload["affinity_id"]
            stdout = payload.get("stdout", "")
            stderr = payload.get("stderr", "")
            success = payload["success"]
            timed_out = payload.get("timed_out", False)
            if not isinstance(container_id, str) or not container_id:
                raise TypeError("invalid container_id")
            if not isinstance(affinity_id, str) or not affinity_id:
                raise TypeError("invalid affinity_id")
            if not isinstance(stdout, str) or not isinstance(stderr, str):
                raise TypeError("invalid output")
            if not isinstance(success, bool) or not isinstance(timed_out, bool):
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
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise PodmanGatewayError(
                "gateway returned an invalid Podman response",
                response=payload,
            ) from exc

    def check_returncode(self) -> "CommandResult":
        if not self.success:
            raise PodmanCommandError(self)
        return self


class PodmanGatewayClient:
    """Shareable asynchronous client with one pooled aiohttp session."""

    def __init__(
        self,
        gateway_url: str,
        *,
        service: str = "podman",
        request_timeout: float = 310.0,
        headers: Optional[Mapping[str, str]] = None,
        http_session: Optional[aiohttp.ClientSession] = None,
    ) -> None:
        gateway_url = gateway_url.rstrip("/")
        parsed = urlsplit(gateway_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("gateway_url must be an absolute HTTP(S) URL")
        if not service:
            raise ValueError("service must be non-empty")
        if request_timeout <= 0:
            raise ValueError("request_timeout must be greater than zero")
        self.gateway_url = gateway_url
        self.service = service
        self.request_timeout = request_timeout
        self.headers = dict(headers or {})
        self._http_session = http_session
        self._owns_http_session = http_session is None
        self._open_lock = asyncio.Lock()

    @property
    def is_open(self) -> bool:
        return self._http_session is not None and not self._http_session.closed

    async def open(self) -> "PodmanGatewayClient":
        """Open the shared HTTP pool and return this client."""
        async with self._open_lock:
            if self.is_open:
                return self
            if not self._owns_http_session and self._http_session is not None:
                raise PodmanGatewayError("the supplied aiohttp session is closed")
            self._http_session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.request_timeout)
            )
            self._owns_http_session = True
        return self

    async def aclose(self) -> None:
        """Close the owned HTTP pool; active containers are not deleted."""
        async with self._open_lock:
            session = self._http_session
            if session is not None and self._owns_http_session and not session.closed:
                await session.close()
            self._http_session = None

    async def __aenter__(self) -> "PodmanGatewayClient":
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
        return await self._request("GET", "health")

    async def handshake(
        self,
        *,
        image: Optional[str] = None,
        client_id: Optional[str] = None,
    ) -> "PodmanSession":
        """Create a container and return its independent session."""
        payload: JsonObject = {"service": self.service}
        if image is not None:
            payload["image"] = image
        if client_id is not None:
            payload["client_id"] = client_id
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
        workdir: str = "/workspace",
    ) -> CommandResult:
        """Execute one command in the container selected by affinity_id."""
        response = await self._post(
            "affinity/podman",
            {
                "service": self.service,
                "affinity_id": affinity_id,
                "command": command,
                "stdin": stdin,
                "timeout": timeout,
                "workdir": workdir,
            },
        )
        return CommandResult.from_payload(response)

    async def close(self, affinity_id: str) -> JsonObject:
        """Delete one container and release its affinity binding."""
        return await self._post(
            "affinity/close",
            {"service": self.service, "affinity_id": affinity_id},
        )

    @asynccontextmanager
    async def session(
        self,
        *,
        image: Optional[str] = None,
        client_id: Optional[str] = None,
    ) -> AsyncIterator["PodmanSession"]:
        """Handshake and guarantee close around an async context block."""
        podman_session = await self.handshake(image=image, client_id=client_id)
        try:
            yield podman_session
        finally:
            await podman_session.close()


class PodmanSession:
    """One container returned by an asynchronous gateway handshake."""

    def __init__(
        self,
        *,
        client: PodmanGatewayClient,
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
        workdir: str = "/workspace",
        check: bool = False,
    ) -> CommandResult:
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

    async def close(self) -> Optional[JsonObject]:
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


async def async_main(args: argparse.Namespace) -> int:
    commands = args.command or [
        "printf 'ai2 hello\\n' > /workspace/hello.txt",
        "cat /workspace/hello.txt",
    ]
    async with PodmanGatewayClient(args.gateway) as client:
        async with client.session(
            image=args.image,
            client_id=args.client_id,
        ) as session:
            print(f"container_id={session.container_id}")
            print(f"instance_id={session.instance_id}")
            for command in commands:
                result = await session.execute(command, check=True)
                if result.stdout:
                    end = "" if result.stdout.endswith("\n") else "\n"
                    print(result.stdout, end=end)
    print("closed=true")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a Podman session through LiteRegistry Gateway"
    )
    parser.add_argument(
        "--gateway",
        default=os.getenv("PODMAN_GATEWAY_URL"),
        help="Gateway URL, or set PODMAN_GATEWAY_URL",
    )
    parser.add_argument("--image", help="Optional OCI image override")
    parser.add_argument("--client-id", default="podman-client-demo")
    parser.add_argument(
        "--command",
        action="append",
        help="Command to execute; repeat for multiple commands",
    )
    args = parser.parse_args()
    if not args.gateway:
        parser.error("--gateway or PODMAN_GATEWAY_URL is required")
    return asyncio.run(async_main(args))


if __name__ == "__main__":
    raise SystemExit(main())
