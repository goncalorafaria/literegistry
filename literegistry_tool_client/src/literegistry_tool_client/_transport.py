"""Shared asynchronous HTTP transport for tool-service clients."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
import json
from typing import Any

import aiohttp


RETRYABLE_HTTP_STATUSES = frozenset({429, 502, 503})
HTTP_RETRY_BASE_DELAY_SECONDS = 0.5
HTTP_RETRY_MAX_DELAY_SECONDS = 8.0


class ToolClient(ABC):
    """Base class for asynchronous JSON-over-HTTP tool-service clients.

    A caller may inject an existing ``aiohttp.ClientSession`` to share
    connection pools across many client objects. Otherwise, a short-lived
    session is created for each request.
    """

    def __init__(
        self,
        server_url: str,
        *,
        timeout: float,
        max_retries: int,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        if not isinstance(server_url, str) or not server_url.strip():
            raise ValueError("server_url must be a non-empty string")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        if max_retries < 1:
            raise ValueError("max_retries must be at least 1")
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self._http_session = http_session

    @property
    @abstractmethod
    def request_name(self) -> str:
        """Human-readable operation name used in transport errors."""

    def _prepare_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Apply tool-specific fields before sending a request."""
        return payload

    @abstractmethod
    async def execute(self, **tool_input: Any) -> dict[str, Any]:
        """Execute tool-specific keyword arguments through the transport."""

    async def _post_with_retries_async(
        self,
        payload: dict[str, Any],
        *,
        server_url: str | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        body = self._prepare_payload(payload)
        target_url = server_url or self.server_url
        delay = HTTP_RETRY_BASE_DELAY_SECONDS
        request_timeout = aiohttp.ClientTimeout(
            total=self.timeout if timeout is None else timeout
        )
        transient_network_errors = (
            aiohttp.ClientConnectionError,
            aiohttp.ServerDisconnectedError,
            asyncio.TimeoutError,
        )

        async def post_once(
            session: aiohttp.ClientSession, *, set_timeout: bool
        ) -> tuple[int, str]:
            kwargs: dict[str, Any] = {"json": body}
            if set_timeout:
                kwargs["timeout"] = request_timeout
            async with session.post(target_url, **kwargs) as response:
                return response.status, await response.text()

        for attempt in range(self.max_retries):
            try:
                if self._http_session is None:
                    async with aiohttp.ClientSession(
                        timeout=request_timeout, trust_env=False
                    ) as session:
                        response_status, response_text = await post_once(
                            session, set_timeout=False
                        )
                else:
                    response_status, response_text = await post_once(
                        self._http_session, set_timeout=True
                    )

                if (
                    response_status in RETRYABLE_HTTP_STATUSES
                    and attempt < self.max_retries - 1
                ):
                    await asyncio.sleep(delay)
                    delay = min(delay * 2.0, HTTP_RETRY_MAX_DELAY_SECONDS)
                    continue

                if response_status >= 400:
                    error = RuntimeError(
                        f"{self.request_name} failed with HTTP "
                        f"{response_status} at {target_url}: "
                        f"{response_text[:1000]}"
                    )
                    if attempt == self.max_retries - 1:
                        raise error
                    await asyncio.sleep(delay)
                    delay = min(delay * 2.0, HTTP_RETRY_MAX_DELAY_SECONDS)
                    continue

                try:
                    result = json.loads(response_text)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(
                        f"{self.request_name} response was not valid JSON"
                    ) from exc
                if not isinstance(result, dict):
                    raise RuntimeError(
                        f"{self.request_name} response was not a JSON object"
                    )
                return result
            except transient_network_errors as exc:
                if attempt == self.max_retries - 1:
                    raise RuntimeError(
                        f"{self.request_name} failed after "
                        f"{self.max_retries} attempts at {target_url}: {exc}"
                    ) from exc
                await asyncio.sleep(delay)
                delay = min(delay * 2.0, HTTP_RETRY_MAX_DELAY_SECONDS)

        raise RuntimeError("unreachable")
