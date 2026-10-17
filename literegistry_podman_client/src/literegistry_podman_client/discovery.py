"""Shared, failure-driven gateway discovery without request retries."""
from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from urllib.parse import urlsplit


class GatewayDiscovery:
    """Share one instance among clients on the same event loop.

    Successful discovery is cached until a transport failure invalidates its
    generation. There is no TTL or background refresh. Canceling one waiter
    does not cancel discovery for the other clients.
    """

    def __init__(self, discover: Callable[[], Awaitable[str]]) -> None:
        self._discover = discover
        self._url: str | None = None
        self._generation = 0
        self._pending: asyncio.Task[tuple[str, int]] | None = None

    @classmethod
    def from_registry(
        cls, location: str, *, endpoint_name: str = "gateway", timeout: float = 10.0,
    ) -> "GatewayDiscovery":
        """Resolve a named endpoint; requires the optional ``literegistry`` package."""
        if timeout <= 0:
            raise ValueError("timeout must be positive")

        async def discover() -> str:
            from literegistry.coop.endpoints import get_endpoint_registry

            async def lookup() -> str:
                registry = get_endpoint_registry(location)
                try:
                    record = await registry.get(endpoint_name)
                    if record is None:
                        raise LookupError(f"no live gateway endpoint {endpoint_name!r}")
                    return record.uri
                finally:
                    await registry.close()

            return await asyncio.wait_for(lookup(), timeout=timeout)

        return cls(discover)

    async def _lookup(self) -> tuple[str, int]:
        try:
            url = (await self._discover()).rstrip("/")
            parsed = urlsplit(url)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                raise ValueError("discovered gateway must be an absolute HTTP(S) URL")
            self._generation += 1
            self._url = url
            return url, self._generation
        finally:
            self._pending = None

    async def resolve(self) -> tuple[str, int]:
        if self._url is not None:
            return self._url, self._generation
        if self._pending is None:
            self._pending = asyncio.create_task(self._lookup())
            # Retrieve exceptions even if every waiter was canceled.
            self._pending.add_done_callback(
                lambda task: None if task.cancelled() else task.exception()
            )
        return await asyncio.shield(self._pending)

    def invalidate(self, url: str, generation: int) -> None:
        """Ignore late failures from requests against an older selection."""
        if self._url == url and self._generation == generation:
            self._url = None
