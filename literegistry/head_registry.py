"""Redis failover through a stable backend-neutral endpoint registry.

The head registry can be a filesystem directory, SQLite database, or a separate
Redis instance. The data-plane Redis publishes its current URL there under the
``redis`` endpoint name.
"""

from __future__ import annotations

import asyncio
from contextlib import suppress
import logging
import os
from pathlib import Path
import time
from typing import Awaitable, Callable, TypeVar
from urllib.parse import urlsplit, urlunsplit

from literegistry.coop.endpoints import (
    EndpointRecord,
    get_endpoint_registry,
    normalize_endpoint_registry,
)
from literegistry.kvstore import filesystem_registry_path
from literegistry.kvstore import KeyValueStore
from literegistry.redis import RedisKVStore, redact_redis_url


logger = logging.getLogger(__name__)
_T = TypeVar("_T")
HEAD_REGISTRY_SCHEME = "head://"
HEAD_REGISTRY_PREFIX = "head+"


class HeadRegistryClosedError(RuntimeError):
    """Raised when an operation is attempted after the store is closed."""


def is_head_registry_uri(value: object) -> bool:
    """Return whether a value selects Redis through a head registry."""

    return isinstance(value, str) and (
        value.startswith(HEAD_REGISTRY_PREFIX)
        or value.startswith(HEAD_REGISTRY_SCHEME)
    )


def head_registry_backend(value: str | Path) -> str:
    """Return the file, SQLite, or Redis backend used for endpoint discovery."""

    text = str(value).strip()
    if not text:
        raise ValueError("head_registry must be non-empty")
    if text.startswith(HEAD_REGISTRY_PREFIX):
        text = text[len(HEAD_REGISTRY_PREFIX) :]
    elif text.startswith(HEAD_REGISTRY_SCHEME):
        # Backward compatibility for the original filesystem-only head URI.
        text = "file://" + text[len(HEAD_REGISTRY_SCHEME) :]
    return normalize_endpoint_registry(text)


def head_registry_uri(root: str | Path) -> str:
    """Return a canonical ``head+<backend>://...`` failover URI."""

    return HEAD_REGISTRY_PREFIX + head_registry_backend(root)


def head_registry_path(uri: str) -> Path:
    """Return the path for a file-backed head registry."""

    backend = head_registry_backend(uri)
    if not backend.startswith("file:"):
        raise ValueError("head registry is not file-backed")
    return filesystem_registry_path(backend).absolute()


def _beaker_reachable_url(url: str) -> str:
    """Use Beaker's host gateway for an endpoint on this task's own node."""

    node = os.getenv("BEAKER_NODE_HOSTNAME", "").strip()
    gateway = os.getenv("BEAKER_HOST_GATEWAY", "").strip()
    parsed = urlsplit(url)
    if not node or not gateway or parsed.hostname != node:
        return url

    credentials = ""
    if parsed.username is not None:
        credentials = parsed.username
        if parsed.password is not None:
            credentials += f":{parsed.password}"
        credentials += "@"
    host = f"[{gateway}]" if ":" in gateway and not gateway.startswith("[") else gateway
    port = f":{parsed.port}" if parsed.port is not None else ""
    return urlunsplit(
        (parsed.scheme, f"{credentials}{host}{port}", parsed.path, parsed.query, parsed.fragment)
    )


class HeadRegistryKVStore(KeyValueStore):
    """A Redis KV store whose current URL is discovered from shared storage.

    Operations wait indefinitely by default while Redis is unavailable. They
    remain cancellable, and :meth:`close` wakes any callers waiting for a
    replacement endpoint.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        endpoint_name: str = "redis",
        poll_interval: float = 1.0,
        refresh_interval: float = 5.0,
        db: int = 0,
    ) -> None:
        if poll_interval <= 0 or refresh_interval <= 0:
            raise ValueError("poll_interval and refresh_interval must be positive")
        self.root = head_registry_backend(root)
        self._display_root = (
            redact_redis_url(self.root)
            if self.root.startswith(("redis://", "rediss://"))
            else self.root
        )
        self.endpoint_name = endpoint_name
        self.poll_interval = float(poll_interval)
        self.refresh_interval = float(refresh_interval)
        self.db = db
        self.endpoint_registry = get_endpoint_registry(self.root)
        self._client: RedisKVStore | None = None
        self._endpoint_identity: tuple[str, str] | None = None
        self._last_head_check = 0.0
        self._last_unavailable_log = 0.0
        self._last_unavailable_url: str | None = None
        self._connection_lock = asyncio.Lock()
        self._closed = False
        self._closed_event = asyncio.Event()

    @property
    def current_url(self) -> str | None:
        return self._client.url if self._client is not None else None

    async def _sleep(self) -> None:
        if self._closed:
            raise HeadRegistryClosedError("head registry store is closed")
        try:
            await asyncio.wait_for(self._closed_event.wait(), timeout=self.poll_interval)
        except TimeoutError:
            return
        raise HeadRegistryClosedError("head registry store is closed")

    async def _read_endpoint(self) -> EndpointRecord | None:
        try:
            return await self.endpoint_registry.get(self.endpoint_name)
        except Exception as exc:
            logger.warning(
                "Cannot read head registry %s while discovering %s: %s",
                self._display_root,
                self.endpoint_name,
                exc,
            )
            return None

    async def _disconnect_locked(self) -> None:
        client, self._client = self._client, None
        self._endpoint_identity = None
        if client is not None:
            with suppress(Exception):
                await client.close()

    async def _connect(self, *, force_check: bool = False) -> RedisKVStore:
        while True:
            if self._closed:
                raise HeadRegistryClosedError("head registry store is closed")
            async with self._connection_lock:
                now = time.monotonic()
                if (
                    self._client is not None
                    and not force_check
                    and now - self._last_head_check < self.refresh_interval
                ):
                    return self._client

                record = await self._read_endpoint()
                self._last_head_check = now
                if record is not None:
                    endpoint_url = _beaker_reachable_url(record.uri)
                    identity = (endpoint_url, record.publisher_id)
                    if self._client is not None and identity == self._endpoint_identity:
                        return self._client

                    candidate = RedisKVStore(
                        endpoint_url,
                        db=self.db,
                        raise_on_error=True,
                        log_connections=False,
                    )
                    try:
                        await candidate.ping()
                    except Exception as exc:
                        with suppress(Exception):
                            await candidate.close()
                        log_now = time.monotonic()
                        if (
                            endpoint_url != self._last_unavailable_url
                            or log_now - self._last_unavailable_log >= 30.0
                        ):
                            logger.warning(
                                "Redis endpoint %s from head registry %s is unavailable: %s",
                                redact_redis_url(endpoint_url),
                                self._display_root,
                                exc,
                            )
                            self._last_unavailable_log = log_now
                            self._last_unavailable_url = endpoint_url
                    else:
                        previous = self._client
                        old_url = previous.url if previous is not None else None
                        self._client = candidate
                        self._endpoint_identity = identity
                        self._last_unavailable_url = None
                        if previous is not None:
                            with suppress(Exception):
                                await previous.close()
                        logger.info(
                            "Connected head registry %s to Redis %s%s",
                            self._display_root,
                            redact_redis_url(endpoint_url),
                            (
                                f" (replaced {redact_redis_url(old_url)})"
                                if old_url
                                else ""
                            ),
                        )
                        return candidate

                # Keep a working old client across a short publication gap.
                if self._client is not None and not force_check:
                    return self._client
            force_check = True
            await self._sleep()

    async def _invalidate(self, client: RedisKVStore) -> None:
        async with self._connection_lock:
            if self._client is client:
                failed_url = client.url
                await self._disconnect_locked()
                self._last_head_check = 0.0
                logger.warning(
                    "Lost Redis %s; waiting for a healthy endpoint in head registry %s",
                    redact_redis_url(failed_url),
                    self._display_root,
                )

    async def _execute(
        self,
        operation: Callable[[RedisKVStore], Awaitable[_T]],
    ) -> _T:
        force_check = False
        while True:
            client = await self._connect(force_check=force_check)
            try:
                return await operation(client)
            except HeadRegistryClosedError:
                raise
            except Exception:
                await self._invalidate(client)
                force_check = True
                await self._sleep()

    async def ping(self) -> bool:
        return await self._execute(lambda client: client.ping())

    async def get(self, key: str):
        return await self._execute(lambda client: client.get(key))

    async def set(self, key: str, value, ttl_seconds=None) -> bool:
        return await self._execute(
            lambda client: client.set(key, value, ttl_seconds=ttl_seconds)
        )

    async def delete(self, key: str) -> bool:
        return await self._execute(lambda client: client.delete(key))

    async def exists(self, key: str) -> bool:
        return await self._execute(lambda client: client.exists(key))

    async def keys(self, prefix: str | None = None) -> list[str]:
        return await self._execute(lambda client: client.keys(prefix=prefix))

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._closed_event.set()
        async with self._connection_lock:
            await self._disconnect_locked()
        await self.endpoint_registry.close()
