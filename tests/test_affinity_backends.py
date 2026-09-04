"""Shared affinity contract tests for every registry database backend."""

from __future__ import annotations

import asyncio
import time

import pytest

from literegistry import (
    FileSystemKVStore,
    RedisKVStore,
    SQLiteKVStore,
    get_kvstore,
)
from literegistry.affinity import (
    AffinityBindingTypeMismatch,
    SoftAffinityBindingStore,
    StrictAffinityBindingStore,
)


class _MemoryRedis:
    """Small redis.asyncio-compatible backend with millisecond TTLs."""

    def __init__(self) -> None:
        self.values: dict[str, bytes] = {}
        self.deadlines: dict[str, float] = {}
        self.closed = False

    def _expire(self, key: str) -> None:
        deadline = self.deadlines.get(key)
        if deadline is not None and deadline <= time.monotonic():
            self.values.pop(key, None)
            self.deadlines.pop(key, None)

    async def ping(self) -> bool:
        return True

    async def get(self, key: str):
        self._expire(key)
        return self.values.get(key)

    async def set(self, key: str, value: bytes, **options) -> bool:
        self.values[key] = bytes(value)
        if "px" in options:
            self.deadlines[key] = time.monotonic() + options["px"] / 1000
        else:
            self.deadlines.pop(key, None)
        return True

    async def delete(self, key: str) -> int:
        self._expire(key)
        self.deadlines.pop(key, None)
        return int(self.values.pop(key, None) is not None)

    async def exists(self, key: str) -> int:
        self._expire(key)
        return int(key in self.values)

    async def scan_iter(self, match: str = "*"):
        prefix = match[:-1] if match.endswith("*") else match
        for key in list(self.values):
            self._expire(key)
            if key in self.values and key.startswith(prefix):
                yield key.encode("utf-8")

    async def aclose(self) -> None:
        self.closed = True


def _make_store(backend: str, tmp_path):
    if backend == "file":
        store = get_kvstore(f"file://{tmp_path / 'registry'}")
        assert isinstance(store, FileSystemKVStore)
        return store
    if backend == "sqlite":
        store = get_kvstore(f"sqlite://{tmp_path / 'registry.sqlite3'}")
        assert isinstance(store, SQLiteKVStore)
        return store
    if backend == "redis":
        store = get_kvstore("redis://registry.test:6379", raise_on_error=True)
        assert isinstance(store, RedisKVStore)
        store._redis = _MemoryRedis()
        return store
    raise AssertionError(f"unexpected backend: {backend}")


@pytest.mark.parametrize("backend", ["file", "sqlite", "redis"])
def test_affinity_contract_across_registry_backends(backend, tmp_path) -> None:
    async def scenario() -> None:
        kv = _make_store(backend, tmp_path)
        strict = StrictAffinityBindingStore(kv, default_ttl_seconds=30)
        soft = SoftAffinityBindingStore(kv, default_ttl_seconds=30)

        assert await kv.set("ordinary:health", "healthy")
        first = await strict.bind(
            "podman",
            "container-one",
            "podman-a",
            "http://podman-a:8091",
        )
        second = await strict.bind(
            "podman",
            "container-two",
            "podman-b",
            "http://podman-b:8091",
        )
        image = await soft.bind(
            "docker_mirror",
            "docker.io/library/python:3.11",
            "mirror-a",
            "http://mirror-a:5000",
        )

        assert await strict.resolve("podman", "container-one") == first
        assert await soft.resolve(
            "docker_mirror",
            "docker.io/library/python:3.11",
        ) == image
        with pytest.raises(AffinityBindingTypeMismatch):
            await soft.resolve("podman", "container-one")

        refreshed = await strict.refresh_binding(first)
        assert refreshed.created_at == first.created_at
        assert refreshed.last_used_at >= first.last_used_at
        assert refreshed.expires_at >= first.expires_at

        handed_off = await soft.handoff(
            "docker_mirror",
            "docker.io/library/python:3.11",
            "mirror-b",
            "http://mirror-b:5000",
        )
        assert handed_off is not None
        assert handed_off.server_id == "mirror-b"
        assert handed_off.previous_server_id == "mirror-a"
        assert handed_off.handoff_count == 1

        assert {binding.server_id for binding in await strict.list_bindings("podman")} == {
            "podman-a",
            "podman-b",
        }
        assert [
            binding.server_id
            for binding in await soft.list_bindings("docker_mirror")
        ] == ["mirror-b"]

        assert await strict.release_server("podman-a", service="podman") == 1
        assert await strict.resolve("podman", "container-one") is None
        assert await strict.resolve("podman", "container-two") == second
        assert await strict.release("podman", "container-two")
        assert await kv.get("ordinary:health") == b"healthy"

        await strict.bind(
            "podman",
            "short-lived",
            "podman-c",
            "http://podman-c:8091",
            ttl_seconds=0.03,
        )
        await asyncio.sleep(0.08)
        assert await strict.resolve("podman", "short-lived") is None
        assert await strict.list_bindings("podman") == []

        assert await soft.release(
            "docker_mirror",
            "docker.io/library/python:3.11",
        )
        await kv.close()

    asyncio.run(scenario())
