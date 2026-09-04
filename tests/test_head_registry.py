from __future__ import annotations

import asyncio

import pytest

from literegistry import (
    FileSystemKVStore,
    HeadRegistryClosedError,
    HeadRegistryKVStore,
    SQLiteKVStore,
    get_kvstore,
    head_registry_backend,
    head_registry_path,
    head_registry_uri,
)
from literegistry.coop.endpoints import (
    FilesystemEndpointRegistry,
    get_endpoint_registry,
)
from literegistry.affinity import StrictAffinityBindingStore
from literegistry.registry import ServerRegistry


class _FakeRedis:
    def __init__(self, state: dict) -> None:
        self.state = state
        self.closed = False

    def _check(self) -> None:
        if self.closed or not self.state["online"]:
            raise ConnectionError("redis unavailable")

    async def ping(self):
        self._check()
        return True

    async def get(self, key):
        self._check()
        return self.state["values"].get(key)

    async def set(self, key, value, **_options):
        self._check()
        self.state["values"][key] = value
        return True

    async def delete(self, key):
        self._check()
        return int(self.state["values"].pop(key, None) is not None)

    async def exists(self, key):
        self._check()
        return int(key in self.state["values"])

    async def scan_iter(self, match="*"):
        self._check()
        prefix = match[:-1] if match.endswith("*") else match
        for key in list(self.state["values"]):
            if key.startswith(prefix):
                yield key.encode()

    async def aclose(self):
        self.closed = True


def _install_fake_redis(monkeypatch, states):
    def from_url(url, **_kwargs):
        return _FakeRedis(states[url])

    monkeypatch.setattr("literegistry.redis.redis.from_url", from_url)


def test_head_registry_uri_selects_dynamic_store(tmp_path) -> None:
    uri = head_registry_uri(tmp_path)
    assert uri == f"head+file://{tmp_path}"
    assert head_registry_path(uri) == tmp_path
    assert isinstance(get_kvstore(uri), HeadRegistryKVStore)
    assert isinstance(get_kvstore(head_registry=tmp_path), HeadRegistryKVStore)
    assert isinstance(get_kvstore(f"file://{tmp_path}"), FileSystemKVStore)
    assert head_registry_backend(f"head://{tmp_path}") == f"file://{tmp_path}"
    with pytest.raises(ValueError, match="only one"):
        get_kvstore("redis://old:6379", head_registry=tmp_path)


def test_sqlite_head_registry_discovers_data_plane_redis(
    monkeypatch, tmp_path
) -> None:
    async def check() -> None:
        data_url = "redis://data-plane:6379"
        states = {data_url: {"online": True, "values": {}}}
        _install_fake_redis(monkeypatch, states)
        sqlite_uri = f"sqlite://{tmp_path / 'head.sqlite3'}"
        endpoints = get_endpoint_registry(sqlite_uri)
        await endpoints.publish(
            "redis", data_url, publisher_id="data-plane-1", ttl_seconds=60
        )
        store = get_kvstore("head+" + sqlite_uri)
        assert isinstance(store, HeadRegistryKVStore)
        assert isinstance(store.endpoint_registry.store, SQLiteKVStore)
        try:
            assert await store.set("through-sqlite-head", "yes")
            assert states[data_url]["values"]["through-sqlite-head"] == b"yes"
        finally:
            await store.close()
            await endpoints.close()

    asyncio.run(check())


def test_redis_head_registry_discovers_separate_data_plane_redis(
    monkeypatch,
) -> None:
    async def check() -> None:
        head_url = "redis://head-control:6379"
        data_url = "redis://data-plane:6380"
        states = {
            head_url: {"online": True, "values": {}},
            data_url: {"online": True, "values": {}},
        }
        _install_fake_redis(monkeypatch, states)
        endpoints = get_endpoint_registry(head_url)
        await endpoints.publish(
            "redis", data_url, publisher_id="data-plane-1", ttl_seconds=60
        )
        store = get_kvstore("head+" + head_url)
        try:
            assert await store.set("through-redis-head", "yes")
            assert states[data_url]["values"]["through-redis-head"] == b"yes"
        finally:
            await store.close()
            await endpoints.close()

    asyncio.run(check())


@pytest.mark.parametrize("head_backend", ["file", "sqlite", "redis"])
def test_affinity_through_every_head_registry_backend(
    monkeypatch,
    tmp_path,
    head_backend,
) -> None:
    async def check() -> None:
        data_url = "redis://data-plane:6380"
        states = {data_url: {"online": True, "values": {}}}
        if head_backend == "file":
            head_url = f"file://{tmp_path / 'head'}"
        elif head_backend == "sqlite":
            head_url = f"sqlite://{tmp_path / 'head.sqlite3'}"
        else:
            head_url = "redis://head-control:6379"
            states[head_url] = {"online": True, "values": {}}

        _install_fake_redis(monkeypatch, states)
        endpoints = get_endpoint_registry(head_url)
        await endpoints.publish(
            "redis",
            data_url,
            publisher_id="data-plane-1",
            ttl_seconds=60,
        )
        store = get_kvstore("head+" + head_url)
        affinity = StrictAffinityBindingStore(store)
        try:
            created = await affinity.bind(
                "podman",
                "container-one",
                "podman-a",
                "http://podman-a:8091",
            )
            assert await affinity.resolve("podman", "container-one") == created
            assert await affinity.list_bindings("podman") == [created]
            assert await affinity.release_server("podman-a", "podman") == 1
            assert await affinity.resolve("podman", "container-one") is None
        finally:
            await store.close()
            await endpoints.close()

    asyncio.run(check())


def test_operation_waits_for_replacement_redis(monkeypatch, tmp_path) -> None:
    async def check() -> None:
        old_url = "redis://old:6379"
        new_url = "redis://new:6380"
        states = {
            old_url: {"online": True, "values": {}},
            new_url: {"online": True, "values": {}},
        }
        _install_fake_redis(monkeypatch, states)
        endpoints = FilesystemEndpointRegistry(tmp_path)
        await endpoints.publish(
            "redis", old_url, publisher_id="old", ttl_seconds=60
        )
        store = HeadRegistryKVStore(
            tmp_path,
            poll_interval=0.01,
            refresh_interval=0.01,
        )
        try:
            assert await store.set("before", "old") is True
            assert store.current_url == old_url

            states[old_url]["online"] = False
            pending = asyncio.create_task(store.set("after", "new"))
            await asyncio.sleep(0.04)
            assert not pending.done()

            await endpoints.publish(
                "redis", new_url, publisher_id="new", ttl_seconds=60
            )
            assert await asyncio.wait_for(pending, timeout=1) is True
            assert store.current_url == new_url
            assert states[new_url]["values"]["after"] == b"new"
        finally:
            await store.close()
            await endpoints.close()

    asyncio.run(check())


def test_server_heartbeat_reregisters_after_failover(monkeypatch, tmp_path) -> None:
    async def check() -> None:
        old_url = "redis://old:6379"
        new_url = "redis://new:6380"
        states = {
            old_url: {"online": True, "values": {}},
            new_url: {"online": True, "values": {}},
        }
        _install_fake_redis(monkeypatch, states)
        endpoints = FilesystemEndpointRegistry(tmp_path)
        await endpoints.publish(
            "redis", old_url, publisher_id="old", ttl_seconds=60
        )
        store = HeadRegistryKVStore(tmp_path, poll_interval=0.01)
        registry = ServerRegistry(store)
        try:
            await registry.register_server("http://worker", 8000, {"model_path": "x"})
            key = f"server_{registry.server_id}"
            assert key in states[old_url]["values"]

            states[old_url]["online"] = False
            heartbeat = asyncio.create_task(registry.heartbeat("http://worker", 8000))
            await asyncio.sleep(0.04)
            assert not heartbeat.done()
            await endpoints.publish(
                "redis", new_url, publisher_id="new", ttl_seconds=60
            )
            await asyncio.wait_for(heartbeat, timeout=1)
            assert key in states[new_url]["values"]
        finally:
            await store.close()
            await endpoints.close()

    asyncio.run(check())


def test_close_wakes_operations_waiting_for_initial_redis(tmp_path) -> None:
    async def check() -> None:
        store = HeadRegistryKVStore(tmp_path, poll_interval=10)
        pending = asyncio.create_task(store.get("missing"))
        await asyncio.sleep(0.02)
        await store.close()
        with pytest.raises(HeadRegistryClosedError):
            await asyncio.wait_for(pending, timeout=1)

    asyncio.run(check())
