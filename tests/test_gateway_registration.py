from __future__ import annotations

import asyncio
from fnmatch import fnmatch
import json
from unittest.mock import patch

from literegistry import FileSystemKVStore, RedisKVStore
from literegistry.client import RegistryClient
from literegistry.gateway import GatewayRegistration


class FakeServerRegistry:
    server_id = "gateway-server-id"

    def __init__(self) -> None:
        self.registrations = []
        self.heartbeats = []
        self.deregistrations = 0

    async def register_server(self, url, port, metadata):
        self.registrations.append((url, port, metadata))
        return self.server_id

    async def heartbeat(self, url, port, data=None):
        self.heartbeats.append((url, port, data))

    async def deregister(self):
        self.deregistrations += 1


class FakeRedis:
    def __init__(self) -> None:
        self.values = {}
        self.closed = False

    async def ping(self):
        return True

    async def get(self, key):
        return self.values.get(key)

    async def set(self, key, value, **kwargs):
        self.values[key] = value
        return True

    async def delete(self, key):
        return int(self.values.pop(key, None) is not None)

    async def exists(self, key):
        return int(key in self.values)

    async def scan_iter(self, match="*"):
        for key in list(self.values):
            if fnmatch(key, match):
                yield key.encode()

    async def aclose(self):
        self.closed = True


def _registration(
    *,
    store,
    registry=None,
    heartbeat_interval=0.005,
    leader_lock_path=None,
):
    return GatewayRegistration(
        store=store,
        registry=registry,
        advertise_host="gateway.example",
        advertise_port=8080,
        instance_id="gateway-production",
        heartbeat_interval=heartbeat_interval,
        worker_count=8,
        leader_lock_path=leader_lock_path,
    )


def test_gateway_registration_metadata_and_heartbeat() -> None:
    async def scenario():
        registry = FakeServerRegistry()
        registration = _registration(store=object(), registry=registry)

        await registration.start()
        await asyncio.sleep(0.02)
        await registration.stop()

        return registration, registry

    registration, registry = asyncio.run(scenario())

    assert registry.registrations == [
        (
            "http://gateway.example",
            8080,
            registration.metadata(),
        )
    ]
    assert registration.metadata()["model_path"] == "gateway"
    assert registration.metadata()["instance_id"] == "gateway-production"
    assert registration.metadata()["worker_count"] == 8
    assert registry.heartbeats
    assert registry.heartbeats[-1][2] == registration.heartbeat_data()
    assert registry.deregistrations == 1


def test_gateway_workers_share_one_lifecycle_and_handoff(tmp_path) -> None:
    async def scenario():
        lock_path = tmp_path / "gateway-production.lock"
        first_registry = FakeServerRegistry()
        second_registry = FakeServerRegistry()
        first = _registration(
            store=object(),
            registry=first_registry,
            leader_lock_path=lock_path,
        )
        second = _registration(
            store=object(),
            registry=second_registry,
            leader_lock_path=lock_path,
        )

        await first.start()
        await second.start()
        initial_state = (
            first.is_leader,
            second.is_leader,
            len(first_registry.registrations),
            len(second_registry.registrations),
        )
        await first.stop()
        await asyncio.sleep(0.02)
        handoff_state = (
            second.is_leader,
            len(second_registry.registrations),
        )
        await second.stop()
        return first, second, first_registry, second_registry, initial_state, handoff_state

    first, second, first_registry, second_registry, initial, handoff = asyncio.run(
        scenario()
    )

    assert initial == (True, False, 1, 0)
    assert first.registry.server_id == second.registry.server_id
    assert first.registry.server_id.startswith("gateway-")
    assert handoff == (True, 1)
    assert first_registry.deregistrations == 1
    assert second_registry.deregistrations == 1


def test_gateway_registration_round_trip_with_filesystem_store(tmp_path) -> None:
    async def scenario():
        store = FileSystemKVStore(tmp_path / "registry")
        reader = RegistryClient(store, cache_ttl=1)
        registration = _registration(store=store)

        await registration.start()
        await asyncio.sleep(0.02)
        models_while_running = await reader.models(force=True)
        raw = await store.get(f"server_{registration.registry.server_id}")
        await registration.stop()
        models_after_stop = await reader.models(force=True)
        await store.close()
        return models_while_running, models_after_stop, json.loads(raw)

    running, stopped, record = asyncio.run(scenario())

    assert len(running["gateway"]) == 1
    assert running["gateway"][0]["uri"] == "http://gateway.example:8080"
    assert record["data"]["status"] == "healthy"
    assert "gateway" not in stopped


def test_gateway_registration_round_trip_with_redis_store() -> None:
    async def scenario():
        fake_redis = FakeRedis()
        with patch("literegistry.redis.redis.from_url", return_value=fake_redis):
            store = RedisKVStore("redis://registry.example:6379")
            reader = RegistryClient(store, cache_ttl=1)
            registration = _registration(store=store)

            await registration.start()
            await asyncio.sleep(0.02)
            models_while_running = await reader.models(force=True)
            raw = await store.get(f"server_{registration.registry.server_id}")
            await registration.stop()
            models_after_stop = await reader.models(force=True)
            await store.close()
        return fake_redis, models_while_running, models_after_stop, json.loads(raw)

    fake_redis, running, stopped, record = asyncio.run(scenario())

    assert len(running["gateway"]) == 1
    assert running["gateway"][0]["uri"] == "http://gateway.example:8080"
    assert record["data"]["status"] == "healthy"
    assert "gateway" not in stopped
    assert fake_redis.closed is True
