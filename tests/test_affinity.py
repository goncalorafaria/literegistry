"""Tests for strict/soft affinity persistence and KV capabilities."""

import asyncio
import json

import pytest

from literegistry.affinity import (
    AffinityBindingConflict,
    AffinityBindingStore,
    AffinityBindingTypeMismatch,
    InvalidAffinityBinding,
    SoftAffinityBinding,
    SoftAffinityBindingStore,
    StrictAffinityBinding,
    StrictAffinityBindingStore,
)
from literegistry.kvstore import FileSystemKVStore
from literegistry.redis import RedisKVStore


class LegacyMemoryStore:
    """A pre-TTL/prefix store used to verify compatibility fallbacks."""

    def __init__(self):
        self.values = {}

    async def get(self, key):
        return self.values.get(key)

    async def set(self, key, value):
        self.values[key] = (
            value.encode("utf-8")
            if isinstance(value, str)
            else value
        )
        return True

    async def delete(self, key):
        return self.values.pop(key, None) is not None

    async def exists(self, key):
        return key in self.values

    async def keys(self):
        return list(self.values)


class FakeRedis:
    def __init__(self):
        self.values = {
            b"affinity:tool:one": b"1",
            b"server_one": b"2",
        }
        self.set_calls = []
        self.scan_matches = []

    async def set(self, key, value, **kwargs):
        self.set_calls.append((key, value, kwargs))
        self.values[key.encode()] = value
        return True

    async def scan_iter(self, match):
        self.scan_matches.append(match)
        prefix = match.removesuffix("*").encode()
        for key in self.values:
            if key.startswith(prefix):
                yield key


def test_shared_store_api_is_abstract():
    with pytest.raises(TypeError):
        AffinityBindingStore(LegacyMemoryStore())


def test_strict_bind_resolve_hashes_id_and_persists_type(tmp_path):
    async def check():
        now = [100.0]
        kv = FileSystemKVStore(tmp_path / "registry")
        store = StrictAffinityBindingStore(
            kv,
            default_ttl_seconds=30,
            clock=lambda: now[0],
        )

        binding = await store.bind(
            service="tools/code",
            affinity_id="raw-secret-session-id",
            server_id="worker-a-uuid",
            server_uri="http://worker-a:9000",
        )

        assert isinstance(binding, StrictAffinityBinding)
        assert not hasattr(store, "handoff")
        keys = await kv.keys(prefix="affinity:")
        assert keys == [
            store.binding_key(
                "tools/code",
                "raw-secret-session-id",
            )
        ]
        assert "tools%2Fcode" in keys[0]
        assert "raw-secret-session-id" not in keys[0]
        payload = await kv.get(keys[0])
        assert b"raw-secret-session-id" not in payload
        assert json.loads(payload)["affinity_type"] == "strict"

        resolved = await store.resolve(
            "tools/code",
            "raw-secret-session-id",
        )
        assert resolved == binding
        assert resolved.server_id == "worker-a-uuid"
        assert resolved.created_at == 100.0
        assert resolved.expires_at == 130.0

    asyncio.run(check())


def test_touch_refreshes_sliding_expiration_with_legacy_store():
    async def check():
        now = [10.0]
        kv = LegacyMemoryStore()
        store = StrictAffinityBindingStore(
            kv,
            default_ttl_seconds=5,
            clock=lambda: now[0],
        )
        original = await store.bind(
            "stateful-tool",
            "session-1",
            "worker-a",
            "http://worker-a:8000",
        )

        now[0] = 13.0
        touched = await store.touch("stateful-tool", "session-1")

        assert isinstance(touched, StrictAffinityBinding)
        assert touched.created_at == original.created_at
        assert touched.last_used_at == 13.0
        assert touched.expires_at == 18.0

        now[0] = 18.0
        assert await store.resolve("stateful-tool", "session-1") is None
        assert kv.values == {}

    asyncio.run(check())


def test_strict_binding_rejects_owner_change():
    async def check():
        store = StrictAffinityBindingStore(
            LegacyMemoryStore(),
            clock=lambda: 100.0,
        )
        await store.bind(
            "tool",
            "session",
            "worker-a",
            "http://worker-a:8000",
        )

        with pytest.raises(AffinityBindingConflict):
            await store.bind(
                "tool",
                "session",
                "worker-b",
                "http://worker-b:8000",
            )

        binding = await store.resolve("tool", "session")
        assert binding.server_id == "worker-a"

    asyncio.run(check())


def test_soft_handoff_changes_owner_and_records_transition():
    async def check():
        now = [100.0]
        store = SoftAffinityBindingStore(
            LegacyMemoryStore(),
            default_ttl_seconds=20,
            clock=lambda: now[0],
        )
        original = await store.bind(
            "tool",
            "session",
            "worker-a",
            "http://worker-a:8000",
        )

        assert isinstance(original, SoftAffinityBinding)
        assert original.handoff_count == 0
        now[0] = 108.0
        handed_off = await store.handoff(
            "tool",
            "session",
            "worker-b",
            "http://worker-b:8000",
        )

        assert isinstance(handed_off, SoftAffinityBinding)
        assert handed_off.created_at == original.created_at
        assert handed_off.server_id == "worker-b"
        assert handed_off.previous_server_id == "worker-a"
        assert handed_off.handoff_count == 1
        assert handed_off.last_handoff_at == 108.0
        assert handed_off.last_used_at == 108.0
        assert handed_off.expires_at == 128.0
        assert await store.resolve("tool", "session") == handed_off

    asyncio.run(check())


def test_store_types_cannot_resolve_each_others_bindings():
    async def check():
        kv = LegacyMemoryStore()
        strict = StrictAffinityBindingStore(kv, clock=lambda: 100.0)
        soft = SoftAffinityBindingStore(kv, clock=lambda: 100.0)

        await strict.bind(
            "tool",
            "strict-session",
            "worker-a",
            "http://worker-a:8000",
        )
        await soft.bind(
            "tool",
            "soft-session",
            "worker-b",
            "http://worker-b:8000",
        )

        with pytest.raises(AffinityBindingTypeMismatch):
            await soft.resolve("tool", "strict-session")
        with pytest.raises(AffinityBindingTypeMismatch):
            await strict.resolve("tool", "soft-session")

        assert len(await strict.list_bindings("tool")) == 1
        assert len(await soft.list_bindings("tool")) == 1

    asyncio.run(check())


def test_legacy_records_without_type_are_strict():
    async def check():
        now = [100.0]
        kv = LegacyMemoryStore()
        strict = StrictAffinityBindingStore(kv, clock=lambda: now[0])
        key = strict.binding_key("tool", "legacy-session")
        kv.values[key] = json.dumps(
            {
                "service": "tool",
                "affinity_id_hash": strict.hash_affinity_id(
                    "legacy-session"
                ),
                "server_id": "worker-a",
                "server_uri": "http://worker-a:8000",
                "created_at": 90.0,
                "last_used_at": 95.0,
                "expires_at": 200.0,
                "version": 1,
            }
        ).encode()

        binding = await strict.resolve("tool", "legacy-session")
        assert isinstance(binding, StrictAffinityBinding)

    asyncio.run(check())


def test_bindings_are_namespaced_and_can_be_released_by_server():
    async def check():
        kv = LegacyMemoryStore()
        store = StrictAffinityBindingStore(kv, clock=lambda: 100.0)

        await store.bind(
            "tool-a",
            "same-id",
            "worker-1",
            "http://worker-1:8000",
        )
        await store.bind(
            "tool-b",
            "same-id",
            "worker-1",
            "http://worker-1:8000",
        )
        await store.bind(
            "tool-a",
            "another-id",
            "worker-2",
            "http://worker-2:8000",
        )

        assert len(await store.list_bindings("tool-a")) == 2
        assert len(await store.list_bindings("tool-b")) == 1
        assert await store.release_server("worker-1", service="tool-a") == 1
        assert await store.resolve("tool-a", "same-id") is None
        assert await store.resolve("tool-b", "same-id") is not None
        assert await store.release("tool-b", "same-id") is True

    asyncio.run(check())


def test_invalid_binding_payload_is_rejected():
    async def check():
        kv = LegacyMemoryStore()
        store = StrictAffinityBindingStore(kv)
        key = store.binding_key("tool", "session")
        kv.values[key] = b"not-json"

        with pytest.raises(InvalidAffinityBinding):
            await store.resolve("tool", "session")

    asyncio.run(check())


def test_filesystem_store_supports_ttl_and_prefix_listing(
    tmp_path,
    monkeypatch,
):
    async def check():
        now = [1_000.0]
        monkeypatch.setattr(
            "literegistry.kvstore.time.time",
            lambda: now[0],
        )
        kv = FileSystemKVStore(tmp_path / "registry")
        await kv.set("affinity:one", "one", ttl_seconds=5)
        await kv.set("affinity:two", "two")
        await kv.set("server_one", "server")

        assert set(await kv.keys(prefix="affinity:")) == {
            "affinity:one",
            "affinity:two",
        }

        now[0] = 1_006.0
        assert await kv.get("affinity:one") is None
        assert await kv.exists("affinity:one") is False
        assert await kv.keys(prefix="affinity:") == ["affinity:two"]
        assert not (kv.root / "affinity:one").exists()

    asyncio.run(check())


def test_redis_store_uses_native_ttl_and_prefix_scan():
    async def check():
        client = FakeRedis()
        kv = RedisKVStore("redis://unused")
        kv._redis = client

        assert await kv.set(
            "affinity:tool:two",
            "payload",
            ttl_seconds=1.25,
        )
        assert client.set_calls == [
            (
                "affinity:tool:two",
                b"payload",
                {"px": 1250},
            )
        ]

        keys = await kv.keys(prefix="affinity:tool:")
        assert keys == [
            "affinity:tool:one",
            "affinity:tool:two",
        ]
        assert client.scan_matches == ["affinity:tool:*"]

    asyncio.run(check())


@pytest.mark.parametrize("ttl", [0, -1, float("inf"), float("nan")])
def test_invalid_ttls_are_rejected(tmp_path, ttl):
    async def check():
        kv = FileSystemKVStore(tmp_path / "registry")
        store = StrictAffinityBindingStore(kv)

        with pytest.raises(ValueError):
            await kv.set("key", "value", ttl_seconds=ttl)
        with pytest.raises(ValueError):
            await store.bind(
                "tool",
                "session",
                "worker",
                "http://worker:8000",
                ttl_seconds=ttl,
            )

    asyncio.run(check())
