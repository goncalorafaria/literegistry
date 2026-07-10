import asyncio
import json

from literegistry.client import RegistryClient


class DummyStore:
    def __init__(self):
        self.values = {}

    async def get(self, key):
        value = self.values.get(key)
        if value is None:
            return None
        return value.encode("utf-8")

    async def set(self, key, value):
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        self.values[key] = value
        return True

    async def delete(self, key):
        return True

    async def exists(self, key):
        return False

    async def keys(self):
        return list(self.values.keys())


def test_cache_ttl_defaults_to_half_heartbeat_window():
    registry = RegistryClient(
        DummyStore(),
        max_heartbeat_interval=10,
    )

    assert registry.cache_ttl == 5


def test_cache_ttl_is_clamped_to_half_heartbeat_window():
    registry = RegistryClient(
        DummyStore(),
        cache_ttl=30,
        max_heartbeat_interval=10,
    )

    assert registry.cache_ttl == 5


def test_model_cache_refreshes_after_half_heartbeat_ttl():
    store = DummyStore()
    registry = RegistryClient(
        store,
        cache_ttl=1,
        max_heartbeat_interval=10,
    )

    async def run():
        await store.set(
            "server_one",
            json.dumps(
                {
                    "uri": "http://localhost:8000",
                    "metadata": {"model_path": "first-model"},
                    "last_heartbeat": 1_000_000_000,
                }
            ),
        )
        registry.max_heartbeat_interval = 10_000_000_000

        first = await registry.models()
        await store.set(
            "server_two",
            json.dumps(
                {
                    "uri": "http://localhost:8001",
                    "metadata": {"model_path": "second-model"},
                    "last_heartbeat": 1_000_000_000,
                }
            ),
        )
        cached = await registry.models()

        registry._cache_timestamps[registry._models_cache_key] -= 2
        refreshed = await registry.models()

        return first, cached, refreshed

    first, cached, refreshed = asyncio.run(run())

    assert list(first) == ["first-model"]
    assert list(cached) == ["first-model"]
    assert set(refreshed) == {"first-model", "second-model"}


def test_model_name_cannot_collide_with_aggregate_cache_key():
    store = DummyStore()
    registry = RegistryClient(
        store,
        cache_ttl=60,
        max_heartbeat_interval=120,
    )
    model_name = "_model_path"

    async def run():
        await store.set(
            "server_one",
            json.dumps(
                {
                    "uri": "http://localhost:8000",
                    "metadata": {"model_path": model_name},
                    "last_heartbeat": 1_000_000_000,
                }
            ),
        )
        registry.max_heartbeat_interval = 10_000_000_000

        models = await registry.models()
        uris = await registry.get_all(model_name)
        return models, uris

    models, uris = asyncio.run(run())

    assert model_name in models
    assert uris == ["http://localhost:8000"]


def test_model_specific_invalidation_also_invalidates_aggregate_models_cache():
    registry = RegistryClient(DummyStore())
    registry._cache[registry._models_cache_key] = {"model-a": []}
    registry._cache_timestamps[registry._models_cache_key] = 1
    registry._cache["model-a"] = ["http://localhost:8000"]
    registry._cache_timestamps["model-a"] = 1

    registry.invalidate_cache("model-a")

    assert "model-a" not in registry._cache
    assert registry._models_cache_key not in registry._cache
