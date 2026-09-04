import asyncio
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from literegistry.cache_server import (
    CacheRequest,
    CacheServer,
    CacheServerConfig,
    CacheServiceClient,
)


class FakeRegistry:
    def __init__(self):
        self.registrations = []
        self.heartbeats = 0
        self.deregistrations = 0

    async def register_server(self, *args):
        self.registrations.append(args)

    async def heartbeat(self, *args):
        self.heartbeats += 1

    async def deregister(self):
        self.deregistrations += 1


class FakeRedis:
    def __init__(self):
        self.values = {}
        self.expirations = {}
        self.pings = 0
        self.closed = False

    async def ping(self):
        self.pings += 1
        return True

    async def get(self, key):
        return self.values.get(key)

    async def set(self, key, value, ex=None):
        self.values[key] = value
        self.expirations[key] = ex
        return True

    async def delete(self, key):
        return int(self.values.pop(key, None) is not None)

    async def aclose(self):
        self.closed = True


class FakeHTTP:
    def __init__(self):
        self.requests = []
        self.started = False
        self.closed = False

    async def __aenter__(self):
        self.started = True
        return self

    async def __aexit__(self, *args):
        self.closed = True

    async def request_with_rotation(self, endpoint, payload):
        self.requests.append((endpoint, payload))
        if payload["operation"] == "get":
            return {"success": True, "hit": True, "value": {"answer": 42}}, 0
        return {"success": True, "hit": True}, 0


def make_server(**overrides):
    config = CacheServerConfig(
        registry="redis://controller.example:6379",
        backend_redis="redis://private-cache.example:6379/0",
        heartbeat_interval=0.001,
        **overrides,
    )
    registry = FakeRegistry()
    backend = FakeRedis()
    with (
        patch("literegistry.cache_server.get_kvstore"),
        patch(
            "literegistry.cache_server.ServerRegistry",
            return_value=registry,
        ),
    ):
        server = CacheServer(config)

    async def start_fake_backend():
        server.cache = backend
        await backend.ping()

    server._start_backend = start_fake_backend
    return server, registry, backend


def test_cache_request_validates_operation_fields():
    with pytest.raises(ValidationError):
        CacheRequest(operation="set", key="key")
    with pytest.raises(ValidationError):
        CacheRequest(operation="get", key="key", ttl_seconds=10)


def test_cache_server_set_get_delete_round_trip():
    server, _, backend = make_server()

    async def scenario():
        await server.start()
        stored = await server.execute(
            CacheRequest(
                operation="set",
                key="search-key",
                value={"results": [1, 2, 3]},
                ttl_seconds=90,
            )
        )
        found = await server.execute(
            CacheRequest(operation="get", key="search-key")
        )
        deleted = await server.execute(
            CacheRequest(operation="delete", key="search-key")
        )
        missing = await server.execute(
            CacheRequest(operation="get", key="search-key")
        )
        await server.cleanup_async()
        return stored, found, deleted, missing

    stored, found, deleted, missing = asyncio.run(scenario())

    assert stored.hit is True
    assert found.hit is True
    assert found.value == {"results": [1, 2, 3]}
    assert deleted.hit is True
    assert missing.hit is False
    storage_key = "literegistry-cache:v1:search-key"
    assert backend.expirations[storage_key] == 90
    assert backend.closed is True


def test_cache_server_registers_and_heartbeats_only_as_http_service():
    server, registry, _ = make_server()

    async def scenario():
        await server.start()
        await asyncio.sleep(0.01)
        await server.cleanup_async()

    asyncio.run(scenario())

    assert len(registry.registrations) == 1
    url, port, metadata = registry.registrations[0]
    assert url.startswith("http://")
    assert port == 1215
    assert metadata["model_path"] == "cache"
    assert metadata["backend"] == "http-cache"
    assert registry.heartbeats > 0
    assert registry.deregistrations == 1


def test_managed_backend_is_loopback_only_and_evicting():
    config = CacheServerConfig(
        registry="redis://controller.example:6379",
        backend_redis=None,
        backend_port=6380,
        maxmemory="2gb",
        maxmemory_policy="allkeys-lfu",
    )
    with (
        patch("literegistry.cache_server.get_kvstore"),
        patch(
            "literegistry.cache_server.ServerRegistry",
            return_value=FakeRegistry(),
        ),
        patch(
            "literegistry.cache_server.shutil.which",
            return_value="/usr/bin/redis-server",
        ),
    ):
        server = CacheServer(config)
        command = server._backend_command()

    assert command == [
        "/usr/bin/redis-server",
        "--bind",
        "127.0.0.1",
        "--protected-mode",
        "yes",
        "--save",
        "",
        "--appendonly",
        "no",
        "--port",
        "6380",
        "--maxmemory",
        "2gb",
        "--maxmemory-policy",
        "allkeys-lfu",
    ]


def test_cache_service_client_uses_registered_http_api():
    client = CacheServiceClient(registry=object())
    fake_http = FakeHTTP()
    client.http = fake_http

    async def scenario():
        await client.start()
        found = await client.get("search-key")
        stored = await client.set("search-key", {"answer": 42}, ttl_seconds=60)
        await client.close()
        return found, stored

    found, stored = asyncio.run(scenario())

    assert found.hit is True
    assert found.value == {"answer": 42}
    assert stored.hit is True
    assert fake_http.requests == [
        ("cache", {"operation": "get", "key": "search-key"}),
        (
            "cache",
            {
                "operation": "set",
                "key": "search-key",
                "value": {"answer": 42},
                "ttl_seconds": 60,
            },
        ),
    ]
    assert fake_http.started is True
    assert fake_http.closed is True
