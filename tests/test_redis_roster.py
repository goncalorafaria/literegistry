"""Redis roster integration tests (host Redis or LITEREGISTRY_TEST_REDIS_IMAGE)."""

import asyncio
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from unittest.mock import AsyncMock

import pytest
import redis.asyncio as redis

from literegistry.redis import RedisKVStore
from literegistry.registry import ServerRegistry


@pytest.fixture
def redis_socket():
    executable = shutil.which("redis-server")
    image = os.environ.get("LITEREGISTRY_TEST_REDIS_IMAGE")
    if executable:
        command = [executable]
    elif image and shutil.which("apptainer"):
        command = ["apptainer", "exec", "--cleanenv", image, "redis-server"]
    else:
        pytest.skip("requires redis-server or LITEREGISTRY_TEST_REDIS_IMAGE")
    with tempfile.TemporaryDirectory(prefix="roster-test-") as directory:
        path = Path(directory) / "redis.sock"
        process = subprocess.Popen(
            [*command, "--port", "0", "--unixsocket", str(path),
             "--save", "", "--appendonly", "no"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        try:
            deadline = time.monotonic() + 10
            while not path.exists():
                if process.poll() is not None:
                    pytest.fail(process.stderr.read().decode())
                if time.monotonic() >= deadline:
                    pytest.fail("Redis startup timed out")
                time.sleep(0.02)
            yield str(path)
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            process.stderr.close()


def test_indexed_roster_lifecycle(redis_socket, monkeypatch):
    async def scenario():
        client = redis.Redis(unix_socket_path=redis_socket)
        store = RedisKVStore(raise_on_error=True)
        store._redis = client
        now = [1000.0]
        monkeypatch.setattr("literegistry.registry.time.time", lambda: now[0])
        first = ServerRegistry(store, max_heartbeat_interval=20)
        second = ServerRegistry(store, max_heartbeat_interval=20)
        key = f"server_{first.server_id}"
        index = store.SERVER_HEARTBEATS_KEY
        try:
            # Neither populated nor empty roster reads may scan the database.
            monkeypatch.setattr(client, "scan_iter", lambda **kw: pytest.fail("roster scanned"))
            assert await first.roster() == {"servers": []}
            async with client.pipeline() as pipe:
                for number in range(10000):
                    pipe.set(f"affinity:mirror:{number}", b"binding")
                await pipe.execute()
            await first.register_server("http://first", 8001, {"model": "one"})
            now[0] = 1010.0
            await second.register_server("http://second", 8002)
            assert await client.zscore(index, key) == 1000.0
            now[0] = 1020.0
            assert len((await first.roster())["servers"]) == 2  # Inclusive cutoff.
            now[0] = 1021.0
            get = AsyncMock(wraps=store.get)
            monkeypatch.setattr(store, "get", get)
            assert [s["port"] for s in (await first.roster())["servers"]] == [8002]
            assert [call.args[0] for call in get.call_args_list] == [f"server_{second.server_id}"]
            # A wider window must still find older entries.
            first.max_heartbeat_interval = 100
            assert len((await first.roster())["servers"]) == 2
            first.max_heartbeat_interval = 20
            await first.heartbeat("http://first", 8001, {"requests": 3})
            assert await client.zscore(index, key) == 1021.0
            assert json.loads(await client.get(key))["data"] == {"requests": 3}
            assert len((await first.roster())["servers"]) == 2
            await second.deregister()
            assert await client.zscore(index, f"server_{second.server_id}") is None
            assert not await client.exists(f"server_{second.server_id}")
            # Index loss and legacy records recover on the next heartbeat.
            await client.delete(index)
            assert (await first.roster())["servers"] == []
            await first.heartbeat("http://first", 8001)
            assert len((await first.roster())["servers"]) == 1
            # Missing and corrupt records cannot become routable through the index.
            await client.set("server_corrupt", b"invalid json")
            await client.zadd(index, {"server_missing": now[0], "server_corrupt": now[0]})
            assert len((await first.roster())["servers"]) == 1
            await client.delete(key)
            await first.heartbeat("http://first", 8001)
            assert json.loads(await client.get(key))["metadata"] == {"model": "one"}
            await first.deregister()
            assert (await first.roster())["servers"] == []
        finally:
            await store.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("raise_on_error", [False, True])
def test_index_errors_follow_store_policy(raise_on_error):
    async def scenario():
        store = RedisKVStore(raise_on_error=raise_on_error)
        store._redis = AsyncMock()
        store._redis.zrangebyscore.side_effect = RuntimeError("unavailable")
        if raise_on_error:
            with pytest.raises(RuntimeError, match="unavailable"):
                await store.active_server_keys(0)
        else:
            assert await store.active_server_keys(0) == []

    asyncio.run(scenario())


def test_indexed_endpoint_lifecycle(redis_socket, monkeypatch):
    from literegistry.coop.endpoints import EndpointRegistry

    async def scenario():
        client = redis.Redis(unix_socket_path=redis_socket)
        store = RedisKVStore(raise_on_error=True)
        store._redis = client
        registry = EndpointRegistry(store, location="test")
        now = [1000.0]
        monkeypatch.setattr("literegistry.coop.endpoints.time.time", lambda: now[0])
        try:
            async with client.pipeline() as pipe:
                for number in range(10000):
                    pipe.set(f"affinity:example:{number}", b"binding")
                await pipe.execute()
            await registry.publish("gateway", "http://first", publisher_id="first")
            monkeypatch.setattr(client, "scan_iter", lambda **kw: pytest.fail("endpoint scanned"))
            now[0] += 1
            await registry.publish("gateway", "http://second", publisher_id="second")
            assert (await registry.get("gateway")).uri == "http://second"
            # A malformed publisher cannot obscure another live publisher.
            await store.set_endpoint(registry.prefix("gateway"),
                                     registry.key("gateway", "bad"), "invalid", ttl_seconds=30)
            assert (await registry.get("gateway")).uri == "http://second"
            assert await registry.delete("gateway", publisher_id="first")
            assert (await registry.get("gateway")).publisher_id == "second"
            # Native TTL expiration, independent of publisher/reader wall clocks.
            await client.pexpire(registry.key("gateway", "second"), 1)
            await asyncio.sleep(0.02)
            assert await registry.get("gateway") is None
            assert await client.zscore(store._endpoint_index(registry.prefix("gateway")),
                                       registry.key("gateway", "second")) is None
            assert await registry.delete("gateway")  # malformed record
            assert await registry.get("gateway") is None
            assert not await registry.delete("gateway")
            assert await registry.get("gateway") is None  # empty marker avoids SCAN
        finally:
            await registry.close()

    asyncio.run(scenario())


def test_legacy_endpoint_scan_until_republication(redis_socket, monkeypatch):
    from literegistry.coop.endpoints import EndpointRegistry

    async def scenario():
        client = redis.Redis(unix_socket_path=redis_socket)
        store = RedisKVStore(raise_on_error=True)
        store._redis = client
        registry = EndpointRegistry(store, location="test")
        try:
            record = await registry.publish("gateway", "http://legacy", publisher_id="old")
            await client.delete(store._endpoint_index(registry.prefix("gateway")))
            scan = AsyncMock(wraps=store.keys)
            monkeypatch.setattr(store, "keys", scan)
            assert await registry.get("gateway") == record
            scan.assert_awaited_once()
            await registry.publish("gateway", "http://updated", publisher_id="old")
            scan.reset_mock()
            assert (await registry.get("gateway")).uri == "http://updated"
            scan.assert_not_awaited()
        finally:
            await registry.close()

    asyncio.run(scenario())
