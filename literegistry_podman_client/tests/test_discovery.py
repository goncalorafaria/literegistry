import asyncio
from contextlib import asynccontextmanager

import aiohttp
from aiohttp import web
import pytest

from literegistry_podman_client import GatewayDiscovery, PodmanClient, PodmanGatewayError


def test_discovery_shared_by_1024_callers_and_failure_generations():
    async def scenario():
        calls = 0

        async def lookup():
            nonlocal calls
            calls += 1
            await asyncio.sleep(0.01)
            return "http://gateway/"

        discovery = GatewayDiscovery(lookup)
        first = await asyncio.gather(*(discovery.resolve() for _ in range(1024)))
        assert set(first) == {("http://gateway", 1)}
        for _ in range(10):
            assert await discovery.resolve() == first[0]
        assert calls == 1
        discovery.invalidate(*first[0])
        second = await asyncio.gather(*(discovery.resolve() for _ in range(1024)))
        assert set(second) == {("http://gateway", 2)}
        assert calls == 2
        discovery.invalidate(*first[0])  # Late failure, even at the same URL.
        assert await discovery.resolve() == second[0]
        assert calls == 2

    asyncio.run(scenario())


def test_canceled_waiter_and_failed_discovery_are_shared():
    async def scenario():
        started, release = asyncio.Event(), asyncio.Event()
        calls = 0

        async def lookup():
            nonlocal calls
            calls += 1
            started.set()
            await release.wait()
            if calls == 1:
                raise LookupError("not published")
            return "http://gateway"

        discovery = GatewayDiscovery(lookup)
        canceled = asyncio.create_task(discovery.resolve())
        await started.wait()
        waiters = [asyncio.create_task(discovery.resolve()) for _ in range(100)]
        await asyncio.sleep(0)
        canceled.cancel()
        with pytest.raises(asyncio.CancelledError):
            await canceled
        release.set()
        results = await asyncio.gather(*waiters, return_exceptions=True)
        assert all(isinstance(result, LookupError) for result in results)
        assert calls == 1
        assert await discovery.resolve() == ("http://gateway", 1)
        assert calls == 2

    asyncio.run(scenario())


@asynccontextmanager
async def gateway(handler):
    app = web.Application()
    app.router.add_route("*", "/{path:.*}", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    try:
        yield f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}"
    finally:
        await runner.cleanup()


def test_clients_share_gateway_http_errors_keep_selection_transport_does_not_retry():
    async def scenario():
        calls = 0
        commands = []

        async def failing(request):
            commands.append(request.path)
            if request.path == "/capacity":
                return web.json_response({"error": "worker full"}, status=503)
            # The command reached the server, but its response was lost.
            request.transport.close()
            return web.json_response({"ok": True})

        async def healthy(request):
            commands.append(request.path)
            return web.json_response({"ok": True})

        async with gateway(failing) as first, gateway(healthy) as second:
            async def lookup():
                nonlocal calls
                calls += 1
                await asyncio.sleep(0)
                return first if calls == 1 else second

            discovery = GatewayDiscovery(lookup)
            clients = [PodmanClient(gateway_discovery=discovery) for _ in range(20)]
            try:
                await asyncio.gather(*(client.open() for client in clients))
                assert calls == 1
                assert all(client.mirror_url == first for client in clients)
                with pytest.raises(PodmanGatewayError) as error:
                    await clients[0]._request("POST", "capacity", {})
                assert error.value.status_code == 503
                await clients[1].open()
                assert calls == 1
                with pytest.raises(PodmanGatewayError) as error:
                    await clients[0]._request("POST", "execute", {})
                assert isinstance(error.value.__cause__, aiohttp.ClientError)
                assert commands == ["/capacity", "/execute"]
                # Rediscovery is deferred until the NEXT operation.
                assert calls == 1
                result = await clients[1]._request("POST", "next", {})
                assert result == {"ok": True}
                assert calls == 2
                assert commands == ["/capacity", "/execute", "/next"]
                await asyncio.gather(*(client.open() for client in clients))
                assert calls == 2
            finally:
                await asyncio.gather(*(client.aclose() for client in clients))

    asyncio.run(scenario())


def test_registry_discovery_closes_store_and_validates_url(tmp_path):
    FilesystemEndpointRegistry = pytest.importorskip(
        "literegistry.coop.endpoints"
    ).FilesystemEndpointRegistry

    async def scenario():
        discovery = GatewayDiscovery.from_registry(str(tmp_path))
        with pytest.raises(LookupError):
            await discovery.resolve()
        registry = FilesystemEndpointRegistry(tmp_path)
        try:
            await registry.publish("gateway", "http://gateway", publisher_id="one")
            assert await discovery.resolve() == ("http://gateway", 1)
            await registry.publish("gateway", "http://new", publisher_id="two")
            assert await discovery.resolve() == ("http://gateway", 1)
            discovery.invalidate("http://gateway", 1)
            assert await discovery.resolve() == ("http://new", 2)
        finally:
            await registry.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("url", ["redis://host:6379", "relative", ""])
def test_invalid_discovered_url(url):
    async def lookup():
        return url

    with pytest.raises(ValueError, match="absolute HTTP"):
        asyncio.run(GatewayDiscovery(lookup).resolve())


def test_registry_timeout_cancels_lookup_and_closes_store(monkeypatch):
    from unittest.mock import AsyncMock
    pytest.importorskip("literegistry.coop.endpoints")
    store = type("Registry", (), {})()
    store.get = AsyncMock(side_effect=lambda *args: None)
    store.close = AsyncMock()

    async def slow(name):
        await asyncio.sleep(10)

    store.get.side_effect = slow
    monkeypatch.setattr("literegistry.coop.endpoints.get_endpoint_registry", lambda _: store)
    discovery = GatewayDiscovery.from_registry("unused", timeout=0.01)
    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(discovery.resolve())
    store.close.assert_awaited_once()


def test_request_timeout_invalidates_without_reissuing():
    async def scenario():
        calls = 0
        commands = 0

        async def slow(request):
            nonlocal commands
            commands += 1
            await asyncio.sleep(0.05)
            return web.json_response({"ok": True})

        async with gateway(slow) as url:
            async def lookup():
                nonlocal calls
                calls += 1
                return url

            discovery = GatewayDiscovery(lookup)
            async with PodmanClient(
                gateway_discovery=discovery, request_timeout=0.01
            ) as client:
                with pytest.raises(PodmanGatewayError) as error:
                    await client._request("POST", "execute", {})
                assert isinstance(error.value.__cause__, asyncio.TimeoutError)
                assert commands == calls == 1
                await client.open()
                assert calls == 2
                assert commands == 1

    asyncio.run(scenario())


def test_streaming_rediscovery_keeps_command_and_cursor_without_restarting():
    async def scenario():
        lookups = 0
        starts = 0
        first_output_sent = False
        failed_cursors = []
        resumed_cursors = []
        affinity_id = "a" * 64
        command_id = "c" * 32

        def batch(cursor, data, done):
            return {
                "chunks": [{"stream": "stdout", "data": data}] if data else [],
                "next_cursor": cursor + 1 if data else cursor,
                "done": done, "running": not done, "has_more": False,
                "exit_code": 0 if done else None, "success": True if done else None,
                "timed_out": False, "error": None,
            }

        async def first(request):
            nonlocal starts, first_output_sent
            if request.path == "/affinity/handshake":
                return web.json_response({
                    "affinity_id": affinity_id, "container_id": affinity_id,
                    "instance_id": "worker-a", "image": "test",
                })
            if request.path == "/affinity/podman/start":
                starts += 1
                return web.json_response({"command_id": command_id}, status=202)
            assert request.path == "/affinity/podman/output"
            if not first_output_sent:
                assert request.query["cursor"] == "0"
                first_output_sent = True
                return web.json_response(batch(0, "first", False))
            failed_cursors.append(int(request.query["cursor"]))
            request.transport.close()
            return web.json_response({})

        async def second(request):
            if request.path == "/affinity/close":
                return web.json_response({"removed": True})
            assert request.path == "/affinity/podman/output"
            assert request.query["affinity_id"] == affinity_id
            assert request.query["command_id"] == command_id
            cursor = int(request.query["cursor"])
            resumed_cursors.append(cursor)
            return web.json_response(batch(cursor, "second" if cursor == 1 else "", True))

        async with gateway(first) as old_url, gateway(second) as new_url:
            async def lookup():
                nonlocal lookups
                lookups += 1
                return old_url if lookups == 1 else new_url

            discovery = GatewayDiscovery(lookup)
            async with PodmanClient(gateway_discovery=discovery) as client:
                session = await client.handshake()
                job = await session.start_command("build")
                assert (await job.read(wait_ms=0))["stdout"] == "first"
                with pytest.raises(PodmanGatewayError):
                    await job.read(wait_ms=0)
                assert lookups == starts == 1
                assert failed_cursors and set(failed_cursors) == {1}
                assert job.cursor == 1 and not session.closed

                chunks = [chunk async for chunk in job.stream(wait_ms=0)]
                assert chunks == [{"stream": "stdout", "data": "second"}]
                assert resumed_cursors == [1, 2]
                assert (await job.wait())["exit_code"] == 0
                assert lookups == 2 and starts == 1
                await session.close()

    asyncio.run(scenario())
