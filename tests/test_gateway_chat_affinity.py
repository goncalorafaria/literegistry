import asyncio
from contextlib import asynccontextmanager
from dataclasses import replace

import pytest
from aiohttp import web

from literegistry.gateway import RetryConfig, RoutingRequest
from literegistry.gateway.chat_affinity import ChatAffinityRouting
from literegistry.kvstore import FileSystemKVStore


class Registry:
    def __init__(self, store, servers):
        self.store, self.servers = store, servers
        self.reports = []

    async def get_all(self, value, force=False):
        return list(self.servers)

    def report_latency(self, server, latency, *, prob, success):
        self.reports.append((server, success))


@asynccontextmanager
async def backend(name):
    calls = []
    state = {"fail": False}

    async def respond(request):
        calls.append(await request.json())
        if state["fail"]:
            return web.json_response({"error": "busy"}, status=429)
        return web.json_response({"replica": name})

    app = web.Application()
    app.router.add_post("/v1/chat/completions", respond)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    uri = "http://127.0.0.1:" + str(site._server.sockets[0].getsockname()[1])
    try:
        yield uri, state, calls
    finally:
        await runner.cleanup()


def request(session="one", model="model"):
    return RoutingRequest(
        service=model,
        endpoint="v1/chat/completions",
        payload={"model": model, "messages": [{"role": "user", "content": "hello"}]},
        retry=RetryConfig(
            timeout=3, connect_timeout=1, max_retries=2, retry_backoff_seconds=0
        ),
        headers={"X-Session-ID": session},
    )


def test_chat_affinity_reuses_replica_across_gateway_workers_and_scopes_models(
    tmp_path,
):
    async def run():
        async with backend("a") as a, backend("b") as b:
            registry = Registry(FileSystemKVStore(str(tmp_path)), [a[0], b[0]])
            first = ChatAffinityRouting(registry)
            result = await first.forward(request())
            other_worker = ChatAffinityRouting(registry)
            assert (await other_worker.forward(request())).body == result.body
            assert await first.bindings.resolve("gateway-chat:model", "one") is not None
            assert await first.bindings.resolve("gateway-chat:other", "one") is None
            assert await first.bindings.resolve("gateway-chat:model", "two") is None
            assert not first.active and not other_worker.active

    asyncio.run(run())


def test_busy_missing_and_failed_preferred_replica_handoff(tmp_path):
    async def run():
        async with backend("a") as a, backend("b") as b:
            registry = Registry(FileSystemKVStore(str(tmp_path)), [a[0], b[0]])
            routing = ChatAffinityRouting(registry, load_slack=0)
            await routing.bindings.bind("gateway-chat:model", "one", a[0], a[0])
            routing.active[("model", a[0])] = 1
            assert (await routing.forward(request())).body["replica"] == "b"
            routing.active.clear()
            b[1]["fail"] = True
            assert (await routing.forward(request())).body["replica"] == "a"
            assert (
                await routing.bindings.resolve("gateway-chat:model", "one")
            ).server_uri == a[0]
            assert not routing.active
            registry.servers = [b[0]]
            b[1]["fail"] = False
            assert (await routing.forward(request())).body["replica"] == "b"

    asyncio.run(run())


def test_cancellation_releases_inflight_capacity(tmp_path, monkeypatch):
    from literegistry.http import RegistryHTTPClient

    async def cancel(*args, **kwargs):
        raise asyncio.CancelledError

    monkeypatch.setattr(RegistryHTTPClient, "_make_http_request", cancel)

    async def run():
        routing = ChatAffinityRouting(
            Registry(FileSystemKVStore(str(tmp_path)), ["http://replica"])
        )
        try:
            await routing.forward(request())
        except asyncio.CancelledError:
            pass
        else:
            raise AssertionError("cancellation swallowed")
        assert not routing.active
        assert await routing.bindings.resolve("gateway-chat:model", "one") is None

    asyncio.run(run())


def test_affinity_storage_failure_preserves_successful_generation(tmp_path):
    class BrokenBindings:
        async def resolve(self, *args):
            raise OSError("registry unavailable")

        async def handoff(self, *args):
            raise OSError("registry unavailable")

    async def run():
        async with backend("healthy") as model:
            routing = ChatAffinityRouting(
                Registry(FileSystemKVStore(str(tmp_path)), [model[0]])
            )
            routing.bindings = BrokenBindings()
            result = await routing.forward(request())
            assert result.body["replica"] == "healthy"
            assert len(model[2]) == 1
            assert not routing.active

    asyncio.run(run())


@pytest.mark.parametrize(
    "headers,endpoint",
    [
        ({}, "v1/chat/completions"),
        ({"X-Session-ID": ""}, "v1/chat/completions"),
        ({"X-Session-ID": "   "}, "v1/chat/completions"),
        ({"X-Session-ID": "one"}, "v1/completions"),
    ],
)
def test_missing_session_and_other_routes_use_normal_routing(
    tmp_path, headers, endpoint
):
    from unittest.mock import AsyncMock

    async def run():
        routing = ChatAffinityRouting(Registry(FileSystemKVStore(str(tmp_path)), []))
        routing.bindings.resolve = AsyncMock(
            side_effect=AssertionError("affinity must not run")
        )
        routing.fallback.forward = AsyncMock(return_value="normal response")
        req = replace(request(), headers=headers, endpoint=endpoint)
        assert await routing.forward(req) == "normal response"
        routing.fallback.forward.assert_awaited_once_with(req)
        routing.bindings.resolve.assert_not_awaited()

    asyncio.run(run())


def test_gateway_installs_policy_by_default_and_respects_overrides(
    tmp_path, monkeypatch
):
    from literegistry.gateway import Gateway, GatewayConfig, LoadBalancedRouting

    registry = Registry(FileSystemKVStore(str(tmp_path)), [])

    def gateway(**kwargs):
        return Gateway(
            registry, enable_strict_affinity=False, enable_docker_mirror=False, **kwargs
        )

    assert isinstance(gateway().routing, ChatAffinityRouting)
    assert isinstance(
        gateway(config=GatewayConfig(chat_soft_affinity=False)).routing,
        LoadBalancedRouting,
    )
    custom = object()
    assert gateway(routing=custom).routing is custom
    monkeypatch.setenv("CHAT_SOFT_AFFINITY", "false")
    monkeypatch.setenv("CHAT_AFFINITY_LOAD_SLACK", "0")
    config = GatewayConfig.from_env()
    assert not config.chat_soft_affinity
    assert config.chat_affinity_load_slack == 0
    with pytest.raises(ValueError):
        GatewayConfig(chat_affinity_load_slack=-1)
