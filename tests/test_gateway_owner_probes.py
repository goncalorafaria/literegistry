"""A failed pinned command is checked, never replayed onto an owner."""

import asyncio

import aiohttp
from aiohttp import web
import pytest

from literegistry.affinity import StrictAffinityBindingStore
from literegistry.gateway import Gateway, RetryConfig
from literegistry.gateway.affinity import RegistryPinnedTransport, StrictAffinityGateway
from literegistry.http import HTTPResponseError
from literegistry.kvstore import FileSystemKVStore
from tests.test_gateway_strict_affinity import TwoReplicaRegistry, call_json

CID = "a" * 64
PAYLOAD = {"service": "affinity-kv", "affinity_id": CID, "command": "append-once"}


class ProbeTransport:
    def __init__(self):
        self.posts = []
        self.probes = []
        self.failure = aiohttp.ServerDisconnectedError("lost response")
        self.health_error = None
        self.session_response = (200, {"container_id": CID, "status": "active"})

    async def post(self, service, uri, endpoint, payload, retry):
        self.posts.append((uri, endpoint))
        if self.failure:
            raise self.failure
        return {"ok": True}

    async def probe(self, uri, endpoint, timeout):
        self.probes.append((uri, endpoint))
        if endpoint == "health":
            if self.health_error:
                raise self.health_error
            return 200, {"status": "healthy"}
        return self.session_response


async def environment(root, **kwargs):
    registry = TwoReplicaRegistry()
    bindings = StrictAffinityBindingStore(FileSystemKVStore(root))
    await bindings.bind("affinity-kv", CID, "server-a", "mock://replica-a")
    transport = ProbeTransport()
    strict = StrictAffinityGateway(registry, bindings, transport=transport, **kwargs)
    return Gateway(registry, routes=[], strict_affinity=strict).app, strict, registry, transport


@pytest.mark.parametrize("failure,status", [
    (aiohttp.ServerDisconnectedError(), 502), (aiohttp.ClientPayloadError(), 502),
    (asyncio.TimeoutError(), 504), (OSError("refused"), 503),
])
def test_transport_errors_are_structured_without_replay(tmp_path, failure, status):
    async def scenario():
        app, strict, registry, transport = await environment(tmp_path)
        transport.failure = failure
        actual, body = await call_json(app, "/affinity/podman", PAYLOAD)
        assert actual == status
        assert body["execution_outcome"] == "unknown"
        assert body["request_id"]
        assert transport.posts == [("mock://replica-a", "podman")]
        assert transport.probes == [("mock://replica-a", "health"), ("mock://replica-a", f"sessions/{CID}")]
        assert registry.model_forces == [False, True]
    asyncio.run(scenario())


@pytest.mark.parametrize("upstream_status", [502, 503, 504])
def test_upstream_application_errors_are_preserved_without_probe(tmp_path, upstream_status):
    async def scenario():
        app, strict, registry, transport = await environment(tmp_path)
        body = {"detail": "overloaded"}
        transport.failure = HTTPResponseError(upstream_status, body, "mock://replica-a")
        assert await call_json(app, "/affinity/podman", PAYLOAD) == (upstream_status, body)
        assert not transport.probes
        assert len(transport.posts) == 1
        assert registry.model_forces == [False]
    asyncio.run(scenario())


@pytest.mark.parametrize("probe_status,detail,expected_reason", [
    (404, {"error": "container_not_found", "container_id": CID}, "unknown"),
    (410, {"code": "sandbox_lost", "reason": "container_stopped", "container_id": CID,
           "recoverable": False}, "container_stopped"),
])
def test_confirmed_session_loss_becomes_410(tmp_path, probe_status, detail, expected_reason):
    async def scenario():
        app, strict, registry, transport = await environment(tmp_path)
        transport.session_response = probe_status, {"detail": detail}
        status, body = await call_json(app, "/affinity/podman", PAYLOAD)
        assert status == 410
        assert body["code"] == "sandbox_lost"
        assert body["reason"] == expected_reason
        assert body["recoverable"] is False
        assert len(transport.posts) == 1
    asyncio.run(scenario())


@pytest.mark.parametrize("response", [
    (404, {"detail": "Not Found"}), (503, {"detail": "unavailable"}),
    (410, {"detail": {"code": "sandbox_lost", "container_id": "b"*64, "recoverable": False}}),
])
def test_uncertain_or_unsupported_session_probe_does_not_invent_loss(tmp_path, response):
    async def scenario():
        app, strict, registry, transport = await environment(tmp_path)
        transport.session_response = response
        status, body = await call_json(app, "/affinity/podman", PAYLOAD)
        assert status == 502
        assert body["execution_outcome"] == "unknown"
        assert "recoverable" not in body
    asyncio.run(scenario())


def test_owner_outage_cooldown_and_recovery(tmp_path):
    async def scenario():
        app, strict, registry, transport = await environment(tmp_path, probe_cooldown=0.02)
        transport.health_error = OSError("offline")
        assert (await call_json(app, "/affinity/podman", PAYLOAD))[0] == 502
        status, body = await call_json(app, "/affinity/podman", PAYLOAD)
        assert status == 503 and body["execution_outcome"] == "not_sent"
        assert len(transport.posts) == 1 and len(transport.probes) == 1
        await asyncio.sleep(0.03)
        transport.failure = None
        transport.health_error = None
        assert (await call_json(app, "/affinity/podman", PAYLOAD))[0] == 200
        assert len(transport.posts) == 2
    asyncio.run(scenario())


def test_probe_is_shared_bounded_and_survives_cancelled_waiter(tmp_path):
    async def scenario():
        app, strict, registry, transport = await environment(tmp_path, probe_timeout=0.05, max_owner_probes=1)
        started = asyncio.Event()
        async def hanging_probe(uri, endpoint, timeout):
            transport.probes.append((uri, endpoint))
            started.set()
            await asyncio.Event().wait()
        transport.probe = hanging_probe
        binding = await strict.bindings.resolve("affinity-kv", CID)
        await strict.bindings.bind("affinity-kv", "b"*64, "server-b", "mock://replica-b")
        second_binding = await strict.bindings.resolve("affinity-kv", "b"*64)
        first = asyncio.create_task(strict._probe_owner("affinity-kv", binding))
        await started.wait()
        other = asyncio.create_task(strict._probe_owner("affinity-kv", binding))
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        assert await strict._probe_owner("affinity-kv", second_binding) == "unknown"
        assert await asyncio.wait_for(other, 0.3) == "unavailable"
        assert len(transport.probes) == 1
        assert len(strict._owner_probes) == 1
        assert registry.model_forces == [True]
    asyncio.run(scenario())


def test_real_disconnected_response_does_not_repeat_side_effect(tmp_path, caplog):
    async def scenario():
        executions = []
        async def execute(request):
            executions.append(await request.json())
            request.transport.close()  # Effect happened; response never arrived.
            return web.json_response({"ok": True})
        async def health(request):
            return web.json_response({"status": "healthy"})
        async def session(request):
            return web.json_response({"container_id": CID, "status": "active"})
        upstream = web.Application()
        upstream.router.add_post("/podman", execute)
        upstream.router.add_get("/health", health)
        upstream.router.add_get("/sessions/{cid}", session)
        runner = web.AppRunner(upstream)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        uri = f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}"
        registry = TwoReplicaRegistry()
        registry.records[0]["uri"] = uri
        bindings = StrictAffinityBindingStore(FileSystemKVStore(tmp_path))
        await bindings.bind("affinity-kv", CID, "server-a", uri)
        strict = StrictAffinityGateway(registry, bindings, retry=RetryConfig(timeout=1, max_retries=1))
        gateway = Gateway(registry, routes=[], strict_affinity=strict)
        try:
            status, body = await call_json(gateway.app, "/affinity/podman", PAYLOAD)
            assert status == 502
            assert body["code"] == "affinity_upstream_disconnected"
            assert len(executions) == 1
            assert executions[0]["command"] == "append-once"
        finally:
            await runner.cleanup()
    asyncio.run(scenario())
    assert not any("Unhandled gateway error" in record.getMessage() for record in caplog.records)


def test_owner_removed_during_failure_is_confirmed_before_410(tmp_path):
    async def scenario():
        app, strict, registry, transport = await environment(tmp_path)
        async def post(*args):
            registry.records = [registry.records[1]]
            raise aiohttp.ServerDisconnectedError()
        transport.post = post
        status, body = await call_json(app, "/affinity/podman", PAYLOAD)
        assert status == 410
        assert body["code"] == "affinity_owner_lost" and body["recoverable"] is False
        assert not transport.probes
    asyncio.run(scenario())


def test_registry_read_failure_cannot_confirm_owner_loss(tmp_path):
    async def scenario():
        app, strict, registry, transport = await environment(tmp_path)
        models = registry.models
        async def failing_refresh(force=False):
            if force:
                raise OSError("registry unavailable")
            return await models(force=force)
        registry.models = failing_refresh
        status, body = await call_json(app, "/affinity/podman", PAYLOAD)
        assert status == 502
        assert body["execution_outcome"] == "unknown"
        assert not transport.probes
    asyncio.run(scenario())
