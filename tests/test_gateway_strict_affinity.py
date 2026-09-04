"""Gateway-level strict-affinity tests with two isolated mock replicas."""

import asyncio
import json
import logging
import tempfile

from literegistry.affinity import StrictAffinityBindingStore
from literegistry.services.affinity_mock_server import AffinityKVService
from literegistry.gateway.affinity import (
    RegistryPinnedTransport,
    StrictAffinityGateway,
    same_host_loopback_uri,
)
from literegistry.gateway import Gateway, RetryConfig
from literegistry.kvstore import FileSystemKVStore


class TwoReplicaRegistry:
    def __init__(self):
        self.model_forces = []
        self.records = [
            {
                "server_id": "server-a",
                "uri": "mock://replica-a",
                "metadata": {"model_path": "affinity-kv"},
            },
            {
                "server_id": "server-b",
                "uri": "mock://replica-b",
                "metadata": {"model_path": "affinity-kv"},
            },
        ]

    async def models(self, force=False):
        self.model_forces.append(force)
        return {"affinity-kv": list(self.records)}

    async def sample_servers(self, service, n, force=False):
        records = (await self.models(force=force)).get(service, [])
        return [(record["uri"], 1.0 / len(records)) for record in records[:n]]


class InProcessPinnedTransport:
    def __init__(self):
        self.replicas = {
            "mock://replica-a": AffinityKVService(instance_id="replica-a"),
            "mock://replica-b": AffinityKVService(instance_id="replica-b"),
        }
        self.calls = []
        self.probes = []
        self.unavailable = set()

    async def probe(self, service, server_uri):
        self.probes.append(server_uri)
        return server_uri not in self.unavailable

    async def post(self, service, server_uri, endpoint, payload, retry):
        self.calls.append((server_uri, endpoint, dict(payload)))
        if server_uri in self.unavailable:
            raise OSError(f"unreachable replica: {server_uri}")
        replica = self.replicas[server_uri]
        if endpoint == "handshake":
            return await replica.handshake(payload.get("client_id"))
        if endpoint == "kv/put":
            return await replica.put(
                payload["affinity_id"],
                payload["key"],
                payload["value"],
            )
        if endpoint == "kv/get":
            return await replica.get(payload["affinity_id"], payload["key"])
        raise AssertionError(f"unexpected endpoint: {endpoint}")


async def call_json(app, path, payload):
    body = json.dumps(payload).encode("utf-8")
    sent = []
    received = False

    async def receive():
        nonlocal received
        if not received:
            received = True
            return {"type": "http.request", "body": body, "more_body": False}
        return {"type": "http.disconnect"}

    async def send(message):
        sent.append(message)

    await app(
        {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": path,
            "raw_path": path.encode("ascii"),
            "query_string": b"",
            "root_path": "",
            "headers": [(b"content-type", b"application/json")],
            "client": ("test", 1234),
            "server": ("gateway", 80),
        },
        receive,
        send,
    )
    status = next(
        message["status"]
        for message in sent
        if message["type"] == "http.response.start"
    )
    response_body = b"".join(
        message.get("body", b"")
        for message in sent
        if message["type"] == "http.response.body"
    )
    return status, json.loads(response_body)


def test_same_host_uri_rewrites_to_loopback_and_preserves_port():
    assert same_host_loopback_uri(
        "http://gateway.example:8091/base?probe=yes",
        aliases={"gateway.example"},
    ) == "http://127.0.0.1:8091/base?probe=yes"
    assert same_host_loopback_uri(
        "http://remote.example:8091",
        aliases={"gateway.example"},
    ) == "http://remote.example:8091"


def test_registry_pinned_transport_uses_loopback_for_same_host():
    captured = {}

    class RecordingClient:
        def __init__(self, registry, service, **kwargs):
            captured["registry"] = registry
            captured["service"] = service

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def request_server(self, server_uri, endpoint, payload):
            captured["server_uri"] = server_uri
            captured["endpoint"] = endpoint
            captured["payload"] = payload
            return {"ok": True}

    async def check():
        registry = object()
        transport = RegistryPinnedTransport(
            registry,
            client_factory=RecordingClient,
            host_aliases={"gateway.example"},
        )
        result = await transport.post(
            "affinity-kv",
            "http://gateway.example:8091",
            "handshake",
            {"client_id": "client-a"},
            RetryConfig(timeout=2, max_retries=1),
        )

        assert result == {"ok": True}
        assert captured["registry"] is registry
        assert captured["service"] == "affinity-kv"
        assert captured["server_uri"] == "http://127.0.0.1:8091"
        assert captured["endpoint"] == "handshake"
        assert captured["payload"] == {"client_id": "client-a"}

    asyncio.run(check())


def test_registry_pinned_transport_probe_reports_reachability():
    """probe() is a plain GET /health against the exact URI: any HTTP answer
    (even 401 from a token-protected replica) means alive; connection
    failures and timeouts mean unreachable."""
    from aiohttp import web

    async def check():
        async def health(request):
            return web.json_response({"detail": "token required"}, status=401)

        app = web.Application()
        app.router.add_get("/health", health)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = runner.addresses[0][1]
        transport = RegistryPinnedTransport(
            registry=None, host_aliases={"localhost"}, probe_timeout=2.0
        )
        try:
            assert await transport.probe("podman", f"http://127.0.0.1:{port}") is True
            assert await transport.probe("podman", f"http://127.0.0.1:{port}/") is True
        finally:
            await runner.cleanup()
        # Nothing listens there any more.
        assert await transport.probe("podman", f"http://127.0.0.1:{port}") is False

    asyncio.run(check())


def make_environment(root):
    registry = TwoReplicaRegistry()
    transport = InProcessPinnedTransport()
    bindings = StrictAffinityBindingStore(
        FileSystemKVStore(root),
        default_ttl_seconds=60,
    )
    strict = StrictAffinityGateway(
        registry,
        bindings,
        retry=RetryConfig(timeout=2, max_retries=1),
        transport=transport,
    )
    gateway = Gateway(
        registry,
        routes=[],
        strict_affinity=strict,
    )
    return gateway.app, registry, transport, bindings


def test_gateway_pins_handshake_five_writes_and_five_reads():
    async def check(root):
        app, registry, transport, bindings = make_environment(root)
        status, handshake = await call_json(
            app,
            "/affinity/handshake",
            {"service": "affinity-kv", "client_id": "gateway-test"},
        )
        assert status == 200
        affinity_id = handshake["affinity_id"]
        assert handshake["instance_id"] == "replica-a"
        assert True not in registry.model_forces
        registry.model_forces.clear()

        for index in range(5):
            status, response = await call_json(
                app,
                "/affinity/kv/put",
                {
                    "service": "affinity-kv",
                    "affinity_id": affinity_id,
                    "key": f"key-{index}",
                    "value": f"value-{index}",
                },
            )
            assert status == 200
            assert response["instance_id"] == "replica-a"

        for index in range(5):
            status, response = await call_json(
                app,
                "/affinity/kv/get",
                {
                    "service": "affinity-kv",
                    "affinity_id": affinity_id,
                    "key": f"key-{index}",
                },
            )
            assert status == 200
            assert response["instance_id"] == "replica-a"
            assert response["value"] == f"value-{index}"

        binding = await bindings.resolve("affinity-kv", affinity_id)
        assert binding is not None
        assert binding.server_id == "server-a"
        assert binding.server_uri == "mock://replica-a"
        assert {uri for uri, _, _ in transport.calls} == {"mock://replica-a"}
        assert len(transport.calls) == 11
        assert registry.model_forces == [False] * 10

    with tempfile.TemporaryDirectory() as root:
        asyncio.run(check(root))


def test_gateway_demonstrates_two_independent_affinity_bindings():
    async def check(root):
        app, registry, transport, bindings = make_environment(root)
        print(
            "[affinity-demo] AVAILABLE replicas=replica-a,replica-b",
            flush=True,
        )

        status, client_a = await call_json(
            app,
            "/affinity/handshake",
            {"service": "affinity-kv", "client_id": "client-a"},
        )
        assert status == 200

        # Prefer the other replica for the second, still-unbound handshake.
        registry.records.reverse()
        status, client_b = await call_json(
            app,
            "/affinity/handshake",
            {"service": "affinity-kv", "client_id": "client-b"},
        )
        assert status == 200

        sessions = {
            "client-a": client_a,
            "client-b": client_b,
        }
        assert client_a["instance_id"] == "replica-a"
        assert client_b["instance_id"] == "replica-b"
        for client, handshake in sessions.items():
            print(
                f"[affinity-demo] BOUND client={client} "
                f"instance={handshake['instance_id']} "
                f"affinity_id={handshake['affinity_id']}",
                flush=True,
            )

        # Both sessions use identical keys, but store different values on
        # their independently pinned replicas.
        for index in range(5):
            for client, handshake in sessions.items():
                value = f"{client}-value-{index}"
                status, response = await call_json(
                    app,
                    "/affinity/kv/put",
                    {
                        "service": "affinity-kv",
                        "affinity_id": handshake["affinity_id"],
                        "key": f"shared-key-{index}",
                        "value": value,
                    },
                )
                assert status == 200
                assert response["instance_id"] == handshake["instance_id"]

        for index in range(5):
            for client, handshake in sessions.items():
                expected = f"{client}-value-{index}"
                status, response = await call_json(
                    app,
                    "/affinity/kv/get",
                    {
                        "service": "affinity-kv",
                        "affinity_id": handshake["affinity_id"],
                        "key": f"shared-key-{index}",
                    },
                )
                assert status == 200
                assert response["instance_id"] == handshake["instance_id"]
                assert response["value"] == expected

        binding_a = await bindings.resolve(
            "affinity-kv", client_a["affinity_id"]
        )
        binding_b = await bindings.resolve(
            "affinity-kv", client_b["affinity_id"]
        )
        assert binding_a is not None and binding_a.server_id == "server-a"
        assert binding_b is not None and binding_b.server_id == "server-b"
        assert {uri for uri, _, _ in transport.calls} == {
            "mock://replica-a",
            "mock://replica-b",
        }
        print(
            "[affinity-demo] PASS clients remained isolated on two replicas",
            flush=True,
        )

    with tempfile.TemporaryDirectory() as root:
        asyncio.run(check(root))


def test_gateway_owns_podman_handshake_podman_and_close_routes(caplog):
    affinity_id = "a" * 64

    async def check(root):
        class PodmanRegistry:
            def __init__(self):
                self.store = FileSystemKVStore(root)

            async def models(self, force=False):
                return {
                    "podman": [
                        {
                            "server_id": "podman-a",
                            "uri": "http://podman-a:8091",
                        }
                    ]
                }

            async def sample_servers(self, service, n, force=False):
                assert service == "podman"
                return [("http://podman-a:8091", 1.0)]

        class PodmanTransport:
            def __init__(self):
                self.calls = []

            async def post(self, service, server_uri, endpoint, payload, retry):
                self.calls.append((service, server_uri, endpoint, dict(payload)))
                if endpoint == "handshake":
                    return {
                        "affinity_id": affinity_id,
                        "container_id": affinity_id,
                        "instance_id": "podman-a",
                    }
                if endpoint == "podman":
                    return {
                        "affinity_id": affinity_id,
                        "stdout": "ai2 hello\n",
                        "success": True,
                    }
                if endpoint == "close":
                    return {"affinity_id": affinity_id, "removed": True}
                raise AssertionError(f"unexpected endpoint: {endpoint}")

        registry = PodmanRegistry()
        transport = PodmanTransport()
        strict = StrictAffinityGateway(
            registry,
            StrictAffinityBindingStore(registry.store),
            transport=transport,
        )
        gateway = Gateway(
            registry,
            routes=[],
            strict_affinity=strict,
        )

        status, handshake = await call_json(
            gateway.app,
            "/affinity/handshake",
            {"service": "podman", "image": "python:3.12-slim"},
        )
        assert status == 200
        assert handshake["affinity_id"] == affinity_id

        status, podman = await call_json(
            gateway.app,
            "/affinity/podman",
            {
                "service": "podman",
                "affinity_id": affinity_id,
                "command": "echo ai2 hello",
            },
        )
        assert status == 200
        assert podman["stdout"] == "ai2 hello\n"

        status, closed = await call_json(
            gateway.app,
            "/affinity/close",
            {"service": "podman", "affinity_id": affinity_id},
        )
        assert status == 200
        assert closed["removed"] is True
        assert await strict.bindings.resolve("podman", affinity_id) is None
        assert [call[2] for call in transport.calls] == [
            "handshake",
            "podman",
            "close",
        ]

    with tempfile.TemporaryDirectory() as root:
        with caplog.at_level(logging.INFO, logger="literegistry.gateway.affinity"):
            asyncio.run(check(root))

    messages = [record.getMessage() for record in caplog.records]
    assert any(
        "mode=strict event=bound" in message
        and "service='podman'" in message
        and "server_id='podman-a'" in message
        for message in messages
    )
    assert any(
        "mode=strict event=route_complete" in message
        and "endpoint='podman'" in message
        and "binding=hit action=touch" in message
        for message in messages
    )
    assert any(
        "mode=strict event=route_complete" in message
        and "endpoint='close'" in message
        and "binding=hit action=release" in message
        for message in messages
    )
    assert all("echo ai2 hello" not in message for message in messages)


def test_unknown_affinity_id_never_falls_back_to_load_balancing():
    async def check(root):
        app, _, transport, _ = make_environment(root)
        status, response = await call_json(
            app,
            "/affinity/kv/get",
            {
                "service": "affinity-kv",
                "affinity_id": "unknown-id",
                "key": "key-0",
            },
        )
        assert status == 404
        assert "not found" in response["error"]
        assert transport.calls == []

    with tempfile.TemporaryDirectory() as root:
        asyncio.run(check(root))


def test_strict_affinity_confirms_dead_owner_before_forwarding():
    async def check(root):
        app, registry, transport, _ = make_environment(root)
        status, handshake = await call_json(
            app,
            "/affinity/handshake",
            {"service": "affinity-kv"},
        )
        assert status == 200
        affinity_id = handshake["affinity_id"]
        registry.records = [registry.records[1]]
        registry.model_forces.clear()
        transport.unavailable.add("mock://replica-a")
        calls_before = len(transport.calls)

        status, response = await call_json(
            app,
            "/affinity/kv/get",
            {
                "service": "affinity-kv",
                "affinity_id": affinity_id,
                "key": "key-0",
            },
        )
        assert status == 503
        assert "no longer registered" in response["error"]
        assert len(transport.calls) == calls_before
        assert registry.model_forces == [False, True]
        # The roster miss was confirmed against the server itself before failing.
        assert transport.probes == ["mock://replica-a"]

    with tempfile.TemporaryDirectory() as root:
        asyncio.run(check(root))


def test_strict_affinity_forwards_to_live_owner_missing_from_roster():
    """A lagging roster (missed heartbeats, registry hiccup) must not sever
    sessions whose owner is still up: the pinned server is asked directly."""

    async def check(root):
        app, registry, transport, _ = make_environment(root)
        status, handshake = await call_json(
            app,
            "/affinity/handshake",
            {"service": "affinity-kv"},
        )
        assert status == 200
        affinity_id = handshake["affinity_id"]
        status, _ = await call_json(
            app,
            "/affinity/kv/put",
            {
                "service": "affinity-kv",
                "affinity_id": affinity_id,
                "key": "key-0",
                "value": "v0",
            },
        )
        assert status == 200

        # replica-a vanishes from the roster but keeps answering.
        registry.records = [registry.records[1]]
        registry.model_forces.clear()
        transport.probes.clear()

        status, response = await call_json(
            app,
            "/affinity/kv/get",
            {
                "service": "affinity-kv",
                "affinity_id": affinity_id,
                "key": "key-0",
            },
        )
        assert status == 200
        assert response["value"] == "v0"
        assert response["instance_id"] == "replica-a"
        assert registry.model_forces == [False, True]
        assert transport.probes == ["mock://replica-a"]
        assert transport.calls[-1][0] == "mock://replica-a"

    with tempfile.TemporaryDirectory() as root:
        asyncio.run(check(root))


def test_strict_affinity_without_probe_support_keeps_rejecting_off_roster_owner():
    async def check(root):
        app, registry, transport, _ = make_environment(root)
        status, handshake = await call_json(
            app,
            "/affinity/handshake",
            {"service": "affinity-kv"},
        )
        assert status == 200
        registry.records = [registry.records[1]]
        calls_before = len(transport.calls)
        transport.probe = None  # transport predates liveness probing

        status, response = await call_json(
            app,
            "/affinity/kv/get",
            {
                "service": "affinity-kv",
                "affinity_id": handshake["affinity_id"],
                "key": "key-0",
            },
        )
        assert status == 503
        assert "no longer registered" in response["error"]
        assert len(transport.calls) == calls_before

    with tempfile.TemporaryDirectory() as root:
        asyncio.run(check(root))


def test_handshake_retries_unreachable_candidate_before_binding():
    async def check(root):
        app, registry, transport, bindings = make_environment(root)
        registry.records.reverse()
        transport.unavailable.add("mock://replica-b")

        status, handshake = await call_json(
            app,
            "/affinity/handshake",
            {"service": "affinity-kv", "client_id": "failover-test"},
        )

        assert status == 200
        assert handshake["instance_id"] == "replica-a"
        binding = await bindings.resolve(
            "affinity-kv", handshake["affinity_id"]
        )
        assert binding is not None
        assert binding.server_id == "server-a"
        assert [call[0] for call in transport.calls] == [
            "mock://replica-b",
            "mock://replica-a",
        ]

    with tempfile.TemporaryDirectory() as root:
        asyncio.run(check(root))


def test_strict_affinity_rechecks_registration_after_request_failure():
    async def check(root):
        app, registry, transport, _ = make_environment(root)
        status, handshake = await call_json(
            app,
            "/affinity/handshake",
            {"service": "affinity-kv"},
        )
        assert status == 200
        registry.model_forces.clear()
        transport.unavailable.add("mock://replica-a")
        calls_before = len(transport.calls)

        status, response = await call_json(
            app,
            "/affinity/kv/get",
            {
                "service": "affinity-kv",
                "affinity_id": handshake["affinity_id"],
                "key": "key-0",
            },
        )

        assert status == 503
        assert "unavailable" in response["error"]
        assert len(transport.calls) == calls_before + 2
        assert registry.model_forces == [False, True]

    with tempfile.TemporaryDirectory() as root:
        asyncio.run(check(root))
