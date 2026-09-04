import asyncio
from unittest.mock import patch

from literegistry.services.podman_server import RedisRegistration, main


class FakeStore:
    def __init__(self):
        self.pings = 0
        self.closed = False

    async def ping(self):
        self.pings += 1
        return True

    async def close(self):
        self.closed = True


class FakeRegistry:
    server_id = "registry-server-id"

    def __init__(self):
        self.registrations = []
        self.heartbeats = []
        self.deregistered = False

    async def register_server(self, url, port, metadata):
        self.registrations.append((url, port, metadata))
        return self.server_id

    async def heartbeat(self, url, port):
        self.heartbeats.append((url, port))

    async def deregister(self):
        self.deregistered = True


def _registration(store=None, registry=None, heartbeat_interval=10):
    return RedisRegistration(
        registry_url="redis://registry.example:6379\\",
        advertise_host="worker.example",
        advertise_port=8091,
        instance_id="replica-a",
        image="ubuntu:test",
        registry_mirror="http://gateway.example:8080",
        session_limits={
            "max_sessions": 64,
            "memory": "4g",
            "pids_limit": 2048,
            "idle_timeout": 7200,
            "janitor_interval": 300,
            "resource_watchdog_interval": 5,
            "image_prune_until": "24h",
        },
        heartbeat_interval=heartbeat_interval,
        store=store,
        registry=registry,
    )


def test_registry_metadata_describes_affinity_without_a_secret():
    metadata = _registration(store=FakeStore(), registry=FakeRegistry()).metadata()

    assert metadata["model_path"] == "podman"
    assert metadata["instance_id"] == "replica-a"
    assert metadata["registry_mirror"] == "http://gateway.example:8080"
    assert metadata["session_limits"] == {
        "max_sessions": 64,
        "memory": "4g",
        "pids_limit": 2048,
        "idle_timeout": 7200,
        "janitor_interval": 300,
        "resource_watchdog_interval": 5,
        "image_prune_until": "24h",
    }
    assert metadata["affinity"] == {
        "enabled": True,
        "handshake_endpoint": "handshake",
        "command_endpoint": "podman",
        "close_endpoint": "close",
        "id_field": "affinity_id",
    }
    assert metadata["authentication"] == {"type": "none"}
    assert "token" not in str(metadata).lower().replace("token_in_registry", "")


def test_registration_pings_heartbeats_and_deregisters():
    async def check():
        store = FakeStore()
        registry = FakeRegistry()
        registration = _registration(
            store=store,
            registry=registry,
            heartbeat_interval=0.01,
        )

        await registration.start()
        await asyncio.sleep(0.025)
        await registration.stop()

        assert registration.registry_url == "redis://registry.example:6379"
        assert store.pings == 1
        assert registry.registrations[0][0:2] == ("http://worker.example", 8091)
        assert registry.heartbeats
        assert registry.deregistered is True
        assert store.closed is True

    asyncio.run(check())


def test_main_always_starts_exactly_one_http_worker():
    app = object()
    with patch(
        "literegistry.services.podman_server.build_registered_app",
        return_value=app,
    ), patch("literegistry.services.podman_server.uvicorn.run") as run:
        main(
            host="127.0.0.1",
            port=8091,
            advertise_host="127.0.0.1",
            advertise_port=8091,
            registry="redis://registry.example:6379",
            image="ubuntu:test",
            network="none",
            instance_id="podman-affinity-1",
            storage_driver="vfs",
            heartbeat_interval=10.0,
        )

    run.assert_called_once_with(app, host="127.0.0.1", port=8091, workers=1)


def test_main_rejects_non_loopback_bind_without_authentication():
    try:
        main(host="0.0.0.0")
    except ValueError as exc:
        assert "loopback" in str(exc)
    else:
        raise AssertionError("main accepted a public unauthenticated bind")


def test_main_allows_explicit_managed_cluster_bind():
    app = object()
    with patch(
        "literegistry.services.podman_server.build_registered_app",
        return_value=app,
    ), patch("literegistry.services.podman_server.uvicorn.run") as run:
        main(host="0.0.0.0", port=28091, allow_non_loopback=True)

    run.assert_called_once_with(app, host="0.0.0.0", port=28091, workers=1)
