"""Tests for the instance-local affinity mock service."""

import asyncio

from literegistry.services.affinity_mock_server import (
    AffinityKVService,
    AffinityMockConfig,
    AffinitySessionNotFound,
    RegisteredAffinityMockServer,
    create_mock_app,
)


def test_five_writes_and_reads_remain_on_handshake_instance():
    async def check():
        service = AffinityKVService(instance_id="replica-a")
        handshake = await service.handshake(client_id="test-client")
        affinity_id = handshake["affinity_id"]

        assert handshake["instance_id"] == "replica-a"
        assert handshake["client_id"] == "test-client"

        expected = {f"key-{index}": f"value-{index}" for index in range(5)}
        for key, value in expected.items():
            response = await service.put(affinity_id, key, value)
            assert response["instance_id"] == "replica-a"

        for key, value in expected.items():
            response = await service.get(affinity_id, key)
            assert response["value"] == value
            assert response["instance_id"] == "replica-a"
            assert response["affinity_id"] == affinity_id

    asyncio.run(check())


def test_handshake_id_is_rejected_by_a_different_replica():
    async def check():
        replica_a = AffinityKVService(instance_id="replica-a")
        replica_b = AffinityKVService(instance_id="replica-b")
        affinity_id = (await replica_a.handshake())["affinity_id"]
        await replica_a.put(affinity_id, "key", "only-on-a")

        try:
            await replica_b.get(affinity_id, "key")
        except AffinitySessionNotFound as exc:
            assert exc.args == (affinity_id,)
        else:
            raise AssertionError("another replica must reject the affinity ID")

        assert (await replica_a.get(affinity_id, "key"))["value"] == "only-on-a"

    asyncio.run(check())


def test_mock_app_exposes_affinity_protocol_routes():
    app = create_mock_app(AffinityKVService(instance_id="replica-a"))
    paths = {route.path for route in app.routes}

    assert {"/health", "/handshake", "/kv/put", "/kv/get"}.issubset(paths)


def test_registry_metadata_declares_affinity_capabilities():
    server = object.__new__(RegisteredAffinityMockServer)
    server.config = AffinityMockConfig(service_name="affinity-kv")
    server.service = AffinityKVService(instance_id="replica-a")

    metadata = server.metadata()

    assert metadata["model_path"] == "affinity-kv"
    assert metadata["instance_id"] == "replica-a"
    assert metadata["affinity"] == {
        "enabled": True,
        "handshake_endpoint": "handshake",
        "id_field": "affinity_id",
    }


def test_advertised_host_is_independent_from_bind_host(tmp_path):
    server = RegisteredAffinityMockServer(
        AffinityMockConfig(
            registry=str(tmp_path),
            host="0.0.0.0",
            advertise_host="replica-a.internal",
            port=8091,
            instance_id="replica-a",
        )
    )

    assert server.config.host == "0.0.0.0"
    assert server.url == "http://replica-a.internal"
