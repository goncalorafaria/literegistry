from __future__ import annotations

import asyncio
import time

import pytest

from literegistry import cli
from literegistry.console.parser import parse_registry_summary
from literegistry.coop.endpoints import EndpointRecord


class _FakeStore:
    def __init__(self) -> None:
        self.current_url = "redis://reachable-live:6380"
        self.closed = False

    async def close(self) -> None:
        self.closed = True


class _FakeClient:
    def __init__(self, store: _FakeStore) -> None:
        self.store = store

    async def models(self, force: bool = False):
        assert force is True
        return {
            "podman": [
                {
                    "uri": "http://podman-1:8091",
                    "metadata": {"instance_id": "podman-1"},
                }
            ]
        }


class _FakeEndpointRegistry:
    def __init__(self, record: EndpointRecord) -> None:
        self.record = record
        self.closed = False

    async def get(self, name: str) -> EndpointRecord:
        assert name == "redis"
        return self.record

    async def close(self) -> None:
        self.closed = True


def test_registry_view_resolves_head_to_published_live_redis(monkeypatch) -> None:
    store = _FakeStore()
    record = EndpointRecord(
        name="redis",
        uri="redis://user:secret@published-live:6380",
        publisher_id="redis-job-1",
        published_at=time.time(),
    )
    endpoints = _FakeEndpointRegistry(record)
    seen: dict[str, str] = {}

    def fake_store(target: str):
        seen["target"] = target
        return store

    def fake_endpoints(target: str):
        seen["head"] = target
        return endpoints

    monkeypatch.setattr(cli, "get_kvstore", fake_store)
    monkeypatch.setattr(cli, "RegistryClient", _FakeClient)
    monkeypatch.setattr(cli, "get_endpoint_registry", fake_endpoints)

    view = asyncio.run(
        cli._registry_view(
            None,
            "sqlite:///weka/shared/head.sqlite3",
            timeout=1,
        )
    )

    assert seen == {
        "target": "head+sqlite:///weka/shared/head.sqlite3",
        "head": "sqlite:///weka/shared/head.sqlite3",
    }
    assert view.head_registry == "head+sqlite:///weka/shared/head.sqlite3"
    assert view.live_registry == record.uri
    assert view.models["podman"][0]["uri"] == "http://podman-1:8091"
    assert store.closed is True
    assert endpoints.closed is True


def test_summary_prints_resolution_and_redacts_redis_password(
    monkeypatch,
    capsys,
) -> None:
    record = EndpointRecord(
        name="redis",
        uri="redis://user:secret@live.example:6380",
        publisher_id="redis-job-2",
        published_at=time.time(),
    )

    async def fake_view(*_args, **_kwargs):
        return cli.RegistryView(
            models={"podman": [{}, {}], "docker-mirror": [{}]},
            head_registry="head+redis://head-user:head-secret@head.example:6379",
            live_registry=record.uri,
            endpoint=record,
        )

    monkeypatch.setattr(cli, "_registry_view", fake_view)
    cli.check_summary(head_registry="redis://unused:6379")
    output = capsys.readouterr().out

    assert "Head registry: head+redis://head-user:***@head.example:6379" in output
    assert "Live registry: redis://user:***@live.example:6380" in output
    assert "Redis publisher: redis-job-2" in output
    assert "podman :2" in output
    assert "docker-mirror :1" in output
    assert "secret" not in output
    assert [
        (row["model"], row["count"])
        for row in parse_registry_summary(output, ts=1)
    ] == [("podman", 2), ("docker-mirror", 1)]


def test_detail_uses_models_loaded_from_live_registry(monkeypatch, capsys) -> None:
    async def fake_view(*_args, **_kwargs):
        return cli.RegistryView(
            models={
                "podman": [
                    {
                        "uri": "http://podman.example:8091",
                        "metadata": {"instance_id": "podman-7"},
                    }
                ]
            }
        )

    monkeypatch.setattr(cli, "_registry_view", fake_view)
    cli.check_detail(registry="redis://live.example:6379")
    output = capsys.readouterr().out

    assert "podman : 1" in output
    assert "http://podman.example:8091" in output
    assert "podman-7" in output


def test_registry_target_rejects_two_sources_and_timeout_must_be_positive() -> None:
    with pytest.raises(ValueError, match="only one"):
        cli._registry_target("redis://live:6379", "file:///weka/head")
    with pytest.raises(ValueError, match="timeout must be positive"):
        asyncio.run(cli._registry_view("redis://live:6379", None, timeout=0))
