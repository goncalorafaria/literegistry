import asyncio
from io import BytesIO
import json
from types import SimpleNamespace

from literegistry.console import live, parser
from literegistry.coop.endpoints import get_endpoint_registry, EndpointRecord


def test_head_discovers_distinct_live_gateways_and_deduplicates_urls(tmp_path, monkeypatch):
    location = f"sqlite://{tmp_path}/head.sqlite3"

    async def publish():
        registry = get_endpoint_registry(location)
        try:
            await registry.publish("gateway", "http://one:8080", publisher_id="bootstrap")
            await registry.publish("gateway-monitor", "http://one:8080", publisher_id="one")
            await registry.publish("gateway-monitor", "http://two:8080", publisher_id="two")
            await registry.publish("redis", "redis://registry:6379", publisher_id="redis")
        finally:
            await registry.close()

    asyncio.run(publish())
    monkeypatch.setattr(live, "_snapshot", lambda record: record.uri)
    assert set(live.poll_head_gateways("head+" + location)) == {"http://one:8080", "http://two:8080"}
    assert live.poll_head_gateways(location) == []


def test_snapshot_exposes_completed_counts_and_reports_unreachable_gateway(monkeypatch):
    record = EndpointRecord("gateway", "http://one:8080", "trainer", 0)
    payload = {"total_requests": {"/v1/chat/completions": 7}, "window_seconds": 5,
               "recent": {"/v1/chat/completions": {"count": 2, "average_seconds": 0.5,
                                                    "maximum_seconds": 0.8}}}
    calls = []

    def open_response(url, timeout):
        calls.append((url, timeout))
        return BytesIO(json.dumps(payload).encode())

    monkeypatch.setattr(live, "build_opener", lambda *_: SimpleNamespace(open=open_response))
    result = live._snapshot(record)
    assert result["error"] is None
    assert result["rows"][0] == {
        "gateway": "trainer", "url": "http://one:8080", "route": "/v1/chat/completions",
        "completed_total": 7, "recent_completed": 2, "window_seconds": 5,
        "average_seconds": 0.5, "maximum_seconds": 0.8,
    }
    assert calls == [("http://one:8080/gateway-stats", 3)]

    def fail(*args, **kwargs):
        raise OSError("unreachable")

    monkeypatch.setattr(live, "build_opener", fail)
    assert live._snapshot(record)["error"] == "unreachable"


def test_registry_polling_uses_in_process_discovery_and_reports_failure(monkeypatch):
    async def view(registry, head, timeout):
        assert (registry, head, timeout) == ("head+sqlite:///head.sqlite3", None, 4)
        return SimpleNamespace(models={"model": [{}, {}]})

    monkeypatch.setattr("literegistry.cli._registry_view", view)
    rows, error = parser.poll_registry_summary_with_status("head+sqlite:///head.sqlite3")
    assert not error
    assert rows[0]["model"] == "model" and rows[0]["count"] == 2

    def fail(*args):
        raise TimeoutError("discovery timeout")

    monkeypatch.setattr(live, "registry_summary", fail)
    assert parser.poll_registry_summary_with_status("head+sqlite:///head.sqlite3") == (
        [], "Registry discovery failed: discovery timeout")


def test_recent_log_discovery_follows_symlinks(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "worker.log").write_text("ready\n")
    linked = tmp_path / "linked"
    linked.symlink_to(logs, target_is_directory=True)
    assert parser.find_recent_files(linked, ["*.log"]) == [linked / "worker.log"]
