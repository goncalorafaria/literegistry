from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

from literegistry_podman_beaker import live_fire


class _FakeSession:
    def __init__(self, image: str, tracker: dict[str, object]) -> None:
        self.image = image
        self.instance_id = "podman-test"
        self.tracker = tracker
        self.closed = False

    async def execute(self, command: str, **kwargs):
        self.tracker["commands"].append((self.image, command))
        await asyncio.sleep(0)
        return SimpleNamespace(
            success=command != "false",
            timed_out=False,
            stdout_truncated=False,
            stderr_truncated=False,
        )

    async def close(self):
        self.closed = True
        self.tracker["closed"].append(self.image)
        self.tracker["active"] -= 1
        return {"removed": True}


class _FakeClient:
    tracker: dict[str, object] = {}

    def __init__(self, gateway_url: str, **kwargs) -> None:
        self.gateway_url = gateway_url
        self.tracker["connection_limit"] = kwargs["http_session"].connector.limit

    async def open(self):
        return self

    async def aclose(self) -> None:
        self.tracker["client_closed"] = True

    async def handshake(self, *, image: str, client_id: str):
        self.tracker["active"] += 1
        self.tracker["peak"] = max(self.tracker["peak"], self.tracker["active"])
        await asyncio.sleep(0)
        return _FakeSession(image, self.tracker)


def test_live_fire_replays_commands_and_closes_sessions(
    monkeypatch, tmp_path: Path
) -> None:
    manifest = tmp_path / "workloads.jsonl"
    rows = [
        {
            "task_id": f"task-{index}",
            "container_image": f"demo/image:{index}",
            "commands": ["echo first", "false", "echo last"],
        }
        for index in range(4)
    ]
    manifest.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    checkpoint = tmp_path / "complete.txt"
    _FakeClient.tracker = {
        "commands": [],
        "closed": [],
        "active": 0,
        "peak": 0,
        "client_closed": False,
    }
    monkeypatch.setattr(live_fire, "PodmanClient", _FakeClient)
    monkeypatch.setattr(live_fire, "_wait_for_podman", lambda *args: _async_value(32))

    result = asyncio.run(
        live_fire.live_fire(
            "http://gateway.example:8080",
            str(manifest),
            concurrency=2,
            total=4,
            expected_podman=32,
            retries=1,
            checkpoint_file=str(checkpoint),
            log_every=1,
        )
    )

    assert result["successes"] == 4
    assert result["failures"] == 0
    assert result["commands_completed"] == 12
    assert result["nonzero_commands"] == 4
    assert result["instance_counts"] == {"podman-test": 4}
    assert result["peak_live_sessions"] <= 2
    assert _FakeClient.tracker["connection_limit"] == 2
    assert len(_FakeClient.tracker["closed"]) == 4
    assert _FakeClient.tracker["client_closed"] is True
    assert set(checkpoint.read_text(encoding="utf-8").splitlines()) == {
        row["task_id"] for row in rows
    }


async def _async_value(value):
    return value

