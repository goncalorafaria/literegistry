import asyncio
import shlex
from typing import ClassVar

import pytest

from tools import podman_benchmark
from tools.podman_benchmark import (
    PodmanBenchmarkConfig,
    compare_podman_benchmark_results,
    format_podman_benchmark,
    format_podman_scaling_comparison,
    run_podman_benchmark,
)


class FakePodmanExecutionClient:
    started_count = 0
    closed_count = 0
    active_count = 0
    max_active_count = 0
    events: ClassVar[list[tuple[str, str]]] = []

    def __init__(self, gateway_url, **kwargs):
        self.gateway_url = gateway_url
        self.kwargs = kwargs
        self.client_id = kwargs["client_id"]
        self.started = False
        self.container_id = None
        self.token = None

    async def start(self):
        type(self).started_count += 1
        type(self).active_count += 1
        type(self).max_active_count = max(
            type(self).max_active_count,
            type(self).active_count,
        )
        type(self).events.append(("start", self.client_id))
        self.started = True
        self.container_id = f"container-{type(self).started_count}"
        return {
            "affinity_id": self.container_id,
            "container_id": self.container_id,
            "instance_id": f"podman-{type(self).started_count % 2}",
        }

    async def execute(self, *, command, timeout):
        del timeout
        type(self).events.append(("command", self.client_id))
        if command.startswith("printf "):
            self.token = shlex.split(command)[2]
            return {"success": True, "stdout": "", "stderr": ""}
        return {
            "success": True,
            "stdout": f"{self.token}\n",
            "stderr": "",
        }

    async def close(self):
        type(self).closed_count += 1
        type(self).active_count -= 1
        type(self).events.append(("close", self.client_id))
        self.started = False
        return {"affinity_id": self.container_id, "removed": True}


def test_podman_benchmark_isolates_phases_and_tracks_replica_traffic(monkeypatch):
    FakePodmanExecutionClient.started_count = 0
    FakePodmanExecutionClient.closed_count = 0
    FakePodmanExecutionClient.active_count = 0
    FakePodmanExecutionClient.max_active_count = 0
    FakePodmanExecutionClient.events = []
    monkeypatch.setattr(
        podman_benchmark,
        "PodmanExecutionClient",
        FakePodmanExecutionClient,
    )
    result = asyncio.run(
        run_podman_benchmark(
            PodmanBenchmarkConfig(
                gateway_url="http://gateway:8080",
                replicas=2,
                concurrency=(1, 2),
                sessions_per_worker=2,
                commands_per_session=3,
            )
        )
    )

    assert result["measurement"].startswith("barrier-separated startup")
    assert [level["requested_sessions"] for level in result["levels"]] == [2, 4]
    assert [level["successful_sessions"] for level in result["levels"]] == [2, 4]
    assert [level["completed_commands"] for level in result["levels"]] == [6, 12]
    assert all(level["failed_sessions"] == 0 for level in result["levels"])
    assert all(
        level["phases"]["startup"]["sessions_per_second"] > 0
        for level in result["levels"]
    )
    assert all(
        level["phases"]["commands"]["commands_per_second"] > 0
        for level in result["levels"]
    )
    for level, requested in zip(result["levels"], (2, 4), strict=True):
        assert sum(level["instance_distribution"].values()) == requested
        assert (
            sum(level["command_instance_distribution"].values())
            == requested * 3
        )
        assert level["traffic"]["sessions"][
            "all_replicas_received_traffic"
        ] is True
        assert level["traffic"]["commands"][
            "all_replicas_received_traffic"
        ] is True
        assert level["latency"]["write"]["count"] == requested
        assert level["latency"]["read"]["count"] == requested * 2

    for concurrency in (1, 2):
        events = [
            kind
            for kind, client_id in FakePodmanExecutionClient.events
            if f"-c{concurrency}-" in client_id
        ]
        requested = concurrency * 2
        assert events[:requested] == ["start"] * requested
        assert events[requested : requested * 4] == ["command"] * (
            requested * 3
        )
        assert events[requested * 4 :] == ["close"] * requested

    assert FakePodmanExecutionClient.started_count == 6
    assert FakePodmanExecutionClient.closed_count == 6
    assert FakePodmanExecutionClient.active_count == 0
    rendered = format_podman_benchmark(result)
    assert "Podman replicas=2" in rendered
    assert "startup-mean-ms" in rendered
    assert "command-mean-ms" in rendered
    assert "traffic(session|command)" in rendered


def test_podman_benchmark_runs_sequential_waves_without_growing_live_pool(
    monkeypatch,
):
    FakePodmanExecutionClient.started_count = 0
    FakePodmanExecutionClient.closed_count = 0
    FakePodmanExecutionClient.active_count = 0
    FakePodmanExecutionClient.max_active_count = 0
    FakePodmanExecutionClient.events = []
    monkeypatch.setattr(
        podman_benchmark,
        "PodmanExecutionClient",
        FakePodmanExecutionClient,
    )

    result = asyncio.run(
        run_podman_benchmark(
            PodmanBenchmarkConfig(
                gateway_url="http://gateway:8080",
                replicas=2,
                concurrency=(2,),
                sessions_per_worker=1,
                waves=3,
                commands_per_session=2,
            )
        )
    )

    level = result["levels"][0]
    assert level["waves"] == 3
    assert level["sessions_per_wave"] == 2
    assert level["max_live_sessions"] == 2
    assert level["requested_sessions"] == 6
    assert level["completed_commands"] == 12
    assert len(level["wave_results"]) == 3
    assert FakePodmanExecutionClient.max_active_count == 2
    assert FakePodmanExecutionClient.active_count == 0
    for wave_index in range(3):
        events = [
            kind
            for kind, client_id in FakePodmanExecutionClient.events
            if f"-w{wave_index}-" in client_id
        ]
        assert events == [
            "start",
            "start",
            "command",
            "command",
            "command",
            "command",
            "close",
            "close",
        ]


def test_podman_benchmark_keeps_total_load_fixed_across_concurrency(
    monkeypatch,
):
    FakePodmanExecutionClient.started_count = 0
    FakePodmanExecutionClient.closed_count = 0
    FakePodmanExecutionClient.active_count = 0
    FakePodmanExecutionClient.max_active_count = 0
    FakePodmanExecutionClient.events = []
    monkeypatch.setattr(
        podman_benchmark,
        "PodmanExecutionClient",
        FakePodmanExecutionClient,
    )

    result = asyncio.run(
        run_podman_benchmark(
            PodmanBenchmarkConfig(
                gateway_url="http://gateway:8080",
                replicas=2,
                concurrency=(2, 4),
                total_sessions=10,
                commands_per_session=2,
            )
        )
    )

    assert [level["requested_sessions"] for level in result["levels"]] == [
        10,
        10,
    ]
    assert [level["completed_commands"] for level in result["levels"]] == [
        20,
        20,
    ]
    assert result["measurement"].startswith(
        "rolling complete session trajectories"
    )
    assert [level["execution_model"] for level in result["levels"]] == [
        "rolling-complete-session-trajectories",
        "rolling-complete-session-trajectories",
    ]
    assert [level["max_live_sessions"] for level in result["levels"]] == [
        2,
        4,
    ]
    assert all(level["attempted_sessions"] == 10 for level in result["levels"])
    assert all(level["unattempted_sessions"] == 0 for level in result["levels"])
    assert all(
        level["phases"]["trajectory"]["successful_sessions"] == 10
        for level in result["levels"]
    )
    assert FakePodmanExecutionClient.started_count == 20
    assert FakePodmanExecutionClient.closed_count == 20
    assert FakePodmanExecutionClient.max_active_count <= 4
    assert FakePodmanExecutionClient.active_count == 0

    for concurrency in (2, 4):
        client_ids = {
            client_id
            for _, client_id in FakePodmanExecutionClient.events
            if f"-c{concurrency}-" in client_id
        }
        assert len(client_ids) == 10
        for client_id in client_ids:
            assert [
                kind
                for kind, event_client_id in FakePodmanExecutionClient.events
                if event_client_id == client_id
            ] == ["start", "command", "command", "close"]


def test_podman_benchmark_can_warm_every_replica_outside_measurement(
    monkeypatch,
):
    FakePodmanExecutionClient.started_count = 0
    FakePodmanExecutionClient.closed_count = 0
    FakePodmanExecutionClient.active_count = 0
    FakePodmanExecutionClient.max_active_count = 0
    FakePodmanExecutionClient.events = []
    monkeypatch.setattr(
        podman_benchmark,
        "PodmanExecutionClient",
        FakePodmanExecutionClient,
    )

    result = asyncio.run(
        run_podman_benchmark(
            PodmanBenchmarkConfig(
                gateway_url="http://gateway:8080",
                replicas=2,
                concurrency=(1,),
                total_sessions=2,
                commands_per_session=2,
                warmup_all_replicas=True,
                warmup_concurrency=2,
                warmup_max_sessions=8,
            )
        )
    )

    assert result["warmup"]["excluded_from_measurement"] is True
    assert result["warmup"]["all_replicas_warmed"] is True
    assert result["warmup"]["observed_replicas"] == 2
    assert result["warmup"]["attempted_sessions"] == 2
    assert result["levels"][0]["requested_sessions"] == 2
    assert result["levels"][0]["completed_commands"] == 4
    assert all(
        "-warmup-" in client_id
        for kind, client_id in FakePodmanExecutionClient.events
        if kind in {"start", "close"}
    ) is False
    assert "untimed image warmup=2/2" in format_podman_benchmark(result)



def _scaling_result(replicas, startup_rate, command_rate):
    session_distribution = {
        f"podman-{index}": 16 // replicas
        for index in range(replicas)
    }
    command_distribution = {
        instance: count * 3
        for instance, count in session_distribution.items()
    }
    traffic = {
        "expected_replicas": replicas,
        "observed_replicas": replicas,
        "all_replicas_received_traffic": True,
    }
    return {
        "benchmark": "jtc-podman-affinity-throughput",
        "config": {
            "replicas": replicas,
            "concurrency": [8],
            "sessions_per_worker": 2,
            "waves": 1,
            "commands_per_session": 3,
            "image": "ubuntu:test",
            "command_timeout": 10,
            "workdir": "/workspace",
        },
        "levels": [
            {
                "concurrency": 8,
                "requested_sessions": 16,
                "successful_sessions": 16,
                "failed_sessions": 0,
                "instance_distribution": session_distribution,
                "command_instance_distribution": command_distribution,
                "traffic": {
                    "sessions": dict(traffic),
                    "commands": dict(traffic),
                },
                "latency": {
                    "handshake": {"p95_ms": 1000 / startup_rate},
                    "command": {"p95_ms": 1000 / command_rate},
                },
                "phases": {
                    "startup": {
                        "sessions_per_second": startup_rate,
                        "successful_sessions": 16,
                    },
                    "commands": {
                        "commands_per_second": command_rate,
                        "completed_commands": 48,
                    },
                },
            }
        ],
    }


def test_scaling_comparison_reports_speedup_and_full_replica_coverage():
    comparison = compare_podman_benchmark_results(
        [
            _scaling_result(2, 3.8, 38),
            _scaling_result(1, 2, 20),
        ]
    )

    assert comparison["replica_counts"] == [1, 2]
    two_replica = next(
        row for row in comparison["rows"] if row["replicas"] == 2
    )
    assert two_replica["startup_speedup_vs_baseline"] == 1.9
    assert two_replica["startup_scaling_efficiency"] == 0.95
    assert two_replica["command_speedup_vs_baseline"] == 1.9
    assert two_replica["startup_success_rate"] == 1.0
    assert two_replica["command_completion_rate"] == 1.0
    assert two_replica["lifecycle_success_rate"] == 1.0
    assert two_replica["session_replica_coverage"] == "2/2"
    assert two_replica["command_replica_coverage"] == "2/2"
    rendered = format_podman_scaling_comparison(comparison)
    assert "Podman horizontal scaling" in rendered
    assert "traffic(session|command)" in rendered


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"replicas": 0}, "replicas"),
        ({"concurrency": ()}, "concurrency"),
        ({"concurrency": (2, 1)}, "increasing"),
        ({"sessions_per_worker": 0}, "sessions_per_worker"),
        ({"waves": 0}, "waves"),
        (
            {"waves": 2, "total_sessions": 32},
            "cannot be combined",
        ),
        ({"commands_per_session": 1}, "write/read affinity"),
    ],
)
def test_podman_benchmark_rejects_invalid_workloads(override, message):
    values = {
        "gateway_url": "http://gateway:8080",
        "replicas": 1,
        **override,
    }
    with pytest.raises(ValueError, match=message):
        PodmanBenchmarkConfig(**values).validate()
