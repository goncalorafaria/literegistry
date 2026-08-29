import asyncio

import pytest

from tools import podman_benchmark
from tools.podman_benchmark import (
    PodmanCreationBenchmarkConfig,
    format_podman_creation_benchmark,
    run_podman_creation_benchmark,
)


class FakeCreationClient:
    created = 0
    closed = 0

    def __init__(self, gateway_url, **kwargs):
        self.gateway_url = gateway_url
        self.kwargs = kwargs
        self.started = False
        self.container_id = None

    async def start(self):
        type(self).created += 1
        self.started = True
        self.container_id = f"container-{type(self).created}"
        return {
            "container_id": self.container_id,
            "affinity_id": self.container_id,
            "instance_id": f"podman-{type(self).created % 2}",
        }

    async def close(self):
        type(self).closed += 1
        self.started = False
        return {"container_id": self.container_id, "removed": True}


def test_creation_benchmark_excludes_cleanup_and_sweeps_concurrency(monkeypatch):
    FakeCreationClient.created = 0
    FakeCreationClient.closed = 0
    monkeypatch.setattr(
        podman_benchmark,
        "PodmanExecutionClient",
        FakeCreationClient,
    )

    result = asyncio.run(
        run_podman_creation_benchmark(
            PodmanCreationBenchmarkConfig(
                gateway_url="http://gateway:8080",
                replicas=2,
                concurrency=(1, 4),
                cleanup_concurrency=2,
            )
        )
    )

    assert result["measurement"].endswith("cleanup is excluded")
    assert [level["requested_creations"] for level in result["levels"]] == [1, 4]
    assert [level["successful_creations"] for level in result["levels"]] == [1, 4]
    assert all(level["failed_creations"] == 0 for level in result["levels"])
    assert all(level["containers_per_second"] > 0 for level in result["levels"])
    assert all(level["cleanup"]["failed"] == 0 for level in result["levels"])
    assert sum(result["levels"][1]["instance_distribution"].values()) == 4
    assert FakeCreationClient.created == 5
    assert FakeCreationClient.closed == 5
    assert "create-p95-ms" in format_podman_creation_benchmark(result)


def test_creation_benchmark_can_leave_only_final_wave_for_stack_teardown(
    monkeypatch,
):
    FakeCreationClient.created = 0
    FakeCreationClient.closed = 0
    monkeypatch.setattr(
        podman_benchmark,
        "PodmanExecutionClient",
        FakeCreationClient,
    )

    result = asyncio.run(
        run_podman_creation_benchmark(
            PodmanCreationBenchmarkConfig(
                gateway_url="http://gateway:8080",
                replicas=2,
                concurrency=(2, 4),
                skip_final_cleanup=True,
            )
        )
    )

    assert result["levels"][0]["cleanup"]["skipped"] is False
    assert result["levels"][1]["cleanup"]["skipped"] is True
    assert FakeCreationClient.created == 6
    assert FakeCreationClient.closed == 2


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"replicas": 0}, "replicas"),
        ({"concurrency": ()}, "concurrency"),
        ({"concurrency": (4, 2)}, "increasing"),
        ({"cleanup_concurrency": 0}, "cleanup_concurrency"),
    ],
)
def test_creation_benchmark_rejects_invalid_config(override, message):
    values = {
        "gateway_url": "http://gateway:8080",
        "replicas": 1,
        **override,
    }
    with pytest.raises(ValueError, match=message):
        PodmanCreationBenchmarkConfig(**values).validate()
