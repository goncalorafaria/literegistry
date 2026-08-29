from __future__ import annotations

import subprocess

import pytest

from literegistry_podman_beaker import PodmanStackConfig, PodmanStackLauncher
from literegistry_podman_beaker import cli


def _tasks(spec):
    return {task["name"]: task for task in spec["tasks"]}


def test_single_cluster_stack_is_self_contained() -> None:
    config = PodmanStackConfig(
        registry="redis://jupiter.example:59936",
        podman_replicas=4,
        docker_mirror_replicas=2,
        service_clusters=("ai2/jupiter",),
        gateway_cluster="ai2/phobos",
    )
    _, spec = PodmanStackLauncher(config).build_spec(experiment_name="test-stack")
    tasks = _tasks(spec)

    assert set(tasks) == {"docker-mirror", "gateway", "podman"}
    assert tasks["docker-mirror"]["replicas"] == 2
    assert tasks["podman"]["replicas"] == 4
    assert tasks["podman"]["propagateFailure"] is False
    assert tasks["docker-mirror"]["propagateFailure"] is False
    assert tasks["gateway"]["propagateFailure"] is False
    assert all(task["propagatePreemption"] is False for task in tasks.values())
    assert all(task["context"]["autoResume"] is True for task in tasks.values())
    assert tasks["gateway"]["constraints"] == {"cluster": ["ai2/phobos"]}
    assert all(task["hostNetworking"] for task in tasks.values())
    assert all("resources" not in task for task in tasks.values())
    assert all("datadev" not in " ".join(task["command"]) for task in tasks.values())
    assert "literegistry.gateway" in tasks["gateway"]["command"][2]
    assert "literegistry.services.docker_mirror_server" in tasks["docker-mirror"]["command"][2]
    assert "exec literegistry podman " in tasks["podman"]["command"][2]
    assert "podman_affinity_redis_server" not in tasks["podman"]["command"][2]
    assert "--host=0.0.0.0" in tasks["podman"]["command"][2]
    assert "--allow_non_loopback=True" in tasks["podman"]["command"][2]
    assert '--registry_mirror=\\"$PODMAN_REGISTRY_MIRROR\\"' in tasks["podman"]["command"][2]
    assert "GATEWAY_URL=" in tasks["gateway"]["command"][2]
    assert "--docker_mirror_soft_affinity=True" in tasks["gateway"]["command"][2]
    assert "--docker_mirror_affinity_ttl_seconds=604800" in tasks["gateway"]["command"][2]
    assert all("--command-json=" in task["command"][2] for task in tasks.values())
    assert all("export PATH=\"${VIRTUAL_ENV}/bin:${PATH}\"" in task["command"][2] for task in tasks.values())
    assert all("--lock_dir=/tmp/literegistry-port-locks" in task["command"][2] for task in tasks.values())
    assert all(" -- bash -lc " not in task["command"][2] for task in tasks.values())
    assert all(
        not any(item["name"] == "PYTHONPATH" for item in task.get("envVars", []))
        for task in tasks.values()
    )

    for task in tasks.values():
        subprocess.run(["bash", "-n", "-c", task["command"][2]], check=True)


def test_replicas_are_spread_into_cluster_specific_tasks() -> None:
    clusters = ("ai2/neptune", "ai2/saturn", "ai2/jupiter", "ai2/ceres")
    config = PodmanStackConfig(
        registry="redis://registry.example:6379",
        podman_replicas=10,
        docker_mirror_replicas=6,
        service_clusters=clusters,
    )
    _, spec = PodmanStackLauncher(config).build_spec(experiment_name="spread-stack")
    podman = [task for task in spec["tasks"] if task["name"].startswith("podman-")]
    mirrors = [task for task in spec["tasks"] if task["name"].startswith("docker-mirror-")]

    assert [task.get("replicas", 1) for task in podman] == [3, 3, 2, 2]
    assert [task.get("replicas", 1) for task in mirrors] == [2, 2, 1, 1]
    assert [task["constraints"]["cluster"][0] for task in podman] == list(clusters)
    assert sum(task.get("replicas", 1) for task in podman) == 10
    assert sum(task.get("replicas", 1) for task in mirrors) == 6


def test_docker_hub_token_is_a_beaker_secret_reference() -> None:
    config = PodmanStackConfig(
        registry="redis://registry.example:6379",
        docker_hub_username="allenai",
        docker_hub_token_secret="DOCKER_HUB_OAT",
    )
    _, spec = PodmanStackLauncher(config).build_spec(experiment_name="auth-stack")
    mirror = _tasks(spec)["docker-mirror"]

    assert {"name": "DOCKER_HUB_USERNAME", "value": "allenai"} in mirror["envVars"]
    assert {"name": "DOCKER_HUB_TOKEN", "secret": "DOCKER_HUB_OAT"} in mirror["envVars"]
    assert "DOCKER_HUB_OAT" not in mirror["command"][2]




def test_both_docker_hub_credentials_can_be_beaker_secrets() -> None:
    config = PodmanStackConfig(
        registry="redis://registry.example:6379",
        docker_hub_username_secret="docker-username",
        docker_hub_token_secret="docker-token",
    )
    _, spec = PodmanStackLauncher(config).build_spec(
        experiment_name="secret-auth-stack"
    )
    mirror = _tasks(spec)["docker-mirror"]

    assert {
        "name": "DOCKER_HUB_USERNAME",
        "secret": "docker-username",
    } in mirror["envVars"]
    assert {
        "name": "DOCKER_HUB_TOKEN",
        "secret": "docker-token",
    } in mirror["envVars"]
    assert "docker-username" not in mirror["command"][2]
    assert "docker-token" not in mirror["command"][2]


def test_bundled_warm_list_is_not_overridden_by_default() -> None:
    config = PodmanStackConfig(registry="redis://registry.example:6379")
    _, spec = PodmanStackLauncher(config).build_spec(experiment_name="warm-stack")
    env = _tasks(spec)["docker-mirror"].get("envVars", [])
    assert not any(item["name"] == "DOCKER_MIRROR_WARM_IMAGES_FILE" for item in env)


def test_validation_rejects_partial_or_raw_credentials() -> None:
    with pytest.raises(ValueError, match="supplied together"):
        PodmanStackConfig(
            registry="redis://registry.example:6379",
            docker_hub_username="allenai",
        ).validate()
    with pytest.raises(TypeError):
        PodmanStackConfig(
            registry="redis://registry.example:6379",
            docker_hub_token="not-accepted",
        )


def test_fire_cli_previews_stack(capsys) -> None:
    cli.main(
        [
            "preview",
            "--registry=redis://registry.example:6379",
            "--podman-replicas=2",
            "--docker-mirror-replicas=1",
            "--service-cluster=ai2/jupiter,ai2/ceres",
        ]
    )
    output = capsys.readouterr().out

    assert "\"podman_replicas\": 2" in output
    assert "ai2/jupiter" in output
    assert "ai2/ceres" in output


def test_managed_redis_stack_publishes_registry_url_to_all_services() -> None:
    config = PodmanStackConfig(
        podman_replicas=1,
        docker_mirror_replicas=1,
        service_clusters=("ai2/jupiter",),
        gateway_cluster="ai2/phobos",
    )
    _, spec = PodmanStackLauncher(config).build_spec(experiment_name="managed-stack")
    tasks = _tasks(spec)

    assert set(tasks) == {"redis", "docker-mirror", "gateway", "podman"}
    assert tasks["redis"]["constraints"] == {"cluster": ["ai2/phobos"]}
    assert tasks["redis"]["propagateFailure"] is True
    assert tasks["redis"]["propagatePreemption"] is True
    assert tasks["redis"]["context"]["autoResume"] is False
    for service in ("docker-mirror", "gateway", "podman"):
        assert tasks[service]["propagateFailure"] is False
        assert tasks[service]["propagatePreemption"] is False
        assert tasks[service]["context"]["autoResume"] is True
    assert "exec literegistry redis " in tasks["redis"]["command"][2]
    assert "REDIS_URL=" in tasks["redis"]["command"][2]
    for name in ("docker-mirror", "gateway", "podman"):
        command = tasks[name]["command"][2]
        assert "literegistry_podman_registry_managed-stack.url" in command
        assert "literegistry.coop.redis wait" in command


def test_external_redis_does_not_create_redis_task() -> None:
    config = PodmanStackConfig(registry="redis://registry.example:6379")
    _, spec = PodmanStackLauncher(config).build_spec(experiment_name="external-stack")

    assert "redis" not in _tasks(spec)
    assert all(
        "REGISTRY=redis://registry.example:6379" in task["command"][2]
        for task in spec["tasks"]
    )


def test_gateway_can_disable_experimental_mirror_soft_affinity() -> None:
    config = PodmanStackConfig(
        registry="redis://registry.example:6379",
        docker_mirror_soft_affinity=False,
    )
    _, spec = PodmanStackLauncher(config).build_spec(
        experiment_name="soft-affinity-stack"
    )

    command = _tasks(spec)["gateway"]["command"][2]
    assert "--docker_mirror_soft_affinity=False" in command
