from __future__ import annotations

import subprocess

import pytest

from literegistry_podman_beaker import PodmanStackConfig, PodmanStackLauncher
from literegistry_podman_beaker import cli
from literegistry_podman_beaker import launcher as launcher_module


def _tasks(spec):
    return {task["name"]: task for task in spec["tasks"]}


def test_prepare_shared_directory_is_sticky_and_writable(tmp_path) -> None:
    root = tmp_path / "managed" / "head"
    launcher_module._prepare_shared_directory(str(root))

    assert root.is_dir()
    assert root.stat().st_mode & 0o7777 == 0o1777


def test_single_cluster_stack_is_self_contained() -> None:
    config = PodmanStackConfig(
        registry="redis://jupiter.example:59936",
        podman_replicas=4,
        docker_mirror_replicas=2,
        service_clusters=("ai2/jupiter",),
        gateway_cluster="ai2/phobos",
        podman_max_sessions=64,
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
    assert all(task["resources"] == {"gpuCount": 0} for task in tasks.values())
    assert all("datadev" not in " ".join(task["command"]) for task in tasks.values())
    assert "literegistry.gateway" in tasks["gateway"]["command"][2]
    assert "literegistry.services.docker_mirror_server" in tasks["docker-mirror"]["command"][2]
    assert "exec literegistry podman " in tasks["podman"]["command"][2]
    assert "podman_affinity_redis_server" not in tasks["podman"]["command"][2]
    assert "--host=0.0.0.0" in tasks["podman"]["command"][2]
    assert "--allow_non_loopback=True" in tasks["podman"]["command"][2]
    assert "--session_memory=4g" in tasks["podman"]["command"][2]
    assert "--max_sessions=64" in tasks["podman"]["command"][2]
    assert "--session_pids_limit=2048" in tasks["podman"]["command"][2]
    assert "--session_idle_timeout=7200" in tasks["podman"]["command"][2]
    assert "--janitor_interval=300" in tasks["podman"]["command"][2]
    assert "--resource_watchdog_interval=5" in tasks["podman"]["command"][2]
    assert "--image_prune_until=24h" in tasks["podman"]["command"][2]
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
    assert "\"podman_session_memory\": \"4g\"" in output
    assert "\"podman_session_pids_limit\": 2048" in output
    assert "\"podman_session_idle_timeout\": 7200.0" in output
    assert "\"podman_janitor_interval\": 300.0" in output
    assert "\"podman_resource_watchdog_interval\": 5.0" in output
    assert "\"podman_image_prune_until\": \"24h\"" in output
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
    assert tasks["redis"]["propagateFailure"] is False
    assert tasks["redis"]["propagatePreemption"] is False
    assert tasks["redis"]["context"]["autoResume"] is True
    for service in ("docker-mirror", "gateway", "podman"):
        assert tasks[service]["propagateFailure"] is False
        assert tasks[service]["propagatePreemption"] is False
        assert tasks[service]["context"]["autoResume"] is True
    redis_command = tasks["redis"]["command"][2]
    assert "exec literegistry redis " in redis_command
    assert "--head_registry=file:///weka/gfaria/literegistry/.coop/managed-stack" in redis_command
    assert "--data_dir=/weka/gfaria/literegistry/.coop/managed-stack/redis-data" in redis_command
    assert "--persistence=True" in redis_command
    assert "--advertise_host=" in redis_command
    assert "REDIS_ADVERTISE_HOST" in redis_command
    assert "literegistry.coop.endpoints run" not in redis_command
    for service in ("docker-mirror", "gateway", "podman"):
        command = tasks[service]["command"][2]
        assert "/.coop/managed-stack" in command
        assert "REGISTRY=head+file:///weka/gfaria/literegistry/.coop/managed-stack" in command
        assert "--name=redis" not in command
        assert "literegistry_podman_registry_managed-stack.url" not in command
    gateway_command = tasks["gateway"]["command"][2]
    assert "literegistry.coop.endpoints run" in gateway_command
    assert "--name=gateway" in gateway_command
    podman_command = tasks["podman"]["command"][2]
    assert "--name=gateway" in podman_command
    assert "PODMAN_REGISTRY_MIRROR=" in podman_command
    assert "literegistry.coop.endpoints wait" in podman_command


def test_external_redis_does_not_create_redis_task() -> None:
    config = PodmanStackConfig(registry="redis://registry.example:6379")
    _, spec = PodmanStackLauncher(config).build_spec(experiment_name="external-stack")

    assert "redis" not in _tasks(spec)
    assert all(
        "REGISTRY=redis://registry.example:6379" in task["command"][2]
        for task in spec["tasks"]
    )


def test_registry_head_uri_is_passed_to_every_service_without_redis_task() -> None:
    config = PodmanStackConfig(
        registry="head:///weka/shared/head-registry",
        podman_replicas=1,
        docker_mirror_replicas=0,
    )
    _, spec = PodmanStackLauncher(config).build_spec(experiment_name="external-head")
    tasks = _tasks(spec)

    assert "redis" not in tasks
    assert set(tasks) == {"gateway", "podman"}
    assert all(
        "REGISTRY=head+file:///weka/shared/head-registry" in task["command"][2]
        for task in tasks.values()
    )
    assert all(
        "literegistry.coop.redis wait" not in task["command"][2]
        for task in tasks.values()
    )


def test_head_registry_option_normalizes_to_registry_uri() -> None:
    config = PodmanStackConfig(head_registry="/weka/shared/head-registry").validate()
    assert config.resolved_registry() == "head+file:///weka/shared/head-registry"


def test_registry_and_head_registry_are_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="only one"):
        PodmanStackConfig(
            registry="redis://registry.example:6379",
            head_registry="/weka/shared/head-registry",
        ).validate()
    with pytest.raises(ValueError, match="absolute shared path"):
        PodmanStackConfig(head_registry="relative/head").validate()
    with pytest.raises(ValueError):
        PodmanStackConfig(registry="head://relative/head").validate()


@pytest.mark.parametrize(
    ("head_registry", "resolved"),
    [
        (
            "sqlite:///weka/shared/head.sqlite3",
            "head+sqlite:///weka/shared/head.sqlite3",
        ),
        (
            "redis://head.example:6379",
            "head+redis://head.example:6379",
        ),
        (
            "file:///weka/shared/head",
            "head+file:///weka/shared/head",
        ),
    ],
)
def test_head_registry_backends_are_rendered(
    head_registry: str,
    resolved: str,
) -> None:
    config = PodmanStackConfig(head_registry=head_registry).validate()
    assert config.resolved_registry() == resolved


def test_explicit_sqlite_head_launches_managed_redis() -> None:
    config = PodmanStackConfig(
        head_registry="sqlite:///weka/shared/podman-head.sqlite3",
        podman_replicas=1,
        docker_mirror_replicas=0,
    )
    _, spec = PodmanStackLauncher(config).build_spec(
        experiment_name="sqlite-managed-stack"
    )
    tasks = _tasks(spec)

    assert set(tasks) == {"redis", "gateway", "podman"}
    redis_command = tasks["redis"]["command"][2]
    assert "--head_registry=sqlite:///weka/shared/podman-head.sqlite3" in redis_command
    assert (
        "--data_dir=/weka/gfaria/literegistry/.coop/"
        "sqlite-managed-stack/redis-data"
    ) in redis_command
    for service in ("gateway", "podman"):
        assert (
            "REGISTRY=head+sqlite:///weka/shared/podman-head.sqlite3"
            in tasks[service]["command"][2]
        )


def test_existing_cluster_expansion_can_launch_without_new_mirrors() -> None:
    config = PodmanStackConfig(
        registry="redis://registry.example:6379",
        podman_replicas=4,
        docker_mirror_replicas=0,
        service_clusters=("ai2/jupiter", "ai2/ceres"),
        podman_instance_prefix="podman-v1046",
    )
    _, spec = PodmanStackLauncher(config).build_spec(
        experiment_name="existing-cluster-expansion"
    )
    tasks = _tasks(spec)

    assert set(tasks) == {"gateway", "podman-ai2-jupiter", "podman-ai2-ceres"}
    assert all(not name.startswith("docker-mirror") for name in tasks)
    assert all(
        "--name=gateway" in task["command"][2]
        for name, task in tasks.items()
        if name.startswith("podman-")
    )
    assert all(
        'INSTANCE_ID=podman-v1046-\\"${GLOBAL_RANK}\\"' in task["command"][2]
        for name, task in tasks.items()
        if name.startswith("podman-")
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


def test_podman_hardening_can_be_disabled_or_overridden() -> None:
    config = PodmanStackConfig(
        registry="redis://registry.example:6379",
        podman_session_memory=None,
        podman_session_pids_limit=None,
        podman_session_idle_timeout=None,
        podman_janitor_interval=17,
        podman_resource_watchdog_interval=None,
        podman_image_prune_until=None,
    )
    _, spec = PodmanStackLauncher(config).build_spec(experiment_name="custom-hardening")
    command = _tasks(spec)["podman"]["command"][2]

    for option in (
            "--session_memory=",
            "--max_sessions=",
        "--session_pids_limit=",
        "--session_idle_timeout=",
        "--resource_watchdog_interval=",
        "--image_prune_until=",
    ):
        assert option not in command
    assert "--janitor_interval=17" in command


@pytest.mark.parametrize(
    "kwargs",
    [
        {"podman_session_memory": ""},
        {"podman_max_sessions": 0},
        {"podman_session_pids_limit": 0},
        {"podman_session_idle_timeout": 0},
        {"podman_janitor_interval": 0},
        {"podman_resource_watchdog_interval": 0},
        {"podman_image_prune_until": ""},
    ],
)
def test_podman_hardening_validation(kwargs) -> None:
    with pytest.raises(ValueError):
        PodmanStackConfig(**kwargs).validate()
