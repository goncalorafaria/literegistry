from __future__ import annotations

import subprocess

import pytest

from literegistry_base_deployment import BaseDeploymentConfig, BaseDeploymentLauncher
from literegistry_base_deployment import cli


def _tasks(spec):
    return {task["name"]: task for task in spec["tasks"]}


def test_complete_external_registry_stack_has_every_requested_service() -> None:
    config = BaseDeploymentConfig(
        registry="redis://jupiter.example:59936",
        python_replicas=2,
        terminal_replicas=3,
        web_search_replicas=2,
        local_search_replicas=2,
        local_search_corpus_jsonl="/weka/data/corpus.jsonl",
        local_search_index_dir="/weka/data/index",
        generation_model="allenai/generator",
        generation_replicas=2,
        classification_model="allenai/classifier",
        classification_replicas=1,
        service_clusters=("ai2/jupiter",),
        gateway_cluster="ai2/phobos",
        model_cluster="ai2/saturn",
    )
    _, spec = BaseDeploymentLauncher(config).build_spec(experiment_name="base-stack")
    tasks = _tasks(spec)

    assert set(tasks) == {
        "gateway",
        "python",
        "terminal",
        "search",
        "localsearch",
        "vllm-generate-generator",
        "vllm-classify-classifier",
    }
    assert tasks["python"]["replicas"] == 2
    assert tasks["terminal"]["replicas"] == 3
    assert tasks["gateway"]["constraints"] == {"cluster": ["ai2/phobos"]}
    assert "--docker_mirror_soft_affinity=True" in tasks["gateway"]["command"][2]
    assert tasks["vllm-generate-generator"]["constraints"] == {
        "cluster": ["ai2/saturn"]
    }
    assert tasks["vllm-generate-generator"]["resources"] == {"gpuCount": 1}
    for name, task in tasks.items():
        assert "cpuCount" not in task["resources"]
        if not name.startswith("vllm-"):
            assert task["resources"] == {"gpuCount": 0}
    assert all(task["hostNetworking"] for task in tasks.values())
    assert all(task["propagateFailure"] is False for task in tasks.values())
    assert all(task["propagatePreemption"] is False for task in tasks.values())
    assert all(task["context"]["autoResume"] is True for task in tasks.values())
    non_local_tasks = [task for name, task in tasks.items() if name != "localsearch"]
    assert all("datadev" not in " ".join(task["command"]) for task in non_local_tasks)
    assert all("literegistry.coop.ports run" in task["command"][2] for task in non_local_tasks)
    assert all(
        "--lock_dir=/tmp/literegistry-port-locks" in task["command"][2] for task in non_local_tasks
    )

    generation = tasks["vllm-generate-generator"]["command"][2]
    classification = tasks["vllm-classify-classifier"]["command"][2]
    assert "--task=generate" in generation
    assert "--language-model-only" in generation
    assert "--task=classify" in classification
    assert "--language-model-only" not in classification
    local_search = tasks["localsearch"]
    assert local_search["image"]["beaker"] == "goncalof/jtc-local-search-lucene-bm25"
    assert "literegistry.coop.ports run" in local_search["command"][2]
    assert "literegistry.coop.redis wait" in local_search["command"][2]
    assert "/app/search/build_lucene_index.sh" in local_search["command"][2]
    assert "literegistry.services.bm25_server:create_app" in local_search["command"][2]
    assert "literegistry_base_deployment.local_search" not in local_search["command"][2]

    for task in tasks.values():
        subprocess.run(["bash", "-n", "-c", task["command"][2]], check=True)


def test_cpu_replicas_are_spread_across_clusters() -> None:
    config = BaseDeploymentConfig(
        registry="redis://registry.example:6379",
        python_replicas=10,
        terminal_replicas=0,
        web_search_replicas=0,
        service_clusters=("ai2/neptune", "ai2/saturn", "ai2/jupiter", "ai2/ceres"),
    )
    _, spec = BaseDeploymentLauncher(config).build_spec(experiment_name="spread-stack")
    python_tasks = [task for task in spec["tasks"] if task["name"].startswith("python-")]

    assert [task.get("replicas", 1) for task in python_tasks] == [3, 3, 2, 2]
    assert [task["constraints"]["cluster"][0] for task in python_tasks] == [
        "ai2/neptune",
        "ai2/saturn",
        "ai2/jupiter",
        "ai2/ceres",
    ]
    commands = [task["command"][2] for task in python_tasks]
    assert "LR_GLOBAL_RANK=$((0 +" in commands[0]
    assert "LR_GLOBAL_RANK=$((3 +" in commands[1]
    assert "LR_GLOBAL_RANK=$((6 +" in commands[2]
    assert "LR_GLOBAL_RANK=$((8 +" in commands[3]


def test_managed_redis_publishes_registry_url() -> None:
    config = BaseDeploymentConfig(
        python_replicas=1,
        terminal_replicas=0,
        web_search_replicas=0,
        gateway_cluster="ai2/phobos",
    )
    _, spec = BaseDeploymentLauncher(config).build_spec(experiment_name="managed-stack")
    tasks = _tasks(spec)

    assert set(tasks) == {"redis", "gateway", "python"}
    assert all(task["resources"] == {"gpuCount": 0} for task in tasks.values())
    assert tasks["redis"]["constraints"] == {"cluster": ["ai2/phobos"]}
    assert tasks["redis"]["propagateFailure"] is True
    assert tasks["redis"]["propagatePreemption"] is True
    assert tasks["redis"]["context"]["autoResume"] is False
    for service in ("gateway", "python"):
        assert tasks[service]["propagateFailure"] is False
        assert tasks[service]["propagatePreemption"] is False
        assert tasks[service]["context"]["autoResume"] is True
    assert "REDIS_URL=" in tasks["redis"]["command"][2]
    assert "exec literegistry redis" in tasks["redis"]["command"][2]
    for service in ("gateway", "python"):
        assert "literegistry_base_registry_managed-stack.url" in tasks[service]["command"][2]
        assert "literegistry.coop.redis wait" in tasks[service]["command"][2]


def test_beaker_commands_restore_bundled_virtualenv_path() -> None:
    _, spec = BaseDeploymentLauncher(
        BaseDeploymentConfig(
            registry="redis://registry.example:6379",
            python_replicas=1,
            terminal_replicas=1,
            web_search_replicas=1,
        )
    ).build_spec(experiment_name="venv-stack")

    for task in spec["tasks"]:
        command = task["command"][2]
        assert "/opt/literegistry-services-venv" in command
        assert "/opt/literegistry-terminal-venv" in command
        assert "/opt/literegistry-redis-venv" in command
        subprocess.run(["bash", "-n", "-c", command], check=True)


def test_search_credentials_are_only_beaker_secret_references() -> None:
    config = BaseDeploymentConfig(
        registry="redis://registry.example:6379",
        python_replicas=0,
        terminal_replicas=0,
        web_search_replicas=1,
        serper_api_key_secret="MY_SERPER_SECRET",
        jina_api_key_secret="MY_JINA_SECRET",
    )
    _, spec = BaseDeploymentLauncher(config).build_spec(experiment_name="search-stack")
    search = _tasks(spec)["search"]

    assert {"name": "SERPER_API_KEY", "secret": "MY_SERPER_SECRET"} in search["envVars"]
    assert {"name": "JINA_API_KEY", "secret": "MY_JINA_SECRET"} in search["envVars"]
    assert "MY_SERPER_SECRET" not in search["command"][2]


def test_validation_rejects_incomplete_optional_pools() -> None:
    with pytest.raises(ValueError, match="generation_model"):
        BaseDeploymentConfig(generation_replicas=1).validate()
    with pytest.raises(ValueError, match="local_search_corpus_jsonl"):
        BaseDeploymentConfig(local_search_replicas=1).validate()
    with pytest.raises(ValueError, match="Serper and Jina"):
        BaseDeploymentConfig(serper_api_key_secret=None).validate()


def test_fire_cli_previews_stack(capsys) -> None:
    cli.main(
        [
            "preview",
            "--registry=redis://registry.example:6379",
            "--python-replicas=2",
            "--terminal-replicas=0",
            "--web-search-replicas=0",
            "--service-cluster=ai2/jupiter,ai2/ceres",
        ]
    )
    output = capsys.readouterr().out

    assert '"python_replicas": 2' in output
    assert "ai2/jupiter" in output
    assert "ai2/ceres" in output


def test_gateway_can_disable_experimental_mirror_soft_affinity() -> None:
    config = BaseDeploymentConfig(
        registry="redis://registry.example:6379",
        docker_mirror_soft_affinity=False,
    )
    _, spec = BaseDeploymentLauncher(config).build_spec(
        experiment_name="soft-affinity-base"
    )

    command = _tasks(spec)["gateway"]["command"][2]
    assert "--docker_mirror_soft_affinity=False" in command
