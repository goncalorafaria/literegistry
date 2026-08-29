from __future__ import annotations

from importlib import import_module
from pathlib import Path


def test_canonical_modules_are_importable() -> None:
    modules = (
        "literegistry.coop.artifacts",
        "literegistry.coop.ports",
        "literegistry.coop.redis",
        "literegistry.redis",
        "literegistry.runtime",
        "literegistry.services.executable_wrapper",
        "literegistry.services.vllm_wrapper",
        "literegistry.services.sglang_wrapper",
        "literegistry.services.code_server",
        "literegistry.services.terminal_server",
        "literegistry.services.search_server",
        "literegistry.services.bm25_server",
        "literegistry.services.podman",
        "literegistry.services.podman_server",
        "literegistry.services.docker_mirror_server",
        "literegistry.services.docker_mirror_warmup",
        "literegistry.services.affinity_mock_server",
        "literegistry.services.openai_proxy",
        "literegistry.gateway.affinity",
        "literegistry.gateway.mirror",
        "literegistry.gateway.basic",
        "literegistry.gateway.legacy",
    )

    for module in modules:
        import_module(module)


def test_runtime_package_has_no_duplicate_module_shims() -> None:
    package = Path(__file__).resolve().parents[1] / "literegistry"
    duplicate_names = (
        "executable_wrapper.py",
        "vllm_wrapper.py",
        "sglang_wrapper.py",
        "code_server.py",
        "terminal_server.py",
        "search_server.py",
        "podman_affinity_server.py",
        "podman_affinity_redis_server.py",
        "docker_mirror_server.py",
        "docker_mirror_warmup.py",
        "affinity_mock_server.py",
        "openai_proxy.py",
        "gateway_affinity.py",
        "gateway_mirror.py",
        "gateway_basic.py",
        "old_gateway.py",
        "test_benchmark_server.py",
    )

    assert all(not (package / name).exists() for name in duplicate_names)
    assert (package / "runtime.py").is_file()
    assert not (package / "services" / "runtime.py").exists()


def test_gateway_package_exports_application_api() -> None:
    gateway = import_module("literegistry.gateway")

    assert callable(gateway.create_app)
    assert callable(gateway.main)
    assert gateway.Gateway.__module__ == "literegistry.gateway"


def test_bandit_tuning_lives_outside_runtime_package() -> None:
    repository = Path(__file__).resolve().parents[1]

    assert (repository / "tools" / "bandit_tuning.py").is_file()
    assert not (repository / "literegistry" / "bandit_tuning.py").exists()


def test_redis_lives_at_package_root() -> None:
    repository = Path(__file__).resolve().parents[1]

    assert (repository / "literegistry" / "redis.py").is_file()
    assert not (repository / "literegistry" / "infrastructure").exists()
    assert not (repository / "literegistry" / "services" / "redis.py").exists()
