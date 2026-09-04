from __future__ import annotations

from pathlib import Path
import re
import subprocess


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DOCKER_ROOT = PACKAGE_ROOT / "docker"


def test_images_are_self_contained_and_install_this_package() -> None:
    expected = {
        "Dockerfile.redis": "FROM redis:7-bookworm",
        "Dockerfile.services": "FROM python:3.12-slim-bookworm",
        "Dockerfile.terminal": "FROM rust:1.83-bookworm AS xsv-builder",
        "Dockerfile.vllm": "FROM ${VLLM_BASE_IMAGE}",
    }
    assert not (
        PACKAGE_ROOT
        / "src"
        / "literegistry_base_deployment"
        / "local_search.py"
    ).exists()
    for name, base in expected.items():
        contents = (DOCKER_ROOT / name).read_text(encoding="utf-8")
        assert base in contents
        assert "COPY . /opt/literegistry-base-deployment" in contents
        assert "basic_images" not in contents
        assert "/weka/" not in contents


def test_every_image_installs_package_with_literegistry_dependency() -> None:
    for dockerfile in DOCKER_ROOT.glob("Dockerfile.*"):
        if dockerfile.name == "Dockerfile.local-search":
            continue
        contents = dockerfile.read_text(encoding="utf-8")
        assert "/opt/literegistry-base-deployment" in contents


def test_local_search_image_uses_literegistry_service_and_jtc_index_assets() -> None:
    contents = (DOCKER_ROOT / "Dockerfile.local-search").read_text(encoding="utf-8")
    assert "FROM eclipse-temurin:21-jdk-jammy AS java" in contents
    assert '"pyserini==2.3.0"' in contents
    assert "ARG LITEREGISTRY_VERSION=1.0.47" in contents
    assert "COPY search /app/search" in contents
    assert "COPY datadev /app/datadev" not in contents
    assert "literegistry.services.bm25_server" in contents
    assert "datadev.infra.bm25_server" not in contents
    assert "literegistry_base_deployment.local_search" not in contents


def test_redis_uses_shared_service_uid() -> None:
    contents = (DOCKER_ROOT / "Dockerfile.redis").read_text(encoding="utf-8")
    assert "groupmod --gid 10001 redis" in contents
    assert "usermod --uid 10001 --gid 10001 redis" in contents
    assert "chown -R redis:redis /data" in contents


def test_terminal_image_contains_every_allowlisted_external_tool() -> None:
    contents = (DOCKER_ROOT / "Dockerfile.terminal").read_text(encoding="utf-8")
    for command in ("ripgrep", "jq", "pandoc", "xsv", "coreutils", "gawk"):
        assert command in contents


def test_build_all_script_is_valid_and_lists_all_images() -> None:
    script = PACKAGE_ROOT / "scripts" / "build-images.sh"
    subprocess.run(["bash", "-n", str(script)], check=True)
    help_result = subprocess.run(
        ["bash", str(script), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    contents = script.read_text(encoding="utf-8")
    for name in (
        "literegistry-redis",
        "literegistry-base-services",
        "literegistry-base-terminal",
        "literegistry-base-vllm",
        "jtc-local-search-lucene-bm25",
    ):
        assert name in contents
    assert "PUSH_IMAGES" in help_result.stdout
    assert "JTC_BUILD_CONTEXT" in help_result.stdout


def test_readme_shell_examples_are_syntax_valid() -> None:
    readme = (PACKAGE_ROOT / "README.md").read_text(encoding="utf-8")
    bash_blocks = re.findall(r"```bash\n(.*?)```", readme, flags=re.DOTALL)
    assert bash_blocks
    subprocess.run(["bash", "-n"], input="\n".join(bash_blocks), text=True, check=True)
