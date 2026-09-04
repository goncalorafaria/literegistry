from __future__ import annotations

from pathlib import Path
import re
import subprocess

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DOCKER_ROOT = PACKAGE_ROOT / "docker"


@pytest.mark.parametrize(
    ("name", "expected_base", "copies_beaker_package"),
    [
        ("Dockerfile.redis", "FROM redis:7-bookworm", True),
        ("Dockerfile.gateway", "FROM python:3.12-slim-bookworm", True),
        (
            "Dockerfile.podman",
            "FROM quay.io/podman/stable:${PODMAN_VERSION}",
            True,
        ),
        ("Dockerfile.mirror", "FROM registry:3 AS distribution", True),
        ("Dockerfile.warmup", "FROM python:3.12-slim-bookworm", True),
    ],
)
def test_images_use_upstream_bases_and_local_package_context(
    name: str,
    expected_base: str,
    copies_beaker_package: bool,
) -> None:
    contents = (DOCKER_ROOT / name).read_text(encoding="utf-8")

    assert expected_base in contents
    assert (
        "COPY . /opt/literegistry-podman-beaker" in contents
    ) is copies_beaker_package
    assert "goncalof/" not in contents
    assert "basic_images" not in contents
    assert "ARG BASE_IMAGE" not in contents
    assert "/weka/" not in contents


def test_redis_image_uses_public_server_command_as_non_root() -> None:
    contents = (DOCKER_ROOT / "Dockerfile.redis").read_text(encoding="utf-8")

    assert "USER redis" in contents
    assert 'ENTRYPOINT ["literegistry", "redis"]' in contents
    assert 'CMD ["--runtime=local", "--foreground=True", "--port=6379"]' in contents
    assert "groupmod --gid 10001 redis" in contents
    assert "usermod --uid 10001 --gid 10001 redis" in contents
    assert "chown -R redis:redis /data" in contents


def test_gateway_image_calls_literegistry_directly() -> None:
    contents = (DOCKER_ROOT / "Dockerfile.gateway").read_text(encoding="utf-8")

    assert '"literegistry==${LITEREGISTRY_VERSION}"' in contents
    assert 'ENTRYPOINT ["python", "-m", "literegistry.gateway"]' in contents
    assert "pip install --no-cache-dir --no-deps /opt/literegistry-podman-beaker" in contents


def test_mirror_image_uses_canonical_service_path() -> None:
    contents = (DOCKER_ROOT / "Dockerfile.mirror").read_text(encoding="utf-8")

    assert 'ENTRYPOINT ["python", "-m", "literegistry.services.docker_mirror_server"]' in contents


def test_warmup_image_uses_public_podman_client_command() -> None:
    contents = (DOCKER_ROOT / "Dockerfile.warmup").read_text(encoding="utf-8")
    assert 'ENTRYPOINT ["literegistry-podman-warm-podman"]' in contents
    assert "USER warmer" in contents


def test_rootless_podman_image_restores_podman_user() -> None:
    contents = (DOCKER_ROOT / "Dockerfile.podman").read_text(encoding="utf-8")
    assert "USER podman" in contents
    assert "podman:100000:65536" in contents
    assert 'ENTRYPOINT ["/usr/local/bin/literegistry-podman-entrypoint"]' in contents
    entrypoint = (DOCKER_ROOT / "rootless-entrypoint.sh").read_text(encoding="utf-8")
    assert "exec literegistry podman" in entrypoint
    assert "podman_affinity_redis_server" not in entrypoint


def test_mirror_warm_list_is_vendored_and_unique() -> None:
    asset = PACKAGE_ROOT / "src" / "literegistry_podman_beaker" / "assets" / "allenai-tmax-15k-open-instruct-images.txt"
    images = [
        line.strip()
        for line in asset.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]

    assert len(images) == 14_490
    assert len(images) == len(set(images))
    assert "COPY --chown=mirror:mirror src/literegistry_podman_beaker/assets/" in (
        DOCKER_ROOT / "Dockerfile.mirror"
    ).read_text(encoding="utf-8")


def test_build_all_script_is_valid_and_documents_all_images() -> None:
    script = PACKAGE_ROOT / "scripts" / "build-images.sh"
    subprocess.run(["bash", "-n", str(script)], check=True)
    help_result = subprocess.run(
        [str(script), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "literegistry-redis" in script.read_text(encoding="utf-8")
    assert "literegistry-podman-gateway" in script.read_text(encoding="utf-8")
    assert "literegistry-podman-server" in script.read_text(encoding="utf-8")
    assert "literegistry-docker-mirror" in script.read_text(encoding="utf-8")
    assert "literegistry-podman-warmup" in script.read_text(encoding="utf-8")
    assert "PUSH_IMAGES" in help_result.stdout


def test_readme_covers_build_to_cleanup_workflow() -> None:
    readme = (PACKAGE_ROOT / "README.md").read_text(encoding="utf-8")

    required_steps = (
        "## End-to-end setup",
        "beaker account whoami",
        "./scripts/build-images.sh",
        "beaker image create",
        "literegistry-podman-beaker preview",
        "literegistry-podman-beaker launch",
        "/affinity/handshake",
        "literegistry-podman-beaker stop",
    )
    assert all(step in readme for step in required_steps)

    bash_blocks = re.findall(r"```bash\n(.*?)```", readme, flags=re.DOTALL)
    subprocess.run(
        ["bash", "-n"],
        input="\n".join(bash_blocks),
        text=True,
        check=True,
    )
