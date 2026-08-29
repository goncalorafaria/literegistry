from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from literegistry.services.docker_mirror_server import (
    DockerMirrorConfig,
    DockerMirrorSupervisor,
    MirrorRegistration,
    build_distribution_config,
    dockerhub_repo_and_reference,
    write_distribution_config,
)


@pytest.mark.parametrize(
    ("image", "expected"),
    [
        ("alpine", ("library/alpine", "latest")),
        ("ubuntu:24.04", ("library/ubuntu", "24.04")),
        ("docker.io/allenai/foo:dev", ("allenai/foo", "dev")),
        ("registry-1.docker.io/a/b@sha256:abc", ("a/b", "sha256:abc")),
    ],
)
def test_dockerhub_repo_and_reference(image, expected) -> None:
    assert dockerhub_repo_and_reference(image) == expected


def test_health_probe_can_switch_from_tag_to_immutable_digest() -> None:
    config = DockerMirrorConfig(
        registry_url="redis://registry:6379",
        advertise_host="mirror.example",
        health_image="docker.io/library/alpine:3.20",
    )
    assert config.health_url.endswith("/v2/library/alpine/manifests/3.20")
    assert config.health_url_for_digest("sha256:abc").endswith(
        "/v2/library/alpine/manifests/sha256:abc"
    )


def test_distribution_config_and_file_permissions(tmp_path: Path) -> None:
    contents = build_distribution_config(
        host="0.0.0.0",
        port=5000,
        storage_root="/var/lib/registry",
        upstream_url="https://registry-1.docker.io",
        docker_hub_username="user",
        docker_hub_token="secret-token",
    )
    assert 'addr: "0.0.0.0:5000"' in contents
    assert 'remoteurl: "https://registry-1.docker.io"' in contents
    assert 'password: "secret-token"' in contents

    path = tmp_path / "run" / "config.yml"
    write_distribution_config(str(path), contents)
    assert path.read_text() == contents
    assert path.stat().st_mode & 0o777 == 0o600


def test_distribution_config_requires_complete_credentials() -> None:
    with pytest.raises(ValueError, match="supplied together"):
        build_distribution_config(
            host="127.0.0.1",
            port=5000,
            storage_root="/tmp/cache",
            upstream_url="https://registry-1.docker.io",
            docker_hub_username="user",
        )


class _Store:
    async def ping(self):
        return True

    async def close(self):
        self.closed = True


class _Registry:
    server_id = "mirror-test"

    def __init__(self) -> None:
        self.calls = []

    async def register_server(self, url, port, metadata):
        self.calls.append(("register", url, port, metadata))

    async def heartbeat(self, url, port, data=None):
        self.calls.append(("heartbeat", url, port, data))

    async def deregister(self):
        self.calls.append(("deregister",))


def test_registration_is_stateless_and_does_not_publish_credentials() -> None:
    async def check() -> None:
        config = DockerMirrorConfig(
            registry_url="redis://registry:6379",
            advertise_host="mirror.example",
            docker_hub_username="user",
            docker_hub_token="private-token",
            warm_images=("alpine:3.20",),
        )
        registry = _Registry()
        registration = MirrorRegistration(config, store=_Store(), registry=registry)
        await registration.connect()
        await registration.healthy({"health": {"status": "healthy"}})
        await registration.close()

        register = registry.calls[0]
        assert register[:3] == ("register", "http://mirror.example", 5000)
        metadata = register[3]
        assert metadata["model_path"] == "docker-mirror"
        assert metadata["protocol"] == "docker-registry-v2"
        assert "affinity" not in metadata
        assert "private-token" not in json.dumps(metadata)
        assert registry.calls[-1] == ("deregister",)

    asyncio.run(check())


def test_container_defaults_match_open_instruct_setup() -> None:
    dockerfile = (
        Path(__file__).resolve().parents[1] / "docker-mirror" / "Dockerfile"
    ).read_text()
    assert (
        "DOCKER_MIRROR_WARM_IMAGES_FILE=/opt/mirror-assets/"
        "allenai-tmax-15k-open-instruct-images.txt" in dockerfile
    )
    assert "DOCKER_MIRROR_WARM_WORKERS=8" in dockerfile
    assert "DOCKER_MIRROR_WARM_PLATFORM=linux/amd64" in dockerfile


def test_warmer_command_contains_dataset_and_explicit_images() -> None:
    config = DockerMirrorConfig(
        registry_url="redis://registry:6379",
        advertise_host="mirror.example",
        warm_dataset="org/data",
        warm_revision="rev",
        warm_images_file="/assets/images.txt",
        warm_images=("alpine:3.20", "ubuntu:24.04"),
    )
    command = DockerMirrorSupervisor(config, registration=object()).warmer_command()
    assert json.loads(command[command.index("--image") + 1]) == [
        "alpine:3.20",
        "ubuntu:24.04",
    ]
    assert command[command.index("--dataset") + 1] == "org/data"
    assert command[command.index("--images-file") + 1] == "/assets/images.txt"
    assert "--split" not in command
