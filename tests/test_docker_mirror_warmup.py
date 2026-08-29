from __future__ import annotations

import io
from pathlib import Path
import sys
import tarfile
from types import SimpleNamespace

import pytest

from literegistry.services import docker_mirror_warmup as warmup


@pytest.mark.parametrize(
    ("image", "expected"),
    [
        ("alpine:3.20", ("library/alpine", "3.20")),
        ("docker.io/allenai/foo", ("allenai/foo", "latest")),
        ("ghcr.io/allenai/foo:latest", None),
    ],
)
def test_dockerhub_reference_parsing(image, expected) -> None:
    assert warmup.dockerhub_repo_and_reference(image) == expected


def test_discover_images_deduplicates_image_files(tmp_path: Path) -> None:
    for dirname, image in (
        ("one", "ubuntu:24.04"),
        ("two", "ubuntu:24.04"),
        ("three", "alpine:3.20"),
    ):
        directory = tmp_path / dirname
        directory.mkdir()
        (directory / "image.txt").write_text(image + "\n")
    assert warmup.discover_images(tmp_path) == ["alpine:3.20", "ubuntu:24.04"]


def test_safe_extract_rejects_path_traversal(tmp_path: Path) -> None:
    archive_path = tmp_path / "bad.tar"
    with tarfile.open(archive_path, "w") as archive:
        item = tarfile.TarInfo("../image.txt")
        payload = b"bad"
        item.size = len(payload)
        archive.addfile(item, io.BytesIO(payload))
    with tarfile.open(archive_path) as archive:
        with pytest.raises(ValueError, match="escapes destination"):
            warmup._safe_extract_image_files(archive, tmp_path / "out")


def test_extract_ignores_everything_except_image_txt(tmp_path: Path) -> None:
    archive_path = tmp_path / "task-data.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        for name, payload in (
            ("task/one/image.txt", b"ubuntu:24.04\n"),
            ("task/one/problem.txt", b"large task body"),
        ):
            item = tarfile.TarInfo(name)
            item.size = len(payload)
            archive.addfile(item, io.BytesIO(payload))
    destination = tmp_path / "out"
    destination.mkdir()
    with tarfile.open(archive_path, "r:gz") as archive:
        warmup._safe_extract_image_files(archive, destination)

    assert (destination / "task/one/image.txt").read_text() == "ubuntu:24.04\n"
    assert not (destination / "task/one/problem.txt").exists()


def test_bundled_open_instruct_default_image_list() -> None:
    asset = (
        Path(__file__).resolve().parents[1]
        / "docker-mirror"
        / "assets"
        / "allenai-tmax-15k-open-instruct-images.txt"
    )
    images = warmup.load_images_file(str(asset))
    assert len(images) == 14_490
    assert len(set(images)) == 14_490
    assert all(image.startswith("hamishi740/swerl-tmax-v3:") for image in images)


def test_load_images_file_ignores_comments_and_deduplicates(tmp_path: Path) -> None:
    path = tmp_path / "images.txt"
    path.write_text(
        "# generated asset\nubuntu:24.04\nalpine:3.20\nubuntu:24.04\n"
    )
    assert warmup.load_images_file(str(path)) == [
        "alpine:3.20",
        "ubuntu:24.04",
    ]



def test_dataset_download_is_filtered_to_image_files(monkeypatch, tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    (dataset_dir / "image.txt").write_text("alpine:3.20\n")
    calls = []

    def snapshot_download(*args, **kwargs):
        calls.append((args, kwargs))
        return str(dataset_dir)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=snapshot_download),
    )

    result = warmup.resolve_task_data_dir("org/data", revision="abc")

    assert result == dataset_dir
    assert calls[0][0] == ("org/data",)
    assert calls[0][1]["repo_type"] == "dataset"
    assert calls[0][1]["revision"] == "abc"
    assert calls[0][1]["allow_patterns"] == [
        "image.txt",
        "**/image.txt",
        "task-data.tar.gz",
        "**/task-data.tar.gz",
    ]


def test_warm_image_fetches_matching_manifest_and_blobs(monkeypatch) -> None:
    manifests = {
        "library/demo/manifests/latest": (
            {
                "mediaType": "application/vnd.oci.image.index.v1+json",
                "manifests": [
                    {
                        "digest": "sha256:amd64",
                        "platform": {"os": "linux", "architecture": "amd64"},
                    },
                    {
                        "digest": "sha256:arm64",
                        "platform": {"os": "linux", "architecture": "arm64"},
                    },
                ],
            },
            "sha256:index",
        ),
        "library/demo/manifests/sha256:amd64": (
            {
                "mediaType": "application/vnd.oci.image.manifest.v1+json",
                "config": {"digest": "sha256:config"},
                "layers": [{"digest": "sha256:layer"}],
            },
            "sha256:amd64",
        ),
    }
    blobs = []
    monkeypatch.setattr(
        warmup,
        "registry_get_json",
        lambda mirror, path: manifests[path],
    )
    monkeypatch.setattr(
        warmup,
        "registry_get_blob",
        lambda mirror, repo, digest, dry_run: blobs.append((repo, digest)) or 10,
    )

    image, ok, detail = warmup.warm_image(
        "demo:latest", "mirror:5000", platform="linux/amd64"
    )

    assert (image, ok) == ("demo:latest", True)
    assert sorted(blobs) == [
        ("library/demo", "sha256:config"),
        ("library/demo", "sha256:layer"),
    ]
    assert "blobs=2 bytes=20" in detail
