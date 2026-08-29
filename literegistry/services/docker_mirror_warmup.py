"""Warm a Docker Hub pull-through cache from Tmax/SWERL image references.

This implementation follows the registry-API strategy from AllenAI Open
Instruct's ``warm_tmax_registry_mirror.py`` (Apache-2.0): fetch only the
``image.txt`` task-data files from a Hugging Face dataset, then request those
images' manifests and blobs through the mirror. It intentionally needs no
Docker or Podman daemon.
"""

from __future__ import annotations

import concurrent.futures
import fire
import json
from pathlib import Path
from literegistry.coop.artifacts import ensure_directory_artifact
import re
import tarfile
import threading
from typing import Any, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


MANIFEST_ACCEPT = ", ".join(
    [
        "application/vnd.oci.image.index.v1+json",
        "application/vnd.docker.distribution.manifest.list.v2+json",
        "application/vnd.oci.image.manifest.v1+json",
        "application/vnd.docker.distribution.manifest.v2+json",
    ]
)
INDEX_MEDIA_TYPES = {
    "application/vnd.oci.image.index.v1+json",
    "application/vnd.docker.distribution.manifest.list.v2+json",
}
_WARMED_BLOBS: set[str] = set()
_WARMED_BLOBS_LOCK = threading.Lock()


def normalize_mirror(mirror: str) -> str:
    value = re.sub(r"^https?://", "", mirror.strip()).rstrip("/")
    if not value:
        raise ValueError("mirror must be non-empty")
    return value


def _split_registry(image: str) -> tuple[str | None, str]:
    first, separator, remainder = image.partition("/")
    if separator and ("." in first or ":" in first or first == "localhost"):
        return first, remainder
    return None, image


def dockerhub_repo_and_reference(image: str) -> tuple[str, str] | None:
    registry, remainder = _split_registry(image.strip())
    if registry is not None and registry not in {
        "docker.io",
        "registry-1.docker.io",
        "index.docker.io",
    }:
        return None
    if not remainder:
        return None
    if "/" not in remainder:
        remainder = f"library/{remainder}"
    if "@" in remainder:
        return tuple(remainder.rsplit("@", 1))  # type: ignore[return-value]
    last_slash = remainder.rfind("/")
    last_colon = remainder.rfind(":")
    if last_colon > last_slash:
        return remainder[:last_colon], remainder[last_colon + 1 :]
    return remainder, "latest"


def registry_url(mirror: str, path: str) -> str:
    return f"http://{normalize_mirror(mirror)}/v2/{path.lstrip('/')}"


def registry_get_json(mirror: str, path: str) -> tuple[dict[str, Any], str]:
    request = Request(
        registry_url(mirror, path),
        headers={"Accept": MANIFEST_ACCEPT},
    )
    with urlopen(request, timeout=300) as response:
        payload = json.loads(response.read().decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("registry returned a non-object manifest")
        return payload, response.headers.get("Docker-Content-Digest", "")


def registry_get_blob(mirror: str, repo: str, digest: str, dry_run: bool) -> int:
    with _WARMED_BLOBS_LOCK:
        if digest in _WARMED_BLOBS:
            return 0
        _WARMED_BLOBS.add(digest)
    if dry_run:
        return 0
    request = Request(registry_url(mirror, f"{repo}/blobs/{digest}"))
    bytes_read = 0
    with urlopen(request, timeout=300) as response:
        while chunk := response.read(1024 * 1024):
            bytes_read += len(chunk)
    return bytes_read


def _manifest_matches_platform(manifest: dict[str, Any], platform: str) -> bool:
    if platform == "all":
        return True
    expected = platform.split("/")
    if len(expected) not in {2, 3}:
        raise ValueError("platform must be os/arch, os/arch/variant, or all")
    actual = manifest.get("platform", {})
    if actual.get("os") != expected[0] or actual.get("architecture") != expected[1]:
        return False
    return len(expected) == 2 or actual.get("variant") == expected[2]


def _manifest_blob_digests(manifest: dict[str, Any]) -> set[str]:
    digests: set[str] = set()
    config = manifest.get("config", {})
    if isinstance(config, dict) and isinstance(config.get("digest"), str):
        digests.add(config["digest"])
    for layer in manifest.get("layers", []):
        if isinstance(layer, dict) and isinstance(layer.get("digest"), str):
            digests.add(layer["digest"])
    return digests


def warm_image(
    image: str,
    mirror: str,
    *,
    dry_run: bool = False,
    platform: str = "linux/amd64",
) -> tuple[str, bool, str]:
    parsed = dockerhub_repo_and_reference(image)
    if parsed is None:
        return image, True, "skipped non-Docker-Hub image"
    repo, reference = parsed
    try:
        if dry_run:
            return image, True, f"GET {registry_url(mirror, f'{repo}/manifests/{reference}')}"
        manifest, digest = registry_get_json(mirror, f"{repo}/manifests/{reference}")
        if manifest.get("mediaType") in INDEX_MEDIA_TYPES:
            selected = [
                item["digest"]
                for item in manifest.get("manifests", [])
                if isinstance(item, dict)
                and isinstance(item.get("digest"), str)
                and _manifest_matches_platform(item, platform)
            ]
            if not selected:
                return image, False, f"no manifest matched platform {platform}"
            manifests = [
                registry_get_json(mirror, f"{repo}/manifests/{child_digest}")[0]
                for child_digest in selected
            ]
        else:
            manifests = [manifest]
        blob_digests = set().union(*(_manifest_blob_digests(item) for item in manifests))
        bytes_read = sum(
            registry_get_blob(mirror, repo, blob_digest, dry_run=False)
            for blob_digest in sorted(blob_digests)
        )
        return image, True, (
            f"manifest={digest or reference} blobs={len(blob_digests)} bytes={bytes_read}"
        )
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
        return image, False, str(exc)


def _safe_extract_image_files(
    tar: tarfile.TarFile,
    destination: Path,
) -> None:
    """Extract only safe ``image.txt`` members from task-data."""
    root = destination.resolve()
    members = [
        member
        for member in tar.getmembers()
        if Path(member.name).name == "image.txt"
    ]
    for member in members:
        if member.issym() or member.islnk():
            raise ValueError(f"task-data archive contains a link: {member.name}")
        target = (destination / member.name).resolve()
        if target != root and root not in target.parents:
            raise ValueError(f"task-data archive escapes destination: {member.name}")
    tar.extractall(destination, members=members)


def resolve_task_data_dir(dataset: str, revision: str | None = None) -> Path:
    """Download and expose only task-data ``image.txt`` files."""
    from huggingface_hub import snapshot_download

    repo_dir = Path(
        snapshot_download(
            dataset,
            repo_type="dataset",
            revision=revision,
            allow_patterns=[
                "image.txt",
                "**/image.txt",
                "task-data.tar.gz",
                "**/task-data.tar.gz",
            ],
        )
    )
    for tarball in repo_dir.rglob("task-data.tar.gz"):
        extract_dir = Path(f"{tarball}.image-files")

        def ready(directory: Path) -> bool:
            return (directory / ".extract-complete").is_file()

        def extract(staging: Path) -> None:
            with tarfile.open(tarball, mode="r:gz") as archive:
                _safe_extract_image_files(archive, staging)
            (staging / ".extract-complete").write_text("ok\n", encoding="utf-8")

        ensure_directory_artifact(extract_dir, ready=ready, build=extract)
    return repo_dir


def discover_images(task_data_dir: Path) -> list[str]:
    images: set[str] = set()
    for image_file in task_data_dir.rglob("image.txt"):
        image = image_file.read_text(encoding="utf-8").strip()
        if image:
            images.add(image)
    return sorted(images)


def load_images_file(raw: str | None) -> list[str]:
    if not raw:
        return []
    path = Path(raw)
    images = {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    return sorted(images)


def load_tool_configs(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    maybe_path = Path(raw)
    if maybe_path.is_file():
        raw = maybe_path.read_text(encoding="utf-8")
    config = json.loads(raw)
    if not isinstance(config, dict):
        raise ValueError("--tool-configs must be a JSON object or path")
    return config


def _images(value: str | Sequence[str]) -> list[str]:
    if isinstance(value, str):
        return [value]
    return list(value)


def main(
    mirror: str,
    dataset: str | None = None,
    revision: str | None = None,
    tool_configs: str | None = None,
    image: str | Sequence[str] = (),
    images_file: str | None = None,
    workers: int = 8,
    platform: str = "linux/amd64",
    dry_run: bool = False,
    verbose: bool = False,
    images_out: str | None = None,
) -> None:
    """Warm unique Docker Hub images through a registry mirror."""
    if workers < 1:
        raise ValueError("--workers must be at least one")
    loaded_tool_configs = load_tool_configs(tool_configs)
    dataset = dataset or loaded_tool_configs.get("task_data_hf_repo")
    images = set(_images(image))
    images.update(load_images_file(images_file))
    task_data_dir: Path | None = None
    if dataset:
        task_data_dir = resolve_task_data_dir(dataset, revision=revision)
        images.update(discover_images(task_data_dir))
    images = {candidate.strip() for candidate in images if candidate and candidate.strip()}
    if not images:
        raise ValueError(
            "provide --images-file, --dataset, --tool-configs with "
            "task_data_hf_repo, or at least one --image"
        )
    ordered_images = sorted(images)
    if images_out:
        Path(images_out).write_text(
            "\n".join(ordered_images) + "\n",
            encoding="utf-8",
        )

    print(
        f"Discovered {len(ordered_images)} unique images"
        + (f" in {task_data_dir}" if task_data_dir else ""),
        flush=True,
    )
    print(
        f"Warming Docker Hub images through mirror {normalize_mirror(mirror)} "
        f"with {workers} workers",
        flush=True,
    )
    failures: list[tuple[str, str]] = []
    completed_count = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_image = {
            executor.submit(
                warm_image,
                candidate,
                mirror,
                dry_run=dry_run,
                platform=platform,
            ): candidate
            for candidate in ordered_images
        }
        for future in concurrent.futures.as_completed(future_to_image):
            candidate, ok, output = future.result()
            completed_count += 1
            if verbose or not ok:
                print(f"[{'OK' if ok else 'FAILED'}] {candidate}: {output}", flush=True)
            if not ok:
                failures.append((candidate, output))
            elif completed_count % 25 == 0 or completed_count == len(ordered_images):
                print(
                    f"Warmed {completed_count}/{len(ordered_images)} images; "
                    f"failures={len(failures)}",
                    flush=True,
                )
    if failures:
        raise SystemExit(f"{len(failures)} image warmups failed")


if __name__ == "__main__":
    fire.Fire(main)
