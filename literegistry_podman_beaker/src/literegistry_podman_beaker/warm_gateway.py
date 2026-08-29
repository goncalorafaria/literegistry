"""Warm every bundled Tmax image through a LiteRegistry gateway."""

from __future__ import annotations

import asyncio
from importlib.resources import as_file, files
import json
from pathlib import Path
import random
import time
from typing import Any
from urllib.parse import urlsplit
from urllib.request import urlopen

import fire
from tqdm import tqdm

from literegistry.services.docker_mirror_warmup import (
    load_images_file,
    warm_image,
)


ASSET_NAME = "allenai-tmax-15k-open-instruct-images.txt"


def _normalize_gateway_url(gateway_url: str) -> str:
    value = gateway_url.strip().rstrip("/")
    parsed = urlsplit(value)
    if parsed.scheme != "http" or not parsed.netloc:
        raise ValueError("gateway_url must be an absolute http:// URL")
    if parsed.path or parsed.query or parsed.fragment:
        raise ValueError("gateway_url must point to the gateway root")
    return value


def _active_mirror_count(gateway_url: str) -> int:
    with urlopen(f"{gateway_url}/v1/models", timeout=10) as response:
        payload = json.load(response)
    for service in payload.get("data", []):
        if service.get("id") == "docker-mirror":
            return sum(
                record.get("status", "active") == "active"
                for record in service.get("metadata", [])
            )
    return 0


async def _wait_for_mirrors(
    gateway_url: str,
    expected_mirrors: int,
    timeout: float,
) -> int:
    deadline = time.monotonic() + timeout
    last_count = 0
    while time.monotonic() < deadline:
        try:
            last_count = await asyncio.to_thread(_active_mirror_count, gateway_url)
        except Exception:
            last_count = 0
        if last_count >= expected_mirrors:
            return last_count
        await asyncio.sleep(2)
    raise TimeoutError(
        f"gateway exposed {last_count}/{expected_mirrors} expected mirrors "
        f"after {timeout:g}s"
    )


async def _warm_one(
    gateway_url: str,
    image: str,
    semaphore: asyncio.Semaphore,
    platform: str,
) -> tuple[str, bool, str]:
    async with semaphore:
        return await asyncio.to_thread(
            warm_image,
            image,
            gateway_url,
            platform=platform,
        )


async def warm_gateway(
    gateway_url: str,
    *,
    images_file: str | None = None,
    concurrency: int = 32,
    expected_mirrors: int = 1,
    wait_timeout: float = 600,
    platform: str = "linux/amd64",
    shuffle_seed: int = 0,
    limit: int | None = None,
) -> dict[str, Any]:
    """Fill mirror caches through the gateway and create Redis affinities."""
    gateway_url = _normalize_gateway_url(gateway_url)
    if concurrency < 1:
        raise ValueError("concurrency must be at least one")
    if expected_mirrors < 1:
        raise ValueError("expected_mirrors must be at least one")
    if wait_timeout <= 0:
        raise ValueError("wait_timeout must be positive")
    if limit is not None and limit < 1:
        raise ValueError("limit must be at least one when supplied")

    mirror_count = await _wait_for_mirrors(
        gateway_url,
        expected_mirrors,
        wait_timeout,
    )
    if images_file is None:
        asset = files("literegistry_podman_beaker").joinpath("assets", ASSET_NAME)
        with as_file(asset) as asset_path:
            images = load_images_file(str(asset_path))
    else:
        images = load_images_file(str(Path(images_file)))
    random.Random(shuffle_seed).shuffle(images)
    if limit is not None:
        images = images[:limit]

    semaphore = asyncio.Semaphore(concurrency)
    started = time.perf_counter()
    tasks = [
        asyncio.create_task(_warm_one(gateway_url, image, semaphore, platform))
        for image in images
    ]
    failures: list[dict[str, str]] = []
    completed = 0
    with tqdm(total=len(tasks), unit="image", desc="warming gateway") as progress:
        for future in asyncio.as_completed(tasks):
            image, success, detail = await future
            completed += 1
            if not success:
                failures.append({"image": image, "error": detail})
            progress.update(1)
            progress.set_postfix(ok=completed - len(failures), failed=len(failures))

    elapsed = time.perf_counter() - started
    return {
        "gateway_url": gateway_url,
        "mirrors": mirror_count,
        "images": len(images),
        "successes": len(images) - len(failures),
        "failures": len(failures),
        "elapsed_seconds": elapsed,
        "images_per_second": len(images) / elapsed if elapsed else 0.0,
        "failure_details": failures,
    }


def main(
    gateway_url: str,
    images_file: str | None = None,
    concurrency: int = 32,
    expected_mirrors: int = 1,
    wait_timeout: float = 600,
    platform: str = "linux/amd64",
    shuffle_seed: int = 0,
    limit: int | None = None,
) -> None:
    """Run the asynchronous gateway warmer and print its final summary."""
    result = asyncio.run(
        warm_gateway(
            gateway_url,
            images_file=images_file,
            concurrency=concurrency,
            expected_mirrors=expected_mirrors,
            wait_timeout=wait_timeout,
            platform=platform,
            shuffle_seed=shuffle_seed,
            limit=limit,
        )
    )
    print(json.dumps(result, indent=2))
    if result["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    fire.Fire(main)
