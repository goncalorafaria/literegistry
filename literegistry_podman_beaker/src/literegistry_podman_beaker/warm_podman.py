"""Warm session images through the public async Podman client."""

from __future__ import annotations

import asyncio
import hashlib
from importlib.resources import as_file, files
import json
from pathlib import Path
import random
import sys
import time
from typing import Any
from urllib.parse import urlsplit
from urllib.request import urlopen

import aiohttp
import fire
from tqdm import tqdm

from literegistry.services.docker_mirror_warmup import load_images_file
from literegistry_podman_client import PodmanClient


ASSET_NAME = "allenai-tmax-15k-open-instruct-images.txt"


def _normalize_gateway_url(gateway_url: str) -> str:
    value = gateway_url.strip().rstrip("/")
    parsed = urlsplit(value)
    if parsed.scheme != "http" or not parsed.netloc:
        raise ValueError("gateway_url must be an absolute http:// URL")
    if parsed.path or parsed.query or parsed.fragment:
        raise ValueError("gateway_url must point to the gateway root")
    return value


def _active_service_count(gateway_url: str, service_name: str) -> int:
    with urlopen(f"{gateway_url}/v1/models", timeout=10) as response:
        payload = json.load(response)
    for service in payload.get("data", []):
        if service.get("id") == service_name:
            return sum(
                record.get("status", "active") == "active"
                for record in service.get("metadata", [])
            )
    return 0


async def _wait_for_podman(
    gateway_url: str,
    expected_podman: int,
    timeout: float,
) -> int:
    deadline = time.monotonic() + timeout
    last_count = 0
    while time.monotonic() < deadline:
        try:
            last_count = await asyncio.to_thread(
                _active_service_count,
                gateway_url,
                "podman",
            )
        except Exception:
            last_count = 0
        if last_count >= expected_podman:
            return last_count
        await asyncio.sleep(2)
    raise TimeoutError(
        f"gateway exposed {last_count}/{expected_podman} expected Podman replicas "
        f"after {timeout:g}s"
    )


def _load_checkpoint(path: Path | None) -> set[str]:
    if path is None or not path.exists():
        return set()
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


async def _warm_one(
    client: PodmanClient,
    image: str,
    semaphore: asyncio.Semaphore,
    *,
    retries: int,
    execute_probe: bool,
) -> tuple[str, bool, str, int]:
    async with semaphore:
        last_error = "unknown error"
        for attempt in range(1, retries + 1):
            session = None
            try:
                session = await client.handshake(
                    image=image,
                    client_id=(
                        "mirror-warmup-"
                        + hashlib.sha256(image.encode("utf-8")).hexdigest()[:24]
                    ),
                )
                if execute_probe:
                    await session.execute("true", timeout=60, check=True)
                await session.close()
                session = None
                return image, True, "ok", attempt
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            finally:
                if session is not None:
                    try:
                        await session.close()
                    except Exception as close_exc:
                        last_error += (
                            f"; close {type(close_exc).__name__}: {close_exc}"
                        )
            if attempt < retries:
                await asyncio.sleep(min(30.0, float(2 ** (attempt - 1))))
        return image, False, last_error, retries


async def warm_podman(
    gateway_url: str,
    *,
    images_file: str | None = None,
    concurrency: int = 64,
    expected_podman: int = 1,
    wait_timeout: float = 600,
    shuffle_seed: int = 0,
    limit: int | None = None,
    retries: int = 3,
    execute_probe: bool = True,
    checkpoint_file: str | None = None,
    log_every: int = 25,
) -> dict[str, Any]:
    """Pull images via Podman handshake and close every created session."""
    gateway_url = _normalize_gateway_url(gateway_url)
    if concurrency < 1:
        raise ValueError("concurrency must be at least one")
    if expected_podman < 1:
        raise ValueError("expected_podman must be at least one")
    if wait_timeout <= 0:
        raise ValueError("wait_timeout must be positive")
    if limit is not None and limit < 1:
        raise ValueError("limit must be at least one when supplied")
    if retries < 1:
        raise ValueError("retries must be at least one")
    if log_every < 1:
        raise ValueError("log_every must be at least one")

    podman_count = await _wait_for_podman(
        gateway_url,
        expected_podman,
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

    checkpoint_path = Path(checkpoint_file) if checkpoint_file else None
    succeeded_before = _load_checkpoint(checkpoint_path)
    pending = [image for image in images if image not in succeeded_before]
    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        "PODMAN_WARMUP_START "
        f"images={len(images)} resumed={len(images) - len(pending)} "
        f"pending={len(pending)} concurrency={concurrency} "
        f"podman_replicas={podman_count} gateway={gateway_url}",
        flush=True,
    )

    semaphore = asyncio.Semaphore(concurrency)
    checkpoint_lock = asyncio.Lock()
    started = time.perf_counter()
    failures: list[dict[str, Any]] = []
    completed = 0
    successes = 0
    http_session = aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(limit=concurrency),
        timeout=aiohttp.ClientTimeout(total=310),
        trust_env=False,
    )
    client = PodmanClient(
        gateway_url,
        workdir="/tmp",
        request_timeout=310,
        http_session=http_session,
    )
    await client.open()
    try:
        tasks = [
            asyncio.create_task(
                _warm_one(
                    client,
                    image,
                    semaphore,
                    retries=retries,
                    execute_probe=execute_probe,
                )
            )
            for image in pending
        ]
        with tqdm(
            total=len(pending),
            unit="image",
            desc="warming via PodmanClient",
            disable=not sys.stderr.isatty(),
        ) as progress:
            for future in asyncio.as_completed(tasks):
                image, success, detail, attempts = await future
                completed += 1
                if success:
                    successes += 1
                    if checkpoint_path is not None:
                        async with checkpoint_lock:
                            with checkpoint_path.open("a", encoding="utf-8") as stream:
                                stream.write(image + "\n")
                                stream.flush()
                else:
                    failures.append(
                        {"image": image, "error": detail, "attempts": attempts}
                    )
                progress.update(1)
                progress.set_postfix(ok=successes, failed=len(failures))
                if completed % log_every == 0 or completed == len(pending):
                    elapsed = time.perf_counter() - started
                    print(
                        "PODMAN_WARMUP_PROGRESS "
                        f"completed={completed} total={len(pending)} "
                        f"succeeded={successes} failed={len(failures)} "
                        f"images_per_second={completed / elapsed if elapsed else 0:.3f}",
                        flush=True,
                    )
    finally:
        await client.aclose()
        await http_session.close()

    elapsed = time.perf_counter() - started
    result = {
        "gateway_url": gateway_url,
        "podman_replicas": podman_count,
        "images": len(images),
        "resumed": len(images) - len(pending),
        "attempted": len(pending),
        "successes": successes,
        "failures": len(failures),
        "elapsed_seconds": elapsed,
        "images_per_second": len(pending) / elapsed if elapsed else 0.0,
        "connection_limit": concurrency,
        "checkpoint_file": str(checkpoint_path) if checkpoint_path else None,
        "failure_details": failures[:100],
    }
    print("PODMAN_WARMUP_COMPLETE " + json.dumps(result, sort_keys=True), flush=True)
    return result


def run(
    gateway_url: str,
    images_file: str | None = None,
    concurrency: int = 64,
    expected_podman: int = 1,
    wait_timeout: float = 600,
    shuffle_seed: int = 0,
    limit: int | None = None,
    retries: int = 3,
    execute_probe: bool = True,
    checkpoint_file: str | None = None,
    log_every: int = 25,
) -> None:
    """Run the resumable asynchronous Podman-client warmer."""
    result = asyncio.run(
        warm_podman(
            gateway_url,
            images_file=images_file,
            concurrency=concurrency,
            expected_podman=expected_podman,
            wait_timeout=wait_timeout,
            shuffle_seed=shuffle_seed,
            limit=limit,
            retries=retries,
            execute_probe=execute_probe,
            checkpoint_file=checkpoint_file,
            log_every=log_every,
        )
    )
    if result["failures"]:
        raise SystemExit(1)


def main() -> None:
    """Expose :func:`run` through the package's standard Fire CLI."""
    fire.Fire(run)


if __name__ == "__main__":
    main()
