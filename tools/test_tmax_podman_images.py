#!/usr/bin/env python3
"""Exercise representative Tmax images through a LiteRegistry Podman gateway."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import time
from typing import Any

import fire

from literegistry_podman_client import PodmanClient


async def _exercise_image(
    client: PodmanClient,
    image: str,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    started = time.perf_counter()
    session = None
    result: dict[str, Any] = {"image": image, "success": False}
    async with semaphore:
        try:
            handshake_started = time.perf_counter()
            session = await client.handshake(image=image)
            result.update(
                {
                    "container_id": session.container_id,
                    "instance_id": session.instance_id,
                    "handshake_seconds": time.perf_counter() - handshake_started,
                }
            )

            write = await session.execute(
                "printf '%s\\n' 'ai2 hello' > /tmp/literegistry-ai2-hello.txt",
                timeout=60,
                workdir="/",
                check=True,
            )
            read = await session.execute(
                "cat /tmp/literegistry-ai2-hello.txt",
                timeout=60,
                workdir="/",
                check=True,
            )
            python = await session.execute(
                "python -c \"print('python ok')\"",
                timeout=60,
                workdir="/",
            )
            result.update(
                {
                    "success": read.stdout.strip() == "ai2 hello",
                    "write_seconds": write.execution_time,
                    "read_seconds": read.execution_time,
                    "read_stdout": read.stdout.strip(),
                    "python_success": python.success,
                    "python_stdout": python.stdout.strip(),
                    "python_stderr": python.stderr.strip(),
                }
            )
        except Exception as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            if session is not None:
                close_started = time.perf_counter()
                try:
                    await session.close()
                    result["close_success"] = True
                except Exception as exc:
                    result["close_success"] = False
                    result["close_error"] = f"{type(exc).__name__}: {exc}"
                result["close_seconds"] = time.perf_counter() - close_started
            result["total_seconds"] = time.perf_counter() - started
    return result


async def _run(gateway_url: str, images_file: str, concurrency: int) -> dict[str, Any]:
    images = [
        line.strip()
        for line in Path(images_file).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not images:
        raise ValueError("images_file contains no images")
    if concurrency < 1:
        raise ValueError("concurrency must be at least one")

    client = PodmanClient(gateway_url, workdir="/", request_timeout=900)
    await client.open()
    try:
        health = await client.health()
        semaphore = asyncio.Semaphore(concurrency)
        started = time.perf_counter()
        results = await asyncio.gather(
            *(_exercise_image(client, image, semaphore) for image in images)
        )
        elapsed = time.perf_counter() - started
    finally:
        await client.aclose()

    successes = sum(bool(item["success"]) for item in results)
    return {
        "gateway_url": gateway_url,
        "health": health,
        "images": len(images),
        "successes": successes,
        "failures": len(images) - successes,
        "concurrency": concurrency,
        "elapsed_seconds": elapsed,
        "images_per_second": len(images) / elapsed if elapsed else 0.0,
        "instances": sorted(
            {str(item["instance_id"]) for item in results if item.get("instance_id")}
        ),
        "results": results,
    }


def main(
    gateway_url: str,
    images_file: str = "tools/tmax-podman-sample-images.txt",
    concurrency: int = 4,
) -> None:
    """Run the asynchronous image lifecycle test and print JSON results."""
    print(json.dumps(asyncio.run(_run(gateway_url, images_file, concurrency)), indent=2))


if __name__ == "__main__":
    fire.Fire(main)
