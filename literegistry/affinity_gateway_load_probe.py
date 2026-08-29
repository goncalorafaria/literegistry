"""Concurrent session-distribution probe for a running affinity gateway."""

from __future__ import annotations

import asyncio
import json
from collections import Counter
from typing import Any

import aiohttp
import fire


async def _post(
    session: aiohttp.ClientSession,
    gateway: str,
    path: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    url = f"{gateway.rstrip('/')}{path}"
    async with session.post(url, json=payload) as response:
        try:
            body = await response.json()
        except (aiohttp.ContentTypeError, ValueError):
            body = {"error": await response.text()}
        if response.status >= 400:
            raise RuntimeError(f"HTTP {response.status} from {url}: {body}")
        if not isinstance(body, dict):
            raise RuntimeError(f"unexpected response from {url}: {body!r}")
        return body


async def _run_session(
    http: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    gateway: str,
    service: str,
    session_index: int,
    requests_per_session: int,
    verbose: bool,
) -> dict[str, str]:
    async with semaphore:
        client_id = f"load-session-{session_index:04d}"
        handshake = await _post(
            http,
            gateway,
            "/affinity/handshake",
            {"service": service, "client_id": client_id},
        )
        affinity_id = handshake.get("affinity_id")
        instance_id = handshake.get("instance_id")
        if not isinstance(affinity_id, str) or not affinity_id:
            raise RuntimeError(f"session {client_id} received no affinity ID")
        if not isinstance(instance_id, str) or not instance_id:
            raise RuntimeError(f"session {client_id} received no instance ID")

        for request_index in range(requests_per_session):
            key = f"session-{session_index}-key-{request_index}"
            value = f"session-{session_index}-value-{request_index}"
            written = await _post(
                http,
                gateway,
                "/affinity/kv/put",
                {
                    "service": service,
                    "affinity_id": affinity_id,
                    "key": key,
                    "value": value,
                },
            )
            if written.get("instance_id") != instance_id:
                raise RuntimeError(
                    f"session {client_id} write moved from {instance_id} "
                    f"to {written.get('instance_id')}"
                )

            read = await _post(
                http,
                gateway,
                "/affinity/kv/get",
                {
                    "service": service,
                    "affinity_id": affinity_id,
                    "key": key,
                },
            )
            if read.get("instance_id") != instance_id:
                raise RuntimeError(
                    f"session {client_id} read moved from {instance_id} "
                    f"to {read.get('instance_id')}"
                )
            if read.get("value") != value:
                raise RuntimeError(
                    f"session {client_id} read wrong value: {read!r}"
                )

        if verbose:
            print(
                f"[affinity-load] SESSION index={session_index:04d} "
                f"instance={instance_id} affinity_id={affinity_id} "
                f"writes={requests_per_session} reads={requests_per_session}",
                flush=True,
            )
        return {
            "client_id": client_id,
            "affinity_id": affinity_id,
            "instance_id": instance_id,
        }


async def run_load_probe(
    gateway: str,
    service: str = "affinity-kv",
    sessions: int = 50,
    requests_per_session: int = 1,
    concurrency: int = 10,
    verbose: bool = False,
) -> dict[str, Any]:
    """Create independent affinity sessions and report replica ownership."""
    if sessions < 1:
        raise ValueError("sessions must be at least 1")
    if requests_per_session < 1:
        raise ValueError("requests_per_session must be at least 1")
    if concurrency < 1:
        raise ValueError("concurrency must be at least 1")

    semaphore = asyncio.Semaphore(concurrency)
    async with aiohttp.ClientSession() as http:
        results = await asyncio.gather(
            *(
                _run_session(
                    http,
                    semaphore,
                    gateway,
                    service,
                    index,
                    requests_per_session,
                    verbose,
                )
                for index in range(sessions)
            )
        )

    counts = Counter(result["instance_id"] for result in results)
    distribution = dict(sorted(counts.items()))
    print(
        "[affinity-load] DISTRIBUTION "
        f"sessions={sessions} instances={json.dumps(distribution, sort_keys=True)}",
        flush=True,
    )
    print(
        "[affinity-load] PASS every session remained pinned "
        f"writes={sessions * requests_per_session} "
        f"reads={sessions * requests_per_session}",
        flush=True,
    )
    return {
        "status": "passed",
        "service": service,
        "sessions": sessions,
        "requests_per_session": requests_per_session,
        "instances": distribution,
        "results": results,
    }


def main(
    gateway: str = "http://127.0.0.1:8080",
    service: str = "affinity-kv",
    sessions: int = 50,
    requests_per_session: int = 1,
    concurrency: int = 10,
    verbose: bool = False,
) -> None:
    """Run the concurrent session-distribution probe."""
    asyncio.run(
        run_load_probe(
            gateway=gateway,
            service=service,
            sessions=sessions,
            requests_per_session=requests_per_session,
            concurrency=concurrency,
            verbose=verbose,
        )
    )


if __name__ == "__main__":
    fire.Fire(main)
