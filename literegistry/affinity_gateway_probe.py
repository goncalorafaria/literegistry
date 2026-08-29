"""Live strict-affinity probe for a running LiteRegistry Gateway instance."""

from __future__ import annotations

import asyncio
import json
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


async def run_probe(
    gateway: str,
    service: str,
    count: int,
) -> dict[str, Any]:
    if count < 1:
        raise ValueError("count must be at least 1")
    async with aiohttp.ClientSession() as session:
        handshake = await _post(
            session,
            gateway,
            "/affinity/handshake",
            {"service": service, "client_id": "affinity-gateway-probe"},
        )
        affinity_id = handshake.get("affinity_id")
        instance_id = handshake.get("instance_id")
        if not isinstance(affinity_id, str) or not affinity_id:
            raise RuntimeError(f"handshake returned no affinity ID: {handshake}")
        if not isinstance(instance_id, str) or not instance_id:
            raise RuntimeError(f"handshake returned no instance ID: {handshake}")
        print("HANDSHAKE", json.dumps(handshake, sort_keys=True), flush=True)

        for index in range(count):
            result = await _post(
                session,
                gateway,
                "/affinity/kv/put",
                {
                    "service": service,
                    "affinity_id": affinity_id,
                    "key": f"key-{index}",
                    "value": f"value-{index}",
                },
            )
            if result.get("instance_id") != instance_id:
                raise RuntimeError(f"write reached wrong instance: {result}")
            print("PUT", json.dumps(result, sort_keys=True), flush=True)

        for index in range(count):
            result = await _post(
                session,
                gateway,
                "/affinity/kv/get",
                {
                    "service": service,
                    "affinity_id": affinity_id,
                    "key": f"key-{index}",
                },
            )
            if result.get("instance_id") != instance_id:
                raise RuntimeError(f"read reached wrong instance: {result}")
            if result.get("value") != f"value-{index}":
                raise RuntimeError(f"read returned wrong value: {result}")
            print("GET", json.dumps(result, sort_keys=True), flush=True)

    return {
        "status": "passed",
        "service": service,
        "affinity_id": affinity_id,
        "instance_id": instance_id,
        "writes": count,
        "reads": count,
    }


def main(
    gateway: str = "http://127.0.0.1:8080",
    service: str = "affinity-kv",
    count: int = 5,
) -> None:
    """Handshake and verify pinned writes/reads through the LiteRegistry Gateway."""
    result = asyncio.run(run_probe(gateway, service, count))
    print("PASS", json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    fire.Fire(main)

