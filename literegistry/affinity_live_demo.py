"""Self-contained live HTTP demonstration of strict affinity routing."""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from asyncio.subprocess import Process
from typing import Optional

import aiohttp
import fire

from literegistry.affinity_gateway_probe import run_probe
from literegistry.affinity_gateway_load_probe import run_load_probe


async def _wait_until_ready(
    session: aiohttp.ClientSession,
    url: str,
    process: Process,
    *,
    timeout: float = 15.0,
) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    last_error: Optional[Exception] = None
    while loop.time() < deadline:
        if process.returncode is not None:
            raise RuntimeError(
                f"process exited with status {process.returncode} before {url} was ready"
            )
        try:
            async with session.get(url) as response:
                if response.status < 500:
                    return
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            last_error = exc
        await asyncio.sleep(0.05)
    raise RuntimeError(f"timed out waiting for {url}: {last_error}")


async def _stop(process: Process) -> None:
    if process.returncode is not None:
        return
    process.terminate()
    try:
        await asyncio.wait_for(process.wait(), timeout=5)
    except asyncio.TimeoutError:
        process.kill()
        await process.wait()


async def _spawn(*args: str) -> Process:
    return await asyncio.create_subprocess_exec(
        sys.executable,
        *args,
    )


async def run_live_demo(
    *,
    registry: Optional[str] = None,
    gateway_port: int = 18080,
    replica_a_port: int = 18091,
    replica_b_port: int = 18092,
    sessions: int = 20,
    count: int = 1,
    concurrency: int = 10,
) -> dict:
    """Run real Uvicorn replicas and a real gateway in one network namespace."""
    temporary_registry = None
    if registry is None:
        temporary_registry = tempfile.TemporaryDirectory(
            prefix="literegistry-affinity-live-"
        )
        registry = temporary_registry.name

    processes: list[Process] = []
    try:
        print(f"[affinity-live] REGISTRY {registry}", flush=True)
        for instance_id, port in (
            ("replica-a", replica_a_port),
            ("replica-b", replica_b_port),
        ):
            process = await _spawn(
                "-m",
                "literegistry.services.affinity_mock_server",
                f"--registry={registry}",
                "--service_name=affinity-kv",
                f"--instance_id={instance_id}",
                "--host=127.0.0.1",
                f"--port={port}",
            )
            processes.append(process)

        async with aiohttp.ClientSession() as session:
            await _wait_until_ready(
                session,
                f"http://127.0.0.1:{replica_a_port}/health",
                processes[0],
            )
            await _wait_until_ready(
                session,
                f"http://127.0.0.1:{replica_b_port}/health",
                processes[1],
            )
            print(
                "[affinity-live] UPSTREAMS_READY "
                f"replica-a=127.0.0.1:{replica_a_port} "
                f"replica-b=127.0.0.1:{replica_b_port}",
                flush=True,
            )

            gateway = await _spawn(
                "-m",
                "literegistry.gateway",
                f"--registry={registry}",
                "--host=127.0.0.1",
                f"--port={gateway_port}",
                "--access_log=True",
            )
            processes.append(gateway)
            gateway_url = f"http://127.0.0.1:{gateway_port}"
            await _wait_until_ready(
                session,
                f"{gateway_url}/health",
                gateway,
            )
            print(f"[affinity-live] GATEWAY_READY {gateway_url}", flush=True)

            async with session.get(f"{gateway_url}/v1/models") as response:
                models = await response.json()
                if response.status != 200:
                    raise RuntimeError(
                        f"gateway model discovery failed: HTTP {response.status}: {models}"
                    )
            print(
                "[affinity-live] GATEWAY_MODELS "
                f"{json.dumps(models, sort_keys=True)}",
                flush=True,
            )

        print("[affinity-live] DISTRIBUTION_PROBE_BEGIN", flush=True)
        distribution = await run_load_probe(
            gateway_url,
            "affinity-kv",
            sessions=sessions,
            requests_per_session=count,
            concurrency=concurrency,
        )

        print("[affinity-live] SESSION_1_BEGIN", flush=True)
        first = await run_probe(gateway_url, "affinity-kv", count)
        print(
            "[affinity-live] SESSION_1_PINNED "
            f"instance={first['instance_id']} affinity_id={first['affinity_id']}",
            flush=True,
        )

        selected_process = {
            "replica-a": processes[0],
            "replica-b": processes[1],
        }.get(first["instance_id"])
        if selected_process is None:
            raise RuntimeError(
                f"unexpected first-session instance: {first['instance_id']}"
            )
        await _stop(selected_process)
        print(
            "[affinity-live] STOPPED_BOUND_REPLICA "
            f"instance={first['instance_id']}",
            flush=True,
        )

        # Strict affinity must fail rather than hand the existing session to
        # the remaining replica.
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{gateway_url}/affinity/kv/get",
                json={
                    "service": "affinity-kv",
                    "affinity_id": first["affinity_id"],
                    "key": "key-0",
                },
            ) as response:
                strict_failure = await response.json()
                if response.status != 503:
                    raise RuntimeError(
                        "strict session unexpectedly moved replicas: "
                        f"HTTP {response.status}: {strict_failure}"
                    )
        print(
            "[affinity-live] STRICT_FAILURE_CONFIRMED "
            f"instance={first['instance_id']} response={strict_failure}",
            flush=True,
        )

        # A new, unbound session is free to use the surviving replica.
        print("[affinity-live] SESSION_2_BEGIN", flush=True)
        second = await run_probe(gateway_url, "affinity-kv", count)
        if second["instance_id"] == first["instance_id"]:
            raise RuntimeError("second session did not use the surviving replica")
        print(
            "[affinity-live] SESSION_2_PINNED "
            f"instance={second['instance_id']} affinity_id={second['affinity_id']}",
            flush=True,
        )
        print(
            "[affinity-live] FINAL_DISTRIBUTION "
            f"sessions={distribution['sessions']} "
            f"instances={json.dumps(distribution['instances'], sort_keys=True)}",
            flush=True,
        )
        print(
            "[affinity-live] PASS real gateway routed independent sessions "
            f"first={first['instance_id']} second={second['instance_id']} "
            "and preserved strict failure semantics",
            flush=True,
        )
        return {
            "status": "passed",
            "distribution": distribution,
            "first": first,
            "second": second,
        }
    finally:
        for process in reversed(processes):
            await _stop(process)
        if temporary_registry is not None:
            temporary_registry.cleanup()


def main(
    registry: Optional[str] = None,
    gateway_port: int = 18080,
    replica_a_port: int = 18091,
    replica_b_port: int = 18092,
    sessions: int = 20,
    count: int = 1,
    concurrency: int = 10,
) -> None:
    """Start the live environment, run the affinity probe, then clean it up."""
    asyncio.run(
        run_live_demo(
            registry=registry,
            gateway_port=gateway_port,
            replica_a_port=replica_a_port,
            replica_b_port=replica_b_port,
            sessions=sessions,
            count=count,
            concurrency=concurrency,
        )
    )


if __name__ == "__main__":
    fire.Fire(main)
