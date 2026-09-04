"""Replay real TMAX command trajectories through the async Podman client."""

from __future__ import annotations

import asyncio
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import statistics
import sys
import time
from typing import Any

import aiohttp
import fire
from tqdm import tqdm

from literegistry_podman_client import PodmanClient

from .warm_podman import _normalize_gateway_url, _wait_for_podman


def _qualified_image(image: str) -> str:
    image = image.strip()
    if not image:
        raise ValueError("container_image must be non-empty")
    if "/" not in image:
        return f"docker.io/library/{image}"
    first = image.split("/", 1)[0]
    if first == "localhost" or any(marker in first for marker in (".", ":")):
        return image
    return f"docker.io/{image}"


def _load_workloads(path: Path) -> list[dict[str, Any]]:
    workloads: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            task_id = row.get("task_id")
            image = row.get("container_image")
            commands = row.get("commands")
            if not isinstance(task_id, str) or not task_id:
                raise ValueError(f"line {line_number} has no task_id")
            if not isinstance(image, str) or not image:
                raise ValueError(f"line {line_number} has no container_image")
            if not isinstance(commands, list) or not commands:
                raise ValueError(f"line {line_number} has no commands")
            if not all(isinstance(command, str) and command.strip() for command in commands):
                raise ValueError(f"line {line_number} has an invalid command")
            workloads.append(row)
    if not workloads:
        raise ValueError(f"manifest is empty: {path}")
    return workloads


def _load_checkpoint(path: Path | None) -> set[str]:
    if path is None or not path.exists():
        return set()
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int((len(ordered) - 1) * percentile))
    return ordered[index]


async def _replay_one(
    client: PodmanClient,
    workload: dict[str, Any],
    *,
    command_timeout: float,
    retries: int,
    on_session_open: Any,
    on_session_close: Any,
) -> dict[str, Any]:
    task_id = workload["task_id"]
    image = _qualified_image(workload["container_image"])
    commands: list[str] = workload["commands"]
    last_error = "unknown error"

    for attempt in range(1, retries + 1):
        session = None
        started = time.perf_counter()
        startup_seconds = 0.0
        close_seconds = 0.0
        commands_completed = 0
        nonzero_commands = 0
        timed_out_commands = 0
        truncated_commands = 0
        command_seconds: list[float] = []
        try:
            client_id = "tmax-live-" + hashlib.sha256(
                task_id.encode("utf-8")
            ).hexdigest()[:24]
            session = await client.handshake(image=image, client_id=client_id)
            startup_seconds = time.perf_counter() - started
            on_session_open()

            for command in commands:
                command_started = time.perf_counter()
                result = await session.execute(
                    command,
                    timeout=command_timeout,
                    check=False,
                )
                command_seconds.append(time.perf_counter() - command_started)
                commands_completed += 1
                nonzero_commands += int(not result.success)
                timed_out_commands += int(result.timed_out)
                truncated_commands += int(
                    result.stdout_truncated or result.stderr_truncated
                )

            instance_id = session.instance_id
            close_started = time.perf_counter()
            close_result = await session.close()
            close_seconds = time.perf_counter() - close_started
            session = None
            on_session_close()
            if not isinstance(close_result, dict) or close_result.get("removed") is not True:
                raise RuntimeError(f"close did not confirm removal: {close_result!r}")
            return {
                "task_id": task_id,
                "success": True,
                "attempts": attempt,
                "instance_id": instance_id,
                "startup_seconds": startup_seconds,
                "close_seconds": close_seconds,
                "command_seconds": command_seconds,
                "commands_completed": commands_completed,
                "nonzero_commands": nonzero_commands,
                "timed_out_commands": timed_out_commands,
                "truncated_commands": truncated_commands,
                "elapsed_seconds": time.perf_counter() - started,
            }
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        finally:
            if session is not None:
                try:
                    await session.close()
                except Exception as close_exc:
                    last_error += f"; close {type(close_exc).__name__}: {close_exc}"
                finally:
                    on_session_close()
        if attempt < retries:
            await asyncio.sleep(min(30.0, float(2 ** (attempt - 1))))

    return {
        "task_id": task_id,
        "success": False,
        "attempts": retries,
        "error": last_error,
        "startup_seconds": startup_seconds,
        "close_seconds": close_seconds,
        "command_seconds": command_seconds,
        "commands_completed": commands_completed,
        "nonzero_commands": nonzero_commands,
        "timed_out_commands": timed_out_commands,
        "truncated_commands": truncated_commands,
        "elapsed_seconds": time.perf_counter() - started,
    }


async def live_fire(
    gateway_url: str,
    manifest_path: str,
    *,
    concurrency: int = 1000,
    total: int = 4000,
    expected_podman: int = 32,
    wait_timeout: float = 600,
    command_timeout: float = 60,
    retries: int = 3,
    shuffle_seed: int = 0,
    checkpoint_file: str | None = None,
    log_every: int = 100,
    connection_limit: int | None = None,
) -> dict[str, Any]:
    """Run a rolling-window replay of TMAX workloads through ``PodmanClient``."""
    gateway_url = _normalize_gateway_url(gateway_url)
    if concurrency < 1:
        raise ValueError("concurrency must be at least one")
    if total < 1:
        raise ValueError("total must be at least one")
    if command_timeout <= 0:
        raise ValueError("command_timeout must be positive")
    if retries < 1:
        raise ValueError("retries must be at least one")
    if log_every < 1:
        raise ValueError("log_every must be at least one")
    if connection_limit is None:
        connection_limit = concurrency
    if connection_limit < 1:
        raise ValueError("connection_limit must be at least one")

    podman_count = await _wait_for_podman(
        gateway_url, expected_podman, wait_timeout
    )
    workloads = _load_workloads(Path(manifest_path))
    random.Random(shuffle_seed).shuffle(workloads)
    selected = workloads[:total]
    if len(selected) < total:
        raise ValueError(
            f"requested {total} workloads but manifest contains {len(workloads)}"
        )

    checkpoint_path = Path(checkpoint_file) if checkpoint_file else None
    completed_before = _load_checkpoint(checkpoint_path)
    pending = [row for row in selected if row["task_id"] not in completed_before]
    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        "TMAX_LIVE_FIRE_START "
        f"selected={len(selected)} resumed={len(selected) - len(pending)} "
        f"pending={len(pending)} concurrency={concurrency} "
        f"podman_replicas={podman_count} gateway={gateway_url}",
        flush=True,
    )

    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    for workload in pending:
        queue.put_nowait(workload)

    live_sessions = 0
    peak_live_sessions = 0
    successes = 0
    failures: list[dict[str, Any]] = []
    startup_seconds: list[float] = []
    close_seconds: list[float] = []
    command_seconds: list[float] = []
    commands_completed = 0
    nonzero_commands = 0
    timed_out_commands = 0
    truncated_commands = 0
    instance_counts: Counter[str] = Counter()
    checkpoint_lock = asyncio.Lock()
    stats_lock = asyncio.Lock()
    started = time.perf_counter()

    def session_opened() -> None:
        nonlocal live_sessions, peak_live_sessions
        live_sessions += 1
        peak_live_sessions = max(peak_live_sessions, live_sessions)

    def session_closed() -> None:
        nonlocal live_sessions
        live_sessions -= 1

    request_timeout = max(310.0, command_timeout + 30.0)
    http_session = aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(limit=connection_limit),
        timeout=aiohttp.ClientTimeout(total=request_timeout),
        trust_env=False,
    )
    client = PodmanClient(
        gateway_url,
        workdir="/home/user",
        request_timeout=request_timeout,
        http_session=http_session,
    )
    await client.open()
    progress = tqdm(
        total=len(pending),
        unit="trajectory",
        desc="TMAX live fire",
        file=sys.stdout,
        disable=False,
        mininterval=2.0,
        smoothing=0.1,
        dynamic_ncols=False,
    )

    async def worker() -> None:
        nonlocal successes, commands_completed, nonzero_commands
        nonlocal timed_out_commands, truncated_commands
        while True:
            try:
                workload = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            result = await _replay_one(
                client,
                workload,
                command_timeout=command_timeout,
                retries=retries,
                on_session_open=session_opened,
                on_session_close=session_closed,
            )
            async with stats_lock:
                startup_seconds.append(float(result["startup_seconds"]))
                close_seconds.append(float(result["close_seconds"]))
                command_seconds.extend(result["command_seconds"])
                commands_completed += int(result["commands_completed"])
                nonzero_commands += int(result["nonzero_commands"])
                timed_out_commands += int(result["timed_out_commands"])
                truncated_commands += int(result["truncated_commands"])
                if result["success"]:
                    successes += 1
                    instance_id = result.get("instance_id")
                    if isinstance(instance_id, str):
                        instance_counts[instance_id] += 1
                    if checkpoint_path is not None:
                        async with checkpoint_lock:
                            with checkpoint_path.open("a", encoding="utf-8") as stream:
                                stream.write(result["task_id"] + "\n")
                                stream.flush()
                else:
                    failures.append(
                        {
                            "task_id": result["task_id"],
                            "error": result.get("error"),
                            "attempts": result["attempts"],
                        }
                    )
                progress.update(1)
                progress.set_postfix(
                    live=live_sessions,
                    peak=peak_live_sessions,
                    ok=successes,
                    failed=len(failures),
                    commands=commands_completed,
                    refresh=False,
                )
                done = successes + len(failures)
                if done % log_every == 0 or done == len(pending):
                    elapsed = time.perf_counter() - started
                    print(
                        "TMAX_LIVE_FIRE_PROGRESS "
                        f"completed={done} total={len(pending)} "
                        f"succeeded={successes} failed={len(failures)} "
                        f"live_sessions={live_sessions} peak_live_sessions={peak_live_sessions} "
                        f"commands={commands_completed} "
                        f"trajectories_per_second={done / elapsed if elapsed else 0:.3f} "
                        f"commands_per_second={commands_completed / elapsed if elapsed else 0:.3f}",
                        flush=True,
                    )
            queue.task_done()

    try:
        workers = [
            asyncio.create_task(worker())
            for _ in range(min(concurrency, len(pending)))
        ]
        await asyncio.gather(*workers)
    finally:
        progress.close()
        await client.aclose()
        await http_session.close()

    elapsed = time.perf_counter() - started
    result = {
        "gateway_url": gateway_url,
        "manifest_path": manifest_path,
        "podman_replicas": podman_count,
        "selected": len(selected),
        "resumed": len(selected) - len(pending),
        "attempted": len(pending),
        "successes": successes,
        "failures": len(failures),
        "commands_completed": commands_completed,
        "nonzero_commands": nonzero_commands,
        "timed_out_commands": timed_out_commands,
        "truncated_commands": truncated_commands,
        "concurrency": concurrency,
        "connection_limit": connection_limit,
        "peak_live_sessions": peak_live_sessions,
        "elapsed_seconds": elapsed,
        "trajectories_per_second": len(pending) / elapsed if elapsed else 0.0,
        "commands_per_second": commands_completed / elapsed if elapsed else 0.0,
        "startup_seconds_mean": statistics.fmean(startup_seconds) if startup_seconds else 0.0,
        "startup_seconds_p50": _percentile(startup_seconds, 0.50),
        "startup_seconds_p95": _percentile(startup_seconds, 0.95),
        "command_seconds_mean": statistics.fmean(command_seconds) if command_seconds else 0.0,
        "command_seconds_p50": _percentile(command_seconds, 0.50),
        "command_seconds_p95": _percentile(command_seconds, 0.95),
        "close_seconds_mean": statistics.fmean(close_seconds) if close_seconds else 0.0,
        "instance_counts": dict(sorted(instance_counts.items())),
        "checkpoint_file": str(checkpoint_path) if checkpoint_path else None,
        "failure_details": failures[:100],
    }
    print("TMAX_LIVE_FIRE_COMPLETE " + json.dumps(result, sort_keys=True), flush=True)
    return result


def run(
    gateway_url: str,
    manifest_path: str,
    concurrency: int = 1000,
    total: int = 4000,
    expected_podman: int = 32,
    wait_timeout: float = 600,
    command_timeout: float = 60,
    retries: int = 3,
    shuffle_seed: int = 0,
    checkpoint_file: str | None = None,
    log_every: int = 100,
    connection_limit: int | None = None,
) -> None:
    """Run the async TMAX live-fire workload with a rolling concurrency window."""
    result = asyncio.run(
        live_fire(
            gateway_url,
            manifest_path,
            concurrency=concurrency,
            total=total,
            expected_podman=expected_podman,
            wait_timeout=wait_timeout,
            command_timeout=command_timeout,
            retries=retries,
            shuffle_seed=shuffle_seed,
            checkpoint_file=checkpoint_file,
            log_every=log_every,
            connection_limit=connection_limit,
        )
    )
    if result["failures"]:
        raise SystemExit(1)


def main() -> None:
    fire.Fire(run)


if __name__ == "__main__":
    main()
