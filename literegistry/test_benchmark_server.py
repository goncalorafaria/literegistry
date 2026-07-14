"""Continuously load-test a code server and report request failures.

The benchmark is deliberately client-side only: it can measure availability,
latency, and response errors, but cannot determine remote host memory use
without server-side metrics or host access.
"""

import asyncio
import json
import sys
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Running ``python literegistry/test_benchmark_server.py`` places the package
# directory first on sys.path.  Without removing it, ``aiohttp`` resolves its
# stdlib ``http`` import to ``literegistry/http.py`` instead.
if __package__ in (None, ""):
    package_dir = str(Path(__file__).resolve().parent)
    if sys.path and sys.path[0] == package_dir:
        sys.path.pop(0)

import aiohttp


DEFAULT_URL = "http://g3115.hyak.local:8085/python"
DEFAULT_CODE = """\
import json, textwrap, re, math, itertools, collections

c = context
print("Length:", len(c))
print("First 500 chars:", c[:500])
"""
DEFAULT_CONTEXT = (
    "# Benchmark context\n\n"
    "This is representative text supplied to the code executor. " * 256
)


@dataclass
class WindowStats:
    started_at: float = field(default_factory=time.monotonic)
    completed: int = 0
    successes: int = 0
    failures: int = 0
    latencies: list[float] = field(default_factory=list)
    errors: Counter[str] = field(default_factory=Counter)
    examples: deque[dict[str, Any]] = field(default_factory=lambda: deque(maxlen=10))

    def add(
        self,
        *,
        latency: float,
        success: bool,
        error: str | None = None,
        status: int | None = None,
        detail: str | None = None,
    ) -> None:
        self.completed += 1
        self.latencies.append(latency)
        if success:
            self.successes += 1
            return

        self.failures += 1
        label = error or f"http_{status}" if status else error or "unknown"
        self.errors[label] += 1
        self.examples.append(
            {
                "error": label,
                "status": status,
                "latency_s": round(latency, 3),
                "detail": (detail or "")[:500],
            }
        )

    def report(self, *, url: str, concurrency: int, payload_bytes: int) -> dict[str, Any]:
        elapsed = max(time.monotonic() - self.started_at, 1e-9)
        sorted_latencies = sorted(self.latencies)

        def percentile(percent: int) -> float | None:
            if not sorted_latencies:
                return None
            index = round((len(sorted_latencies) - 1) * percent / 100)
            return round(sorted_latencies[index], 3)

        return {
            "url": url,
            "concurrency": concurrency,
            "payload_bytes": payload_bytes,
            "interval_s": round(elapsed, 3),
            "completed": self.completed,
            "successes": self.successes,
            "failures": self.failures,
            "success_rps": round(self.successes / elapsed, 3),
            "completed_rps": round(self.completed / elapsed, 3),
            "p50_latency_s": percentile(50),
            "p95_latency_s": percentile(95),
            "p99_latency_s": percentile(99),
            "max_latency_s": round(max(sorted_latencies), 3) if sorted_latencies else None,
            "errors": dict(self.errors),
            "failure_examples": list(self.examples),
        }


async def _request(
    session: aiohttp.ClientSession, url: str, payload: dict[str, Any], stats: WindowStats, lock: asyncio.Lock
) -> None:
    started = time.monotonic()
    try:
        async with session.post(url, json=payload) as response:
            text = await response.text()
            latency = time.monotonic() - started
            try:
                body = json.loads(text)
            except json.JSONDecodeError:
                body = None

            success = response.status == 200 and (
                not isinstance(body, dict) or body.get("success", True)
            )
            error = None
            if not success:
                error = (
                    f"server_{body.get('error', 'execution_failed')}"
                    if isinstance(body, dict) and body.get("error")
                    else f"http_{response.status}"
                )
            async with lock:
                stats.add(
                    latency=latency,
                    success=success,
                    error=error,
                    status=response.status,
                    detail=text,
                )
    except asyncio.TimeoutError:
        async with lock:
            stats.add(latency=time.monotonic() - started, success=False, error="client_timeout")
    except aiohttp.ClientError as exc:
        async with lock:
            stats.add(
                latency=time.monotonic() - started,
                success=False,
                error=type(exc).__name__,
                detail=str(exc),
            )


async def _run(
    *,
    url: str,
    payload: dict[str, Any],
    concurrency: int,
    requests: int,
    duration: float,
    timeout: float,
    report_interval: float,
) -> None:
    payload_bytes = len(json.dumps(payload).encode())
    stats = WindowStats()
    lock = asyncio.Lock()
    stop_at = time.monotonic() + duration if duration > 0 else None
    issued = 0
    issued_lock = asyncio.Lock()

    async def worker() -> None:
        nonlocal issued
        while True:
            async with issued_lock:
                if (requests and issued >= requests) or (
                    stop_at is not None and time.monotonic() >= stop_at
                ):
                    return
                issued += 1
            await _request(session, url, payload, stats, lock)

    timeout_config = aiohttp.ClientTimeout(total=timeout)
    connector = aiohttp.TCPConnector(limit=concurrency, limit_per_host=concurrency)
    print(
        json.dumps(
            {
                "event": "benchmark_started",
                "url": url,
                "concurrency": concurrency,
                "requests": requests or "unbounded",
                "duration_s": duration or "unbounded",
                "payload_bytes": payload_bytes,
            }
        ),
        flush=True,
    )
    async with aiohttp.ClientSession(timeout=timeout_config, connector=connector) as session:
        workers = [asyncio.create_task(worker()) for _ in range(concurrency)]
        while not all(worker.done() for worker in workers):
            await asyncio.sleep(report_interval)
            async with lock:
                print(
                    json.dumps(
                        {"event": "benchmark_interval", **stats.report(
                            url=url, concurrency=concurrency, payload_bytes=payload_bytes
                        )}
                    ),
                    flush=True,
                )
                stats = WindowStats()
        await asyncio.gather(*workers)

    async with lock:
        print(
            json.dumps(
                {"event": "benchmark_finished", **stats.report(
                    url=url, concurrency=concurrency, payload_bytes=payload_bytes
                )}
            ),
            flush=True,
        )


def main(
    url: str = DEFAULT_URL,
    concurrency: int = 8,
    requests: int = 0,
    duration: float = 0,
    timeout: float = 30,
    report_interval: float = 10,
    context_file: str | None = None,
    context: str | None = None,
    code: str = DEFAULT_CODE,
    max_runtime: float = 2,
) -> None:
    """Run a continuous code-server benchmark; Ctrl-C prints the latest report."""
    if concurrency < 1:
        raise ValueError("concurrency must be at least 1")
    if requests < 0 or duration < 0:
        raise ValueError("requests and duration must be non-negative")
    if timeout <= 0 or report_interval <= 0:
        raise ValueError("timeout and report_interval must be positive")
    if context_file and context is not None:
        raise ValueError("set either context_file or context, not both")

    context_payload = Path(context_file).read_text() if context_file else context or DEFAULT_CONTEXT
    payload = {
        "code": code,
        "context_payload": context_payload,
        "max_runtime": max_runtime,
        "model": "python",
    }
    try:
        asyncio.run(
            _run(
                url=url,
                payload=payload,
                concurrency=concurrency,
                requests=requests,
                duration=duration,
                timeout=timeout,
                report_interval=report_interval,
            )
        )
    except KeyboardInterrupt:
        print(json.dumps({"event": "benchmark_interrupted"}), flush=True)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
