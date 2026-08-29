"""Throughput benchmark for gateway-backed Podman execution sessions."""

from __future__ import annotations

import asyncio
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
import math
from pathlib import Path
import shlex
import sys
import time
from typing import Any
from uuid import uuid4


_CLIENT_SOURCE = (
    Path(__file__).resolve().parents[1]
    / "literegistry_podman_client"
    / "src"
)
if _CLIENT_SOURCE.is_dir() and str(_CLIENT_SOURCE) not in sys.path:
    sys.path.insert(0, str(_CLIENT_SOURCE))

from literegistry_podman_client import PodmanClient, PodmanGatewayError


PODMAN_SESSION_IMAGE = "docker.io/library/ubuntu:24.04"
_RETRYABLE_HTTP_STATUSES = frozenset({429, 502, 503})


class PodmanExecutionClient:
    """Benchmark adapter over the standalone LiteRegistry Podman client.

    The adapter retains the benchmark's original start/execute/close surface
    while keeping all transport and session ownership in LiteRegistry.
    """

    def __init__(
        self,
        gateway_url: str,
        *,
        image: str = PODMAN_SESSION_IMAGE,
        client_id: str | None = None,
        service: str = "podman",
        timeout: float = 70,
        handshake_timeout: float = 300,
        max_retries: int = 3,
        workdir: str = "/workspace",
        http_session: Any | None = None,
    ) -> None:
        if max_retries < 1:
            raise ValueError("max_retries must be at least 1")
        self.image = image
        self.client_id = client_id
        self.timeout = timeout
        self.handshake_timeout = handshake_timeout
        self.max_retries = max_retries
        self._owns_http_session = http_session is None
        self._client = PodmanClient(
            gateway_url,
            service=service,
            workdir=workdir,
            request_timeout=max(timeout, handshake_timeout),
            http_session=http_session,
        )
        self._session: Any | None = None

    @property
    def container_id(self) -> str | None:
        return self._session.container_id if self._session is not None else None

    @property
    def started(self) -> bool:
        return self._session is not None and not self._session.closed

    async def _request_with_retries(self, operation: Any, timeout: float) -> Any:
        delay = 0.5
        for attempt in range(self.max_retries):
            try:
                return await asyncio.wait_for(operation(), timeout=timeout)
            except (PodmanGatewayError, asyncio.TimeoutError) as error:
                status = getattr(error, "status_code", None)
                retryable = status is None or status in _RETRYABLE_HTTP_STATUSES
                if not retryable or attempt == self.max_retries - 1:
                    raise
                await asyncio.sleep(delay)
                delay = min(delay * 2.0, 8.0)
        raise RuntimeError("unreachable")

    async def start(self) -> dict[str, Any]:
        if self.started:
            raise RuntimeError("Podman session is already started")
        self._session = await self._request_with_retries(
            lambda: self._client.handshake(
                image=self.image,
                client_id=self.client_id,
            ),
            self.handshake_timeout,
        )
        return {
            "affinity_id": self._session.affinity_id,
            "container_id": self._session.container_id,
            "instance_id": self._session.instance_id,
            "image": self._session.image,
        }

    async def execute(
        self,
        *,
        command: str,
        stdin: str = "",
        timeout: float = 10,
    ) -> dict[str, Any]:
        if not self.started:
            raise RuntimeError("Podman session is not started; call start() first")
        assert self._session is not None
        result = await self._request_with_retries(
            lambda: self._session.execute(
                command,
                stdin=stdin,
                timeout=timeout,
            ),
            self.timeout,
        )
        return asdict(result)

    async def close(self) -> dict[str, Any] | None:
        if not self.started:
            return None
        assert self._session is not None
        try:
            return await self._request_with_retries(
                self._session.close,
                self.timeout,
            )
        finally:
            if self._owns_http_session:
                await self._client.aclose()


@dataclass(frozen=True)
class PodmanBenchmarkConfig:
    """One workload sweep against a fixed-size Podman replica pool."""

    gateway_url: str
    replicas: int
    concurrency: tuple[int, ...] = (1, 2, 4, 8, 16, 32)
    sessions_per_worker: int = 2
    waves: int = 1
    total_sessions: int | None = None
    commands_per_session: int = 4
    image: str = PODMAN_SESSION_IMAGE
    command_timeout: float = 10
    request_timeout: float = 70
    handshake_timeout: float = 300
    max_retries: int = 3
    workdir: str = "/workspace"
    client_id_prefix: str = "literegistry-podman-benchmark"
    warmup_all_replicas: bool = False
    warmup_concurrency: int = 32
    warmup_max_sessions: int | None = None

    def validate(self) -> PodmanBenchmarkConfig:
        if not self.gateway_url.strip():
            raise ValueError("gateway_url must be non-empty")
        if self.replicas < 1:
            raise ValueError("replicas must be positive")
        if not self.concurrency:
            raise ValueError("concurrency must contain at least one level")
        if any(level < 1 for level in self.concurrency):
            raise ValueError("concurrency levels must be positive")
        if len(set(self.concurrency)) != len(self.concurrency):
            raise ValueError("concurrency levels must be unique")
        if tuple(sorted(self.concurrency)) != self.concurrency:
            raise ValueError("concurrency levels must be increasing")
        if self.sessions_per_worker < 1:
            raise ValueError("sessions_per_worker must be positive")
        if self.waves < 1:
            raise ValueError("waves must be positive")
        if self.total_sessions is not None:
            if self.total_sessions < 1:
                raise ValueError("total_sessions must be positive")
            if self.waves != 1:
                raise ValueError(
                    "waves cannot be combined with fixed total_sessions"
                )
        if self.commands_per_session < 2:
            raise ValueError(
                "commands_per_session must be at least 2 to verify write/read affinity"
            )
        if not self.image.strip():
            raise ValueError("image must be non-empty")
        if self.command_timeout <= 0:
            raise ValueError("command_timeout must be positive")
        if self.request_timeout <= 0 or self.handshake_timeout <= 0:
            raise ValueError("HTTP timeouts must be positive")
        if self.max_retries < 1:
            raise ValueError("max_retries must be at least 1")
        if not self.workdir.strip():
            raise ValueError("workdir must be non-empty")
        if not self.client_id_prefix.strip():
            raise ValueError("client_id_prefix must be non-empty")
        if self.warmup_concurrency < 1:
            raise ValueError("warmup_concurrency must be positive")
        if self.warmup_max_sessions is not None:
            if self.warmup_max_sessions < self.replicas:
                raise ValueError(
                    "warmup_max_sessions must be at least the replica count"
                )
        return self


@dataclass
class _SessionSample:
    client: PodmanExecutionClient
    success: bool = False
    container_id: str | None = None
    instance_id: str | None = None
    handshake_seconds: float | None = None
    command_seconds: list[float] = field(default_factory=list)
    write_seconds: list[float] = field(default_factory=list)
    read_seconds: list[float] = field(default_factory=list)
    completed_commands: int = 0
    closed: bool = False
    close_seconds: float | None = None
    trajectory_seconds: float | None = None
    error: str | None = None


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _latency_summary(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {
            "count": 0,
            "mean_ms": 0.0,
            "p50_ms": 0.0,
            "p95_ms": 0.0,
            "p99_ms": 0.0,
            "max_ms": 0.0,
        }
    return {
        "count": len(values),
        "mean_ms": round(sum(values) * 1000 / len(values), 3),
        "p50_ms": round(_percentile(values, 0.50) * 1000, 3),
        "p95_ms": round(_percentile(values, 0.95) * 1000, 3),
        "p99_ms": round(_percentile(values, 0.99) * 1000, 3),
        "max_ms": round(max(values) * 1000, 3),
    }


def _error_text(error: BaseException) -> str:
    detail = str(error).replace("\n", " ").strip()
    return f"{type(error).__name__}: {detail}"[:500]


def _new_session(
    config: PodmanBenchmarkConfig,
    *,
    concurrency: int,
    wave_index: int,
    session_index: int,
    http_session: Any | None = None,
) -> _SessionSample:
    client = PodmanExecutionClient(
        config.gateway_url,
        image=config.image,
        client_id=(
            f"{config.client_id_prefix}-r{config.replicas}"
            f"-c{concurrency}-w{wave_index}-s{session_index}"
        ),
        timeout=config.request_timeout,
        handshake_timeout=config.handshake_timeout,
        max_retries=config.max_retries,
        workdir=config.workdir,
        http_session=http_session,
    )
    return _SessionSample(client=client)


def _new_warmup_session(
    config: PodmanBenchmarkConfig,
    *,
    session_index: int,
    http_session: Any | None = None,
) -> _SessionSample:
    client = PodmanExecutionClient(
        config.gateway_url,
        image=config.image,
        client_id=(
            f"{config.client_id_prefix}-r{config.replicas}"
            f"-warmup-s{session_index}"
        ),
        timeout=config.request_timeout,
        handshake_timeout=config.handshake_timeout,
        max_retries=config.max_retries,
        workdir=config.workdir,
        http_session=http_session,
    )
    return _SessionSample(client=client)


async def _start_session(sample: _SessionSample) -> None:
    handshake_started = time.perf_counter()
    try:
        handshake = await sample.client.start()
        sample.container_id = sample.client.container_id
        instance_id = handshake.get("instance_id")
        if isinstance(instance_id, str) and instance_id:
            sample.instance_id = instance_id
    except Exception as error:
        sample.error = _error_text(error)
    finally:
        sample.handshake_seconds = time.perf_counter() - handshake_started


async def _run_session_commands(
    config: PodmanBenchmarkConfig,
    sample: _SessionSample,
) -> None:
    if not sample.client.started:
        return
    token = f"ai2-hello-{uuid4().hex}"
    commands = [
        (
            "write",
            (
                "printf '%s\\n' "
                f"{shlex.quote(token)} > "
                f"{shlex.quote(config.workdir + '/.literegistry-throughput')}"
            ),
        ),
        *[
            (
                "read",
                f"cat {shlex.quote(config.workdir + '/.literegistry-throughput')}",
            )
            for _ in range(config.commands_per_session - 1)
        ],
    ]
    try:
        for command_index, (kind, command) in enumerate(commands):
            command_started = time.perf_counter()
            try:
                result = await sample.client.execute(
                    command=command,
                    timeout=config.command_timeout,
                )
            finally:
                elapsed = time.perf_counter() - command_started
                sample.command_seconds.append(elapsed)
                if kind == "write":
                    sample.write_seconds.append(elapsed)
                else:
                    sample.read_seconds.append(elapsed)
            if result.get("success") is not True:
                raise RuntimeError(f"command {command_index} failed: {result}")
            if kind == "read" and result.get("stdout") != f"{token}\n":
                raise RuntimeError(
                    f"command {command_index} returned the wrong affinity data"
                )
            sample.completed_commands += 1
    except Exception as error:
        command_error = _error_text(error)
        sample.error = (
            f"{sample.error}; command: {command_error}"
            if sample.error
            else command_error
        )


async def _close_session(sample: _SessionSample) -> None:
    if not sample.client.started:
        return
    close_started = time.perf_counter()
    try:
        result = await sample.client.close()
        if not result or result.get("removed") is not True:
            raise RuntimeError(f"close did not remove the container: {result}")
        sample.closed = True
    except Exception as error:
        close_error = _error_text(error)
        sample.error = (
            f"{sample.error}; close: {close_error}"
            if sample.error
            else close_error
        )
    finally:
        sample.close_seconds = time.perf_counter() - close_started


async def _warmup_replica_image_caches(
    config: PodmanBenchmarkConfig,
    http_session: Any | None = None,
) -> dict[str, Any]:
    """Create and delete one untimed container on every replica.

    This makes image availability a benchmark precondition, so registry pulls
    and their retries cannot affect the measured startup latency or throughput.
    """

    max_sessions = config.warmup_max_sessions or config.replicas * 32
    samples: list[_SessionSample] = []
    warmed_instances: set[str] = set()
    next_index = 0
    started_at = datetime.now(UTC)
    started = time.perf_counter()

    async def worker() -> None:
        nonlocal next_index
        while (
            len(warmed_instances) < config.replicas
            and next_index < max_sessions
        ):
            session_index = next_index
            next_index += 1
            sample = _new_warmup_session(
                config,
                session_index=session_index,
                http_session=http_session,
            )
            samples.append(sample)
            await _start_session(sample)
            if sample.client.started:
                await _close_session(sample)
            if sample.instance_id and sample.closed:
                warmed_instances.add(sample.instance_id)
            # Do not let a worker accumulate containers after an unconfirmed
            # deletion. Other workers may continue warming healthy replicas.
            if sample.client.started:
                return

    await asyncio.gather(
        *(
            worker()
            for _ in range(min(config.warmup_concurrency, max_sessions))
        )
    )
    wall_seconds = time.perf_counter() - started
    errors = Counter(sample.error for sample in samples if sample.error)
    result = {
        "excluded_from_measurement": True,
        "image": config.image,
        "expected_replicas": config.replicas,
        "observed_replicas": len(warmed_instances),
        "all_replicas_warmed": len(warmed_instances) == config.replicas,
        "warmed_instances": sorted(warmed_instances),
        "attempted_sessions": len(samples),
        "successful_starts": sum(
            sample.container_id is not None for sample in samples
        ),
        "confirmed_deletes": sum(sample.closed for sample in samples),
        "wall_seconds": round(wall_seconds, 6),
        "started_at": started_at.isoformat(),
        "finished_at": datetime.now(UTC).isoformat(),
        "errors": [
            {"error": error, "count": count}
            for error, count in errors.most_common(10)
        ],
    }
    if not result["all_replicas_warmed"]:
        raise RuntimeError(
            "image warm-up did not reach every replica: "
            f"{result['observed_replicas']}/{config.replicas} after "
            f"{result['attempted_sessions']} attempts"
        )
    return result


async def _run_bounded(
    samples: list[_SessionSample],
    concurrency: int,
    operation: Any,
) -> None:
    next_index = 0

    async def worker() -> None:
        nonlocal next_index
        while next_index < len(samples):
            sample_index = next_index
            next_index += 1
            await operation(samples[sample_index])

    await asyncio.gather(
        *(worker() for _ in range(min(concurrency, len(samples))))
    )


def _distribution_summary(
    distribution: Counter[str],
    replicas: int,
) -> dict[str, Any]:
    counts = list(distribution.values())
    observed = len(counts)
    mean = sum(counts) / observed if observed else 0.0
    variance = (
        sum((count - mean) ** 2 for count in counts) / observed
        if observed
        else 0.0
    )
    return {
        "expected_replicas": replicas,
        "observed_replicas": observed,
        "all_replicas_received_traffic": observed == replicas,
        "min_per_observed_replica": min(counts, default=0),
        "max_per_observed_replica": max(counts, default=0),
        "mean_per_observed_replica": round(mean, 3),
        "coefficient_of_variation": round(
            math.sqrt(variance) / mean if mean else 0.0,
            6,
        ),
    }


async def _run_trajectory_level(
    config: PodmanBenchmarkConfig,
    concurrency: int,
    http_session: Any | None = None,
) -> dict[str, Any]:
    """Continuously run complete create/command/delete session trajectories."""

    assert config.total_sessions is not None
    samples: list[_SessionSample] = []
    next_index = 0
    total_started = time.perf_counter()

    async def worker() -> None:
        nonlocal next_index
        while next_index < config.total_sessions:
            session_index = next_index
            next_index += 1
            sample = _new_session(
                config,
                concurrency=concurrency,
                wave_index=0,
                session_index=session_index,
                http_session=http_session,
            )
            samples.append(sample)
            trajectory_started = time.perf_counter()
            await _start_session(sample)
            if sample.client.started:
                await _run_session_commands(config, sample)
                await _close_session(sample)
            sample.trajectory_seconds = time.perf_counter() - trajectory_started
            sample.success = (
                sample.error is None
                and sample.completed_commands == config.commands_per_session
                and sample.closed
            )
            # A worker may only take another trajectory after confirmed deletion.
            # If deletion failed, stop this worker instead of accumulating orphans.
            if sample.client.started:
                return

    await asyncio.gather(
        *(worker() for _ in range(min(concurrency, config.total_sessions)))
    )
    wall_seconds = time.perf_counter() - total_started

    successful = [sample for sample in samples if sample.success]
    successful_starts = sum(sample.container_id is not None for sample in samples)
    completed_commands = sum(sample.completed_commands for sample in samples)
    closed_sessions = sum(sample.closed for sample in samples)
    errors = Counter(sample.error for sample in samples if sample.error)
    session_instances = Counter(
        sample.instance_id for sample in samples if sample.instance_id
    )
    command_instances: Counter[str] = Counter()
    for sample in samples:
        if sample.instance_id:
            command_instances[sample.instance_id] += sample.completed_commands

    handshake_attempts = [
        sample.handshake_seconds
        for sample in samples
        if sample.handshake_seconds is not None
    ]
    handshake = [
        sample.handshake_seconds
        for sample in samples
        if sample.container_id is not None
        and sample.handshake_seconds is not None
    ]
    command = [
        latency for sample in samples for latency in sample.command_seconds
    ]
    write = [
        latency for sample in samples for latency in sample.write_seconds
    ]
    read = [
        latency for sample in samples for latency in sample.read_seconds
    ]
    close = [
        sample.close_seconds
        for sample in samples
        if sample.close_seconds is not None
    ]
    trajectory = [
        sample.trajectory_seconds
        for sample in samples
        if sample.trajectory_seconds is not None
    ]
    return {
        "replicas": config.replicas,
        "concurrency": concurrency,
        "execution_model": "rolling-complete-session-trajectories",
        "requested_sessions": config.total_sessions,
        "attempted_sessions": len(samples),
        "successful_sessions": len(successful),
        "failed_sessions": len(samples) - len(successful),
        "unattempted_sessions": config.total_sessions - len(samples),
        "requested_commands": config.total_sessions * config.commands_per_session,
        "completed_commands": completed_commands,
        "max_live_sessions": concurrency,
        "unique_container_ids": len(
            {sample.container_id for sample in samples if sample.container_id}
        ),
        "instance_distribution": dict(sorted(session_instances.items())),
        "command_instance_distribution": dict(sorted(command_instances.items())),
        "traffic": {
            "sessions": _distribution_summary(
                session_instances,
                config.replicas,
            ),
            "commands": _distribution_summary(
                command_instances,
                config.replicas,
            ),
        },
        "wall_seconds": round(wall_seconds, 6),
        "sessions_per_second": round(len(successful) / wall_seconds, 6),
        "commands_per_second": round(completed_commands / wall_seconds, 6),
        "sessions_per_second_per_replica": round(
            len(successful) / wall_seconds / config.replicas,
            6,
        ),
        "latency": {
            "handshake_attempt": _latency_summary(handshake_attempts),
            "handshake": _latency_summary(handshake),
            "command": _latency_summary(command),
            "write": _latency_summary(write),
            "read": _latency_summary(read),
            "close": _latency_summary(close),
            "trajectory": _latency_summary(trajectory),
        },
        "phases": {
            "startup": {
                "wall_seconds": round(wall_seconds, 6),
                "successful_sessions": successful_starts,
                "sessions_per_second": round(
                    successful_starts / wall_seconds,
                    6,
                ),
                "latency": _latency_summary(handshake),
                "attempt_latency": _latency_summary(handshake_attempts),
            },
            "commands": {
                "wall_seconds": round(wall_seconds, 6),
                "completed_commands": completed_commands,
                "commands_per_second": round(
                    completed_commands / wall_seconds,
                    6,
                ),
                "latency": _latency_summary(command),
                "write_latency": _latency_summary(write),
                "read_latency": _latency_summary(read),
            },
            "close": {
                "wall_seconds": round(wall_seconds, 6),
                "closed_sessions": closed_sessions,
                "sessions_per_second": round(
                    closed_sessions / wall_seconds,
                    6,
                ),
                "latency": _latency_summary(close),
            },
            "trajectory": {
                "wall_seconds": round(wall_seconds, 6),
                "successful_sessions": len(successful),
                "sessions_per_second": round(
                    len(successful) / wall_seconds,
                    6,
                ),
                "latency": _latency_summary(trajectory),
            },
        },
        "errors": [
            {"error": error, "count": count}
            for error, count in errors.most_common(10)
        ],
    }


async def _run_level(
    config: PodmanBenchmarkConfig,
    concurrency: int,
    http_session: Any | None = None,
) -> dict[str, Any]:
    if config.total_sessions is not None:
        return await _run_trajectory_level(
            config, concurrency, http_session=http_session
        )

    sessions_per_wave = concurrency * config.sessions_per_worker
    wave_sizes = [sessions_per_wave] * config.waves
    requested_sessions = sum(wave_sizes)
    samples: list[_SessionSample] = []
    wave_results: list[dict[str, Any]] = []
    total_started = time.perf_counter()
    startup_wall_seconds = 0.0
    command_wall_seconds = 0.0
    close_wall_seconds = 0.0
    successful_starts = 0

    for wave_index, wave_size in enumerate(wave_sizes):
        wave_samples = [
            _new_session(
                config,
                concurrency=concurrency,
                wave_index=wave_index,
                session_index=session_index,
                http_session=http_session,
            )
            for session_index in range(wave_size)
        ]

        startup_started = time.perf_counter()
        await _run_bounded(wave_samples, concurrency, _start_session)
        wave_startup_seconds = time.perf_counter() - startup_started
        startup_wall_seconds += wave_startup_seconds

        command_samples = [
            sample for sample in wave_samples if sample.client.started
        ]
        successful_starts += len(command_samples)
        command_started = time.perf_counter()
        await _run_bounded(
            command_samples,
            concurrency,
            lambda sample: _run_session_commands(config, sample),
        )
        wave_command_seconds = time.perf_counter() - command_started
        command_wall_seconds += wave_command_seconds

        close_started = time.perf_counter()
        await _run_bounded(command_samples, concurrency, _close_session)
        wave_close_seconds = time.perf_counter() - close_started
        close_wall_seconds += wave_close_seconds

        for sample in wave_samples:
            sample.success = (
                sample.error is None
                and sample.completed_commands == config.commands_per_session
                and sample.closed
            )
        wave_completed_commands = sum(
            sample.completed_commands for sample in wave_samples
        )
        wave_successful = sum(sample.success for sample in wave_samples)
        wave_results.append(
            {
                "wave": wave_index + 1,
                "requested_sessions": wave_size,
                "successful_sessions": wave_successful,
                "failed_sessions": wave_size - wave_successful,
                "completed_commands": wave_completed_commands,
                "phases": {
                    "startup": {
                        "wall_seconds": round(wave_startup_seconds, 6),
                        "successful_sessions": len(command_samples),
                        "sessions_per_second": round(
                            len(command_samples) / wave_startup_seconds,
                            6,
                        ),
                    },
                    "commands": {
                        "wall_seconds": round(wave_command_seconds, 6),
                        "completed_commands": wave_completed_commands,
                        "commands_per_second": round(
                            wave_completed_commands / wave_command_seconds,
                            6,
                        ),
                    },
                    "close": {
                        "wall_seconds": round(wave_close_seconds, 6),
                        "closed_sessions": sum(
                            sample.closed
                            for sample in wave_samples
                        ),
                    },
                },
            }
        )
        samples.extend(wave_samples)

    wall_seconds = time.perf_counter() - total_started

    successful = [sample for sample in samples if sample.success]
    completed_commands = sum(sample.completed_commands for sample in samples)
    errors = Counter(sample.error for sample in samples if sample.error)
    session_instances = Counter(
        sample.instance_id for sample in samples if sample.instance_id
    )
    command_instances: Counter[str] = Counter()
    for sample in samples:
        if sample.instance_id:
            command_instances[sample.instance_id] += sample.completed_commands

    handshake = [
        sample.handshake_seconds
        for sample in samples
        if sample.container_id is not None
        and sample.handshake_seconds is not None
    ]
    command = [
        latency
        for sample in samples
        for latency in sample.command_seconds
    ]
    write = [
        latency for sample in samples for latency in sample.write_seconds
    ]
    read = [
        latency for sample in samples for latency in sample.read_seconds
    ]
    close = [
        sample.close_seconds
        for sample in samples
        if sample.close_seconds is not None
    ]
    return {
        "replicas": config.replicas,
        "concurrency": concurrency,
        "waves": len(wave_sizes),
        "sessions_per_wave": sessions_per_wave,
        "max_live_sessions": max(wave_sizes),
        "requested_sessions": requested_sessions,
        "successful_sessions": len(successful),
        "failed_sessions": len(samples) - len(successful),
        "requested_commands": requested_sessions * config.commands_per_session,
        "completed_commands": completed_commands,
        "unique_container_ids": len(
            {sample.container_id for sample in samples if sample.container_id}
        ),
        "instance_distribution": dict(sorted(session_instances.items())),
        "command_instance_distribution": dict(sorted(command_instances.items())),
        "traffic": {
            "sessions": _distribution_summary(
                session_instances,
                config.replicas,
            ),
            "commands": _distribution_summary(
                command_instances,
                config.replicas,
            ),
        },
        "wave_results": wave_results,
        "wall_seconds": round(wall_seconds, 6),
        "sessions_per_second": round(len(successful) / wall_seconds, 6),
        "commands_per_second": round(completed_commands / wall_seconds, 6),
        "sessions_per_second_per_replica": round(
            len(successful) / wall_seconds / config.replicas,
            6,
        ),
        "latency": {
            "handshake": _latency_summary(handshake),
            "command": _latency_summary(command),
            "write": _latency_summary(write),
            "read": _latency_summary(read),
            "close": _latency_summary(close),
        },
        "phases": {
            "startup": {
                "wall_seconds": round(startup_wall_seconds, 6),
                "successful_sessions": successful_starts,
                "sessions_per_second": round(
                    successful_starts / startup_wall_seconds,
                    6,
                ),
                "latency": _latency_summary(handshake),
            },
            "commands": {
                "wall_seconds": round(command_wall_seconds, 6),
                "completed_commands": completed_commands,
                "commands_per_second": round(
                    completed_commands / command_wall_seconds,
                    6,
                ),
                "latency": _latency_summary(command),
                "write_latency": _latency_summary(write),
                "read_latency": _latency_summary(read),
            },
            "close": {
                "wall_seconds": round(close_wall_seconds, 6),
                "closed_sessions": sum(sample.closed for sample in samples),
                "latency": _latency_summary(close),
            },
        },
        "errors": [
            {"error": error, "count": count}
            for error, count in errors.most_common(10)
        ],
    }


async def run_podman_benchmark(
    config: PodmanBenchmarkConfig,
) -> dict[str, Any]:
    """Run a Podman workload against one fixed replica pool."""

    import aiohttp

    config = config.validate()
    connector = aiohttp.TCPConnector(
        limit=0,
        keepalive_timeout=120,
        ttl_dns_cache=300,
        enable_cleanup_closed=True,
    )
    async with aiohttp.ClientSession(
        connector=connector,
        trust_env=False,
    ) as http_session:
        warmup = (
            await _warmup_replica_image_caches(
                config, http_session=http_session
            )
            if config.warmup_all_replicas
            else None
        )
        started_at = datetime.now(UTC)
        levels = [
            await _run_level(
                config, concurrency, http_session=http_session
            )
            for concurrency in config.concurrency
        ]
        finished_at = datetime.now(UTC)
    peak_startup = max(
        levels,
        key=lambda level: level["phases"]["startup"]["sessions_per_second"],
    )
    peak_commands = max(
        levels,
        key=lambda level: level["phases"]["commands"]["commands_per_second"],
    )
    if config.total_sessions is not None:
        measurement = (
            "rolling complete session trajectories; each concurrency worker "
            "creates one container, runs all commands, and confirms deletion "
            "before taking its next trajectory"
        )
    else:
        measurement = (
            "barrier-separated startup, command, and close phases within each "
            "sequential wave; waves never overlap, and command timing starts "
            "only after every handshake in that wave finishes"
        )
    return {
        "benchmark": "literegistry-podman-affinity-throughput",
        "measurement": measurement,
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "config": {
            **asdict(config),
            "concurrency": list(config.concurrency),
        },
        "warmup": warmup,
        "levels": levels,
        "peak": {
            "startup": {
                "concurrency": peak_startup["concurrency"],
                "sessions_per_second": peak_startup["phases"]["startup"][
                    "sessions_per_second"
                ],
            },
            "commands": {
                "concurrency": peak_commands["concurrency"],
                "commands_per_second": peak_commands["phases"]["commands"][
                    "commands_per_second"
                ],
            },
        },
    }


def format_podman_benchmark(result: dict[str, Any]) -> str:
    """Render lifecycle latency, achieved rates, and traffic distribution."""

    lines = [
        (
            f"Podman replicas={result['config']['replicas']} "
            f"total_sessions={result['config'].get('total_sessions')} "
            f"mode={'rolling-lifecycle' if result['config'].get('total_sessions') else 'phased-waves'} "
            f"gateway={result['config']['gateway_url']}"
        ),
        (
            "concurrency  samples/max-live  start-ok  cmd-ok  "
            "startup-mean-ms  startups/s  command-mean-ms  commands/s  "
            "traffic(session|command)"
        ),
    ]
    warmup = result.get("warmup")
    if warmup:
        lines.insert(
            1,
            (
                "untimed image warmup="
                f"{warmup['observed_replicas']}/{warmup['expected_replicas']} "
                f"attempts={warmup['attempted_sessions']} "
                f"seconds={warmup['wall_seconds']:.1f}"
            ),
        )
    expected = result["config"]["replicas"]
    for level in result["levels"]:
        latency = level["latency"]
        session_replicas = level["traffic"]["sessions"]["observed_replicas"]
        command_replicas = level["traffic"]["commands"]["observed_replicas"]
        startup_successes = level["phases"]["startup"]["successful_sessions"]
        requested_commands = level.get(
            "requested_commands",
            level["requested_sessions"]
            * result["config"]["commands_per_session"],
        )
        lines.append(
            f"{level['concurrency']:>11}  "
            f"{level['requested_sessions']:>7}/"
            f"{level.get('max_live_sessions', level['requested_sessions']):<8}  "
            f"{startup_successes:>5}/{level['requested_sessions']:<5}  "
            f"{level['completed_commands']:>5}/{requested_commands:<5}  "
            f"{latency['handshake']['mean_ms']:>15.1f}  "
            f"{level['phases']['startup']['sessions_per_second']:>10.3f}  "
            f"{latency['command']['mean_ms']:>15.1f}  "
            f"{level['phases']['commands']['commands_per_second']:>10.3f}  "
            f"{session_replicas:>2}/{expected:<2}|"
            f"{command_replicas:>2}/{expected:<2}"
        )
    return "\n".join(lines)


def compare_podman_benchmark_results(
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compare identical workloads run against different replica counts."""

    if not results:
        raise ValueError("at least one benchmark result is required")
    ordered = sorted(results, key=lambda result: result["config"]["replicas"])
    replicas = [result["config"]["replicas"] for result in ordered]
    if len(set(replicas)) != len(replicas):
        raise ValueError("benchmark results must use unique replica counts")
    if any(
        result.get("benchmark") not in {
            "literegistry-podman-affinity-throughput",
            "jtc-podman-affinity-throughput",
        }
        for result in ordered
    ):
        raise ValueError("all inputs must be Podman affinity throughput results")

    comparison_fields = (
        "concurrency",
        "sessions_per_worker",
        "waves",
        "total_sessions",
        "commands_per_session",
        "image",
        "command_timeout",
        "workdir",
    )
    baseline_config = ordered[0]["config"]
    for result in ordered[1:]:
        mismatched = [
            field
            for field in comparison_fields
            if result["config"].get(field) != baseline_config.get(field)
        ]
        if mismatched:
            raise ValueError(
                "benchmark workloads differ in: " + ", ".join(mismatched)
            )

    baseline_replicas = replicas[0]
    baseline_levels = {
        level["concurrency"]: level for level in ordered[0]["levels"]
    }
    rows = []
    for result in ordered:
        replica_count = result["config"]["replicas"]
        replica_multiplier = replica_count / baseline_replicas
        for level in result["levels"]:
            concurrency = level["concurrency"]
            baseline = baseline_levels[concurrency]
            startup_rate = level["phases"]["startup"]["sessions_per_second"]
            command_rate = level["phases"]["commands"]["commands_per_second"]
            baseline_startup = baseline["phases"]["startup"][
                "sessions_per_second"
            ]
            baseline_command = baseline["phases"]["commands"][
                "commands_per_second"
            ]
            startup_speedup = (
                startup_rate / baseline_startup if baseline_startup else 0.0
            )
            command_speedup = (
                command_rate / baseline_command if baseline_command else 0.0
            )
            session_traffic = level["traffic"]["sessions"]
            command_traffic = level["traffic"]["commands"]
            startup_successes = level["phases"]["startup"][
                "successful_sessions"
            ]
            completed_commands = level["phases"]["commands"][
                "completed_commands"
            ]
            requested_commands = (
                level["requested_sessions"]
                * result["config"]["commands_per_session"]
            )
            rows.append(
                {
                    "replicas": replica_count,
                    "concurrency": concurrency,
                    "requested_sessions": level["requested_sessions"],
                    "successful_sessions": level["successful_sessions"],
                    "failed_sessions": level["failed_sessions"],
                    "startup_success_rate": round(
                        startup_successes / level["requested_sessions"],
                        6,
                    ),
                    "command_completion_rate": round(
                        completed_commands / requested_commands,
                        6,
                    ),
                    "lifecycle_success_rate": round(
                        level["successful_sessions"]
                        / level["requested_sessions"],
                        6,
                    ),
                    "completed_commands": completed_commands,
                    "requested_commands": requested_commands,
                    "startup_sessions_per_second": startup_rate,
                    "startup_speedup_vs_baseline": round(
                        startup_speedup,
                        6,
                    ),
                    "startup_scaling_efficiency": round(
                        startup_speedup / replica_multiplier,
                        6,
                    ),
                    "commands_per_second": command_rate,
                    "command_speedup_vs_baseline": round(
                        command_speedup,
                        6,
                    ),
                    "command_scaling_efficiency": round(
                        command_speedup / replica_multiplier,
                        6,
                    ),
                    "startup_p95_ms": level["latency"]["handshake"][
                        "p95_ms"
                    ],
                    "command_p95_ms": level["latency"]["command"]["p95_ms"],
                    "session_replica_coverage": (
                        f"{session_traffic['observed_replicas']}/"
                        f"{session_traffic['expected_replicas']}"
                    ),
                    "command_replica_coverage": (
                        f"{command_traffic['observed_replicas']}/"
                        f"{command_traffic['expected_replicas']}"
                    ),
                    "all_replicas_received_sessions": session_traffic[
                        "all_replicas_received_traffic"
                    ],
                    "all_replicas_received_commands": command_traffic[
                        "all_replicas_received_traffic"
                    ],
                    "session_instance_distribution": level[
                        "instance_distribution"
                    ],
                    "command_instance_distribution": level[
                        "command_instance_distribution"
                    ],
                }
            )
    rows.sort(key=lambda row: (row["concurrency"], row["replicas"]))
    return {
        "benchmark": "literegistry-podman-horizontal-scaling-comparison",
        "baseline_replicas": baseline_replicas,
        "replica_counts": replicas,
        "workload": {
            field: baseline_config.get(field)
            for field in comparison_fields
        },
        "rows": rows,
    }


def format_podman_scaling_comparison(result: dict[str, Any]) -> str:
    """Render horizontal scaling and traffic coverage at fixed workloads."""

    lines = [
        (
            "Podman horizontal scaling "
            f"(baseline={result['baseline_replicas']} replica(s))"
        ),
        (
            "replicas  concurrency  start-ok  cmd-ok  startup/s  startup-x  "
            "commands/s  command-x  traffic(session|command)"
        ),
        (
            "                                                    "
            "startup-p95  command-p95"
        ),
    ]
    for row in result["rows"]:
        lines.append(
            f"{row['replicas']:>8}  "
            f"{row['concurrency']:>11}  "
            f"{row['startup_success_rate']:>7.1%}  "
            f"{row['command_completion_rate']:>6.1%}  "
            f"{row['startup_sessions_per_second']:>9.3f}  "
            f"{row['startup_speedup_vs_baseline']:>9.2f}  "
            f"{row['commands_per_second']:>10.3f}  "
            f"{row['command_speedup_vs_baseline']:>9.2f}  "
            f"{row['session_replica_coverage']:>7}|"
            f"{row['command_replica_coverage']:<7}             "
            f"{row['startup_p95_ms']:>11.1f}  "
            f"{row['command_p95_ms']:>11.1f}"
        )
    return "\n".join(lines)

@dataclass(frozen=True)
class PodmanCreationBenchmarkConfig:
    """Concurrent container-creation sweep with cleanup outside measurement."""

    gateway_url: str
    replicas: int
    concurrency: tuple[int, ...] = (
        1,
        2,
        4,
        8,
        16,
        32,
        64,
        128,
        256,
        512,
    )
    image: str = PODMAN_SESSION_IMAGE
    request_timeout: float = 70
    handshake_timeout: float = 300
    max_retries: int = 1
    cleanup_concurrency: int = 16
    skip_final_cleanup: bool = False
    client_id_prefix: str = "literegistry-podman-create-benchmark"

    def validate(self) -> PodmanCreationBenchmarkConfig:
        if not self.gateway_url.strip():
            raise ValueError("gateway_url must be non-empty")
        if self.replicas < 1:
            raise ValueError("replicas must be positive")
        if not self.concurrency:
            raise ValueError("concurrency must contain at least one level")
        if any(level < 1 for level in self.concurrency):
            raise ValueError("concurrency levels must be positive")
        if len(set(self.concurrency)) != len(self.concurrency):
            raise ValueError("concurrency levels must be unique")
        if tuple(sorted(self.concurrency)) != self.concurrency:
            raise ValueError("concurrency levels must be increasing")
        if not self.image.strip():
            raise ValueError("image must be non-empty")
        if self.request_timeout <= 0 or self.handshake_timeout <= 0:
            raise ValueError("HTTP timeouts must be positive")
        if self.max_retries < 1:
            raise ValueError("max_retries must be at least 1")
        if self.cleanup_concurrency < 1:
            raise ValueError("cleanup_concurrency must be positive")
        if not self.client_id_prefix.strip():
            raise ValueError("client_id_prefix must be non-empty")
        return self


@dataclass
class _CreationSample:
    client: PodmanExecutionClient
    success: bool = False
    container_id: str | None = None
    instance_id: str | None = None
    creation_seconds: float = 0
    error: str | None = None
    cleanup_seconds: float | None = None
    cleanup_error: str | None = None


async def _create_container(
    config: PodmanCreationBenchmarkConfig,
    concurrency: int,
    index: int,
    start: asyncio.Event,
) -> _CreationSample:
    client = PodmanExecutionClient(
        config.gateway_url,
        image=config.image,
        client_id=(
            f"{config.client_id_prefix}-r{config.replicas}"
            f"-c{concurrency}-s{index}-{uuid4().hex[:8]}"
        ),
        timeout=config.request_timeout,
        handshake_timeout=config.handshake_timeout,
        max_retries=config.max_retries,
    )
    sample = _CreationSample(client=client)
    await start.wait()
    created_at = time.perf_counter()
    try:
        handshake = await client.start()
        sample.container_id = client.container_id
        instance_id = handshake.get("instance_id")
        if isinstance(instance_id, str) and instance_id:
            sample.instance_id = instance_id
        sample.success = True
    except Exception as error:
        sample.error = _error_text(error)
    finally:
        sample.creation_seconds = time.perf_counter() - created_at
    return sample


async def _cleanup_created_containers(
    samples: list[_CreationSample],
    cleanup_concurrency: int,
) -> float:
    semaphore = asyncio.Semaphore(cleanup_concurrency)

    async def close(sample: _CreationSample) -> None:
        if not sample.client.started:
            return
        async with semaphore:
            started_at = time.perf_counter()
            try:
                result = await sample.client.close()
                if not result or result.get("removed") is not True:
                    raise RuntimeError(
                        f"close did not remove the container: {result}"
                    )
            except Exception as error:
                sample.cleanup_error = _error_text(error)
            finally:
                sample.cleanup_seconds = time.perf_counter() - started_at

    started_at = time.perf_counter()
    await asyncio.gather(*(close(sample) for sample in samples))
    return time.perf_counter() - started_at


async def _run_creation_level(
    config: PodmanCreationBenchmarkConfig,
    concurrency: int,
    *,
    cleanup: bool = True,
) -> dict[str, Any]:
    start = asyncio.Event()
    tasks = [
        asyncio.create_task(
            _create_container(config, concurrency, index, start)
        )
        for index in range(concurrency)
    ]
    creation_started = time.perf_counter()
    start.set()
    samples = await asyncio.gather(*tasks)
    creation_wall_seconds = time.perf_counter() - creation_started

    cleanup_wall_seconds = 0.0
    if cleanup:
        cleanup_wall_seconds = await _cleanup_created_containers(
            samples,
            config.cleanup_concurrency,
        )
    successful = [sample for sample in samples if sample.success]
    instances = Counter(
        sample.instance_id for sample in successful if sample.instance_id
    )
    creation_errors = Counter(
        sample.error for sample in samples if sample.error
    )
    cleanup_errors = Counter(
        sample.cleanup_error for sample in samples if sample.cleanup_error
    )
    return {
        "replicas": config.replicas,
        "concurrency": concurrency,
        "requested_creations": concurrency,
        "successful_creations": len(successful),
        "failed_creations": concurrency - len(successful),
        "unique_container_ids": len(
            {sample.container_id for sample in successful if sample.container_id}
        ),
        "instance_distribution": dict(sorted(instances.items())),
        "creation_wall_seconds": round(creation_wall_seconds, 6),
        "containers_per_second": round(
            len(successful) / creation_wall_seconds,
            6,
        ),
        "creation_latency": _latency_summary(
            [sample.creation_seconds for sample in successful]
        ),
        "cleanup": {
            "skipped": not cleanup,
            "concurrency": config.cleanup_concurrency,
            "wall_seconds": round(cleanup_wall_seconds, 6),
            "failed": sum(cleanup_errors.values()),
            "latency": _latency_summary(
                [
                    sample.cleanup_seconds
                    for sample in samples
                    if sample.cleanup_seconds is not None
                ]
            ),
            "errors": [
                {"error": error, "count": count}
                for error, count in cleanup_errors.most_common(10)
            ],
        },
        "errors": [
            {"error": error, "count": count}
            for error, count in creation_errors.most_common(10)
        ],
    }


async def run_podman_creation_benchmark(
    config: PodmanCreationBenchmarkConfig,
) -> dict[str, Any]:
    """Measure container creation only; close containers after each level."""

    config = config.validate()
    started_at = datetime.now(UTC)
    levels = []
    for index, concurrency in enumerate(config.concurrency):
        final_level = index == len(config.concurrency) - 1
        levels.append(
            await _run_creation_level(
                config,
                concurrency,
                cleanup=not (final_level and config.skip_final_cleanup),
            )
        )
    finished_at = datetime.now(UTC)
    peak = max(levels, key=lambda level: level["containers_per_second"])
    return {
        "benchmark": "literegistry-podman-affinity-container-creation",
        "measurement": (
            "POST /affinity/handshake through container-ready response; "
            "cleanup is excluded"
        ),
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "config": {
            **asdict(config),
            "concurrency": list(config.concurrency),
        },
        "levels": levels,
        "peak": {
            "concurrency": peak["concurrency"],
            "containers_per_second": peak["containers_per_second"],
        },
    }


def format_podman_creation_benchmark(result: dict[str, Any]) -> str:
    """Render container-creation scaling as a compact table."""

    lines = [
        (
            f"Podman creation replicas={result['config']['replicas']} "
            f"gateway={result['config']['gateway_url']}"
        ),
        (
            "concurrency  ok/fail  containers/s  create-p50-ms  "
            "create-p95-ms  create-p99-ms  cleanup-s"
        ),
    ]
    for level in result["levels"]:
        latency = level["creation_latency"]
        lines.append(
            f"{level['concurrency']:>11}  "
            f"{level['successful_creations']:>3}/{level['failed_creations']:<4}  "
            f"{level['containers_per_second']:>12.3f}  "
            f"{latency['p50_ms']:>13.1f}  "
            f"{latency['p95_ms']:>13.1f}  "
            f"{latency['p99_ms']:>13.1f}  "
            f"{level['cleanup']['wall_seconds']:>9.2f}"
        )
    return "\n".join(lines)
