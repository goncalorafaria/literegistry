#!/usr/bin/env python3
"""Benchmark Podman sessions through the LiteRegistry gateway."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from podman_benchmark import (
    PODMAN_SESSION_IMAGE,
    PodmanBenchmarkConfig,
    format_podman_benchmark,
    run_podman_benchmark,
)


def _concurrency_levels(raw: str) -> tuple[int, ...]:
    try:
        levels = tuple(int(value.strip()) for value in raw.split(",") if value.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "concurrency must be a comma-separated list of integers"
        ) from error
    if not levels:
        raise argparse.ArgumentTypeError("at least one concurrency level is required")
    return levels


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Measure Podman session lifecycle and command throughput"
        )
    )
    parser.add_argument("--gateway-url", required=True)
    parser.add_argument(
        "--replicas",
        required=True,
        type=int,
        help="Podman replica count for this run (recorded as a comparison label)",
    )
    parser.add_argument(
        "--concurrency",
        type=_concurrency_levels,
        default=(1, 2, 4, 8, 16, 32, 64, 128),
        help="increasing client concurrency levels (default: 1,2,4,8,16,32,64,128)",
    )
    parser.add_argument("--sessions-per-worker", type=int, default=2)
    parser.add_argument(
        "--waves",
        type=int,
        default=1,
        help=(
            "sequential workload waves per concurrency level; increases samples "
            "without increasing peak live containers (default: 1)"
        ),
    )
    parser.add_argument(
        "--total-sessions",
        type=int,
        help=(
            "fixed total session lifecycles at every concurrency level; each "
            "concurrency worker performs create, commands, and confirmed delete "
            "before taking another lifecycle"
        ),
    )
    parser.add_argument("--commands-per-session", type=int, default=4)
    parser.add_argument("--image", default=PODMAN_SESSION_IMAGE)
    parser.add_argument("--command-timeout", type=float, default=10)
    parser.add_argument("--request-timeout", type=float, default=70)
    parser.add_argument("--handshake-timeout", type=float, default=300)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--workdir", default="/workspace")
    parser.add_argument(
        "--warmup-all-replicas",
        action="store_true",
        help=(
            "before measurement, create and delete one container on every "
            "replica so image pulls are excluded from benchmark timing"
        ),
    )
    parser.add_argument("--warmup-concurrency", type=int, default=32)
    parser.add_argument(
        "--warmup-max-sessions",
        type=int,
        help="maximum untimed sessions allowed to reach all replicas",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--json-only", action="store_true")
    args = parser.parse_args()

    result = asyncio.run(
        run_podman_benchmark(
            PodmanBenchmarkConfig(
                gateway_url=args.gateway_url,
                replicas=args.replicas,
                concurrency=args.concurrency,
                sessions_per_worker=args.sessions_per_worker,
                waves=args.waves,
                total_sessions=args.total_sessions,
                commands_per_session=args.commands_per_session,
                image=args.image,
                command_timeout=args.command_timeout,
                request_timeout=args.request_timeout,
                handshake_timeout=args.handshake_timeout,
                max_retries=args.max_retries,
                workdir=args.workdir,
                warmup_all_replicas=args.warmup_all_replicas,
                warmup_concurrency=args.warmup_concurrency,
                warmup_max_sessions=args.warmup_max_sessions,
            )
        )
    )
    serialized = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    if args.json_only:
        print(serialized, end="")
    else:
        print(format_podman_benchmark(result))
        if args.output:
            print(f"JSON: {args.output}")


if __name__ == "__main__":
    main()
