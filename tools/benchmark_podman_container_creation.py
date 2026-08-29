#!/usr/bin/env python3
"""Benchmark concurrent Podman creation through a LiteRegistry gateway."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from podman_benchmark import (
    PODMAN_SESSION_IMAGE,
    PodmanCreationBenchmarkConfig,
    format_podman_creation_benchmark,
    run_podman_creation_benchmark,
)


def _concurrency_levels(raw: str) -> tuple[int, ...]:
    try:
        levels = tuple(
            int(value.strip()) for value in raw.split(",") if value.strip()
        )
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "concurrency must be a comma-separated list of integers"
        ) from error
    if not levels:
        raise argparse.ArgumentTypeError(
            "at least one concurrency level is required"
        )
    return levels


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Measure handshake-to-container-ready throughput; cleanup is not timed"
        )
    )
    parser.add_argument("--gateway-url", required=True)
    parser.add_argument("--replicas", required=True, type=int)
    parser.add_argument(
        "--concurrency",
        type=_concurrency_levels,
        default=(1, 2, 4, 8, 16, 32, 64, 128, 256, 512),
    )
    parser.add_argument("--image", default=PODMAN_SESSION_IMAGE)
    parser.add_argument("--request-timeout", type=float, default=70)
    parser.add_argument("--handshake-timeout", type=float, default=300)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--cleanup-concurrency", type=int, default=16)
    parser.add_argument(
        "--skip-final-cleanup",
        action="store_true",
        help=(
            "leave final-wave containers for isolated-stack teardown; "
            "earlier levels are still cleaned"
        ),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--json-only", action="store_true")
    args = parser.parse_args()

    result = asyncio.run(
        run_podman_creation_benchmark(
            PodmanCreationBenchmarkConfig(
                gateway_url=args.gateway_url,
                replicas=args.replicas,
                concurrency=args.concurrency,
                image=args.image,
                request_timeout=args.request_timeout,
                handshake_timeout=args.handshake_timeout,
                max_retries=args.max_retries,
                cleanup_concurrency=args.cleanup_concurrency,
                skip_final_cleanup=args.skip_final_cleanup,
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
        print(format_podman_creation_benchmark(result))
        if args.output:
            print(f"JSON: {args.output}")


if __name__ == "__main__":
    main()
