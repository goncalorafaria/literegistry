#!/usr/bin/env python3
"""Compare phase-isolated Podman benchmarks across replica counts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from podman_benchmark import (
    compare_podman_benchmark_results,
    format_podman_scaling_comparison,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare startup and command scaling from equivalent Podman "
            "benchmark JSON files"
        )
    )
    parser.add_argument(
        "results",
        nargs="+",
        type=Path,
        help="JSON results produced by benchmark_podman_execution_client.py",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--json-only", action="store_true")
    args = parser.parse_args()

    benchmarks = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in args.results
    ]
    comparison = compare_podman_benchmark_results(benchmarks)
    serialized = json.dumps(comparison, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    if args.json_only:
        print(serialized, end="")
    else:
        print(format_podman_scaling_comparison(comparison))
        if args.output:
            print(f"JSON: {args.output}")


if __name__ == "__main__":
    main()
