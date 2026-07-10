#!/usr/bin/env python3
"""Compare simple request routing policies on fake servers.

This is intentionally small and deterministic: fake servers have fixed latency
distributions, optional failures, and no real networking. It is useful for
checking whether a routing policy learns the obvious good endpoints.
"""

from __future__ import annotations

import argparse
import heapq
import random
import statistics
from collections import Counter, deque
from dataclasses import dataclass
from typing import Protocol

from literegistry.bandit import Exp3Dynamic


@dataclass(frozen=True)
class FakeServerSpec:
    name: str
    mean_latency: float
    jitter: float
    capacity: int
    failure_rate: float = 0.0
    tail_probability: float = 0.0
    tail_mean_latency: float = 2.0
    tail_jitter: float = 0.5
    congestion_factor: float = 1.5


class FakeServer:
    def __init__(self, spec: FakeServerSpec):
        self.spec = spec
        self.finish_times: list[float] = []
        self.load_samples: list[int] = []

    def current_load(self, now: float) -> int:
        while self.finish_times and self.finish_times[0] <= now:
            heapq.heappop(self.finish_times)
        return len(self.finish_times)

    def call(self, rng: random.Random, now: float) -> tuple[bool, float]:
        current_load = self.current_load(now)
        self.load_samples.append(current_load)
        success = rng.random() >= self.spec.failure_rate

        if rng.random() < self.spec.tail_probability:
            base_latency = rng.gauss(
                self.spec.tail_mean_latency,
                self.spec.tail_jitter,
            )
        else:
            base_latency = rng.gauss(self.spec.mean_latency, self.spec.jitter)

        utilization = current_load / max(1, self.spec.capacity)
        overload = max(0.0, utilization - 1.0)
        latency = base_latency * (1.0 + self.spec.congestion_factor * overload**2)
        latency = max(0.001, latency)
        heapq.heappush(self.finish_times, now + latency)
        return success, latency

    def average_load(self) -> float:
        if not self.load_samples:
            return 0.0
        return statistics.mean(self.load_samples)

    def p95_load(self) -> float:
        if not self.load_samples:
            return 0.0
        return percentile([float(load) for load in self.load_samples], 95)


class Router(Protocol):
    name: str

    def choose(self, servers: list[str]) -> tuple[str, float]:
        ...

    def update(self, server: str, success: bool, latency: float, prob: float) -> None:
        ...


class RandomRouter:
    name = "random"

    def __init__(self, rng: random.Random):
        self.rng = rng

    def choose(self, servers: list[str]) -> tuple[str, float]:
        return self.rng.choice(servers), 1.0 / len(servers)

    def update(self, server: str, success: bool, latency: float, prob: float) -> None:
        return None


class RoundRobinRouter:
    name = "round_robin"

    def __init__(self):
        self.index = 0

    def choose(self, servers: list[str]) -> tuple[str, float]:
        server = servers[self.index % len(servers)]
        self.index += 1
        return server, 1.0 / len(servers)

    def update(self, server: str, success: bool, latency: float, prob: float) -> None:
        return None


class OracleRouter:
    name = "oracle_server_0"

    def __init__(self, specs: dict[str, FakeServerSpec], penalty_latency: float):
        self.best = sorted(specs)[0]

    def choose(self, servers: list[str]) -> tuple[str, float]:
        if self.best in servers:
            return self.best, 1.0
        return servers[0], 1.0

    def update(self, server: str, success: bool, latency: float, prob: float) -> None:
        return None


class LeastInflightRouter:
    name = "least_inflight"

    def __init__(self):
        self.server_objects: dict[str, FakeServer] = {}
        self.now_ref: list[float] = [0.0]

    def bind(self, server_objects: dict[str, FakeServer], now_ref: list[float]) -> None:
        self.server_objects = server_objects
        self.now_ref = now_ref

    def choose(self, servers: list[str]) -> tuple[str, float]:
        now = self.now_ref[0]
        return min(
            servers,
            key=lambda server: (
                self.server_objects[server].current_load(now),
                server,
            ),
        ), 1.0

    def update(self, server: str, success: bool, latency: float, prob: float) -> None:
        return None


class EWMALatencyRouter:
    name = "ewma_latency"

    def __init__(self, alpha: float = 0.15, penalty_latency: float = 5.0):
        self.alpha = alpha
        self.penalty_latency = penalty_latency
        self.estimate: dict[str, float] = {}

    def choose(self, servers: list[str]) -> tuple[str, float]:
        unseen = [server for server in servers if server not in self.estimate]
        if unseen:
            return unseen[0], 1.0 / len(servers)
        return min(servers, key=lambda server: self.estimate[server]), 1.0

    def update(self, server: str, success: bool, latency: float, prob: float) -> None:
        observed = latency if success else self.penalty_latency
        old = self.estimate.get(server, observed)
        self.estimate[server] = self.alpha * observed + (1 - self.alpha) * old


class EpsilonEWMALatencyRouter(EWMALatencyRouter):
    name = "epsilon_ewma"

    def __init__(
        self,
        rng: random.Random,
        alpha: float = 0.15,
        epsilon: float = 0.05,
        penalty_latency: float = 5.0,
    ):
        super().__init__(alpha=alpha, penalty_latency=penalty_latency)
        self.rng = rng
        self.epsilon = epsilon

    def choose(self, servers: list[str]) -> tuple[str, float]:
        unseen = [server for server in servers if server not in self.estimate]
        if unseen:
            return unseen[0], 1.0 / len(servers)
        if self.rng.random() < self.epsilon:
            return self.rng.choice(servers), self.epsilon / len(servers)
        return min(servers, key=lambda server: self.estimate[server]), 1.0 - self.epsilon


class Exp3Router:
    name = "exp3"

    def __init__(self, gamma: float, l_max: float):
        self.bandit = Exp3Dynamic(gamma=gamma, L_max=l_max)

    def choose(self, servers: list[str]) -> tuple[str, float]:
        chosen, probs = self.bandit.get_arm(servers, k=1)
        if not chosen:
            raise RuntimeError("No server selected")
        return chosen[0], probs[0]

    def update(self, server: str, success: bool, latency: float, prob: float) -> None:
        self.bandit.update(server, success=success, latency=latency, prob=prob)


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    rank = min(len(values) - 1, max(0, round((pct / 100.0) * (len(values) - 1))))
    return values[rank]


def run_simulation(
    router: Router,
    specs: list[FakeServerSpec],
    requests: int,
    seed: int,
    penalty_latency: float,
    arrival_rate: float,
) -> dict[str, object]:
    rng = random.Random(seed)
    servers = {spec.name: FakeServer(spec) for spec in specs}
    active = list(servers)
    now_ref = [0.0]
    if hasattr(router, "bind"):
        router.bind(servers, now_ref)
    selections: Counter[str] = Counter()
    latencies: list[float] = []
    failures = 0
    recent = deque(maxlen=500)

    for _ in range(requests):
        now_ref[0] += rng.expovariate(arrival_rate)
        server, prob = router.choose(active)
        success, latency = servers[server].call(rng, now_ref[0])
        router.update(server, success=success, latency=latency, prob=prob)

        selections[server] += 1
        failures += int(not success)
        effective_latency = latency if success else penalty_latency
        latencies.append(effective_latency)
        recent.append(effective_latency)

    return {
        "router": router.name,
        "avg": statistics.mean(latencies),
        "recent_avg": statistics.mean(recent),
        "p50": percentile(latencies, 50),
        "p95": percentile(latencies, 95),
        "failures": failures,
        "selections": selections,
        "loads": {
            name: (server.average_load(), server.p95_load())
            for name, server in servers.items()
        },
    }


def format_selections(selections: Counter[str], total: int) -> str:
    parts = []
    for server, count in sorted(selections.items()):
        parts.append(f"{server}={100 * count / total:5.1f}%")
    return "  ".join(parts)


def format_loads(loads: dict[str, tuple[float, float]]) -> str:
    parts = []
    for server, (avg_load, p95_load) in sorted(loads.items()):
        parts.append(f"{server}={avg_load:.1f}/{p95_load:.0f}")
    return "  ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--gamma", type=float, default=0.2)
    parser.add_argument("--l-max", type=float, default=2.0)
    parser.add_argument("--penalty-latency", type=float, default=5.0)
    parser.add_argument("--arrival-rate", type=float, default=120.0)
    parser.add_argument(
        "--scenario",
        choices=("homogeneous", "two-groups"),
        default="homogeneous",
    )
    args = parser.parse_args()

    if args.scenario == "homogeneous":
        specs = [
            FakeServerSpec(
                f"server_{i}",
                mean_latency=0.25,
                jitter=0.04,
                capacity=32,
                failure_rate=0.01,
                tail_probability=0.04,
                tail_mean_latency=3.0,
                tail_jitter=0.8,
            )
            for i in range(4)
        ]
    else:
        specs = [
            FakeServerSpec(
                "fast_0",
                mean_latency=0.25,
                jitter=0.04,
                capacity=32,
                failure_rate=0.01,
                tail_probability=0.04,
                tail_mean_latency=3.0,
                tail_jitter=0.8,
            ),
            FakeServerSpec(
                "fast_1",
                mean_latency=0.25,
                jitter=0.04,
                capacity=32,
                failure_rate=0.01,
                tail_probability=0.04,
                tail_mean_latency=3.0,
                tail_jitter=0.8,
            ),
            FakeServerSpec(
                "slow_0",
                mean_latency=0.25 / 0.60,
                jitter=0.04 / 0.60,
                capacity=19,
                failure_rate=0.01,
                tail_probability=0.04,
                tail_mean_latency=3.0 / 0.60,
                tail_jitter=0.8 / 0.60,
            ),
            FakeServerSpec(
                "slow_1",
                mean_latency=0.25 / 0.60,
                jitter=0.04 / 0.60,
                capacity=19,
                failure_rate=0.01,
                tail_probability=0.04,
                tail_mean_latency=3.0 / 0.60,
                tail_jitter=0.8 / 0.60,
            ),
        ]
    specs_by_name = {spec.name: spec for spec in specs}

    routers: list[Router] = [
        RandomRouter(random.Random(args.seed + 100)),
        RoundRobinRouter(),
        LeastInflightRouter(),
        EWMALatencyRouter(penalty_latency=args.penalty_latency),
        EpsilonEWMALatencyRouter(
            random.Random(args.seed + 200),
            penalty_latency=args.penalty_latency,
        ),
        Exp3Router(gamma=args.gamma, l_max=args.l_max),
        OracleRouter(specs_by_name, penalty_latency=args.penalty_latency),
    ]

    print(f"Scenario: {args.scenario}")
    print("Fake servers:")
    for spec in specs:
        print(
            f"  {spec.name:12s} mean={spec.mean_latency:.3f}s "
            f"jitter={spec.jitter:.3f}s cap={spec.capacity:3d} "
            f"tail={100 * spec.tail_probability:.1f}%@{spec.tail_mean_latency:.1f}s "
            f"failure={100 * spec.failure_rate:.1f}%"
        )
    print(f"Arrival rate: {args.arrival_rate:.1f} req/s")
    print()
    print(
        f"{'router':18s} {'avg':>8s} {'recent':>8s} {'p50':>8s} "
        f"{'p95':>8s} {'fail':>6s}  selections"
    )
    print("-" * 110)

    for router in routers:
        result = run_simulation(
            router=router,
            specs=specs,
            requests=args.requests,
            seed=args.seed,
            penalty_latency=args.penalty_latency,
            arrival_rate=args.arrival_rate,
        )
        selections = result["selections"]
        print(
            f"{result['router']:18s} "
            f"{result['avg']:8.3f} "
            f"{result['recent_avg']:8.3f} "
            f"{result['p50']:8.3f} "
            f"{result['p95']:8.3f} "
            f"{result['failures']:6d}  "
            f"{format_selections(selections, args.requests)}"
        )
        print(f"{'':18s} load avg/p95: {format_loads(result['loads'])}")


if __name__ == "__main__":
    main()
