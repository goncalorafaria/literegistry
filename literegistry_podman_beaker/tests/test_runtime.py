from __future__ import annotations

import pytest

from literegistry.coop.ports import parse_assignments, port_candidates
from literegistry.coop.redis import redis_ping


def test_port_candidates_are_stable_and_start_with_preference() -> None:
    first = port_candidates(
        "experiment:podman:0",
        "PORT",
        32123,
        minimum=1024,
        maximum=65000,
        attempts=8,
    )
    second = port_candidates(
        "experiment:podman:0",
        "PORT",
        32123,
        minimum=1024,
        maximum=65000,
        attempts=8,
    )
    assert first == second
    assert first[0] == 32123
    assert len(first) == 8


def test_redis_ping_rejects_non_redis_urls_before_network() -> None:
    with pytest.raises(ValueError, match="redis"):
        redis_ping("http://registry.example:6379")


def test_fire_assignment_parser_supports_one_or_many_values() -> None:
    assert parse_assignments("PORT=32123") == {"PORT": 32123}
    assert parse_assignments(("HTTP=8080", "ADMIN=8081")) == {
        "HTTP": 8080,
        "ADMIN": 8081,
    }
