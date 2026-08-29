from __future__ import annotations

import pytest

from literegistry.coop.ports import parse_assignments, port_candidates


def test_port_candidates_are_deterministic_and_keep_preferred_first() -> None:
    first = port_candidates("stack:python:0", "PORT", 22123, minimum=1024, maximum=65000, attempts=8)
    second = port_candidates("stack:python:0", "PORT", 22123, minimum=1024, maximum=65000, attempts=8)

    assert first == second
    assert first[0] == 22123
    assert all(1024 <= port <= 65000 for port in first)


def test_assignment_parser_rejects_duplicates() -> None:
    assert parse_assignments(["PORT=1234", "METRICS_PORT=1235"]) == {
        "PORT": 1234,
        "METRICS_PORT": 1235,
    }
    with pytest.raises(ValueError, match="duplicate"):
        parse_assignments(["PORT=1234", "PORT=1235"])
