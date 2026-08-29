from __future__ import annotations

from pathlib import Path

import pytest

from literegistry.coop.artifacts import (
    IncompleteArtifactError,
    ensure_directory_artifact,
)
from literegistry.coop.ports import parse_assignments, port_candidates
from literegistry.coop.redis import redis_ping


def test_directory_artifact_is_built_atomically_and_reused(tmp_path: Path) -> None:
    target = tmp_path / "index"
    builds: list[Path] = []

    def ready(directory: Path) -> bool:
        return (directory / "ready").is_file()

    def build(staging: Path) -> None:
        builds.append(staging)
        assert staging.parent == tmp_path or staging.parent.parent == tmp_path
        (staging / "payload").write_text("ai2 hello\n", encoding="utf-8")
        (staging / "ready").write_text("ok\n", encoding="utf-8")

    assert ensure_directory_artifact(target, ready=ready, build=build) == target
    assert (target / "payload").read_text(encoding="utf-8") == "ai2 hello\n"
    assert len(builds) == 1

    ensure_directory_artifact(target, ready=ready, build=build)
    assert len(builds) == 1
    assert not list(tmp_path.glob(".index.materialize-*/artifact"))


def test_directory_artifact_preserves_incomplete_nonempty_target(tmp_path: Path) -> None:
    target = tmp_path / "index"
    target.mkdir()
    partial = target / "partial"
    partial.write_text("keep", encoding="utf-8")

    with pytest.raises(IncompleteArtifactError, match="incomplete"):
        ensure_directory_artifact(
            target,
            ready=lambda directory: (directory / "ready").is_file(),
            build=lambda staging: (staging / "ready").write_text("ok", encoding="utf-8"),
        )

    assert partial.read_text(encoding="utf-8") == "keep"


def test_port_candidates_and_assignment_parser_are_canonical() -> None:
    first = port_candidates(
        "experiment:service:0",
        "PORT",
        32123,
        minimum=1024,
        maximum=65000,
        attempts=8,
    )
    second = port_candidates(
        "experiment:service:0",
        "PORT",
        32123,
        minimum=1024,
        maximum=65000,
        attempts=8,
    )
    assert first == second
    assert first[0] == 32123
    assert parse_assignments(("HTTP=8080", "ADMIN=8081")) == {
        "HTTP": 8080,
        "ADMIN": 8081,
    }


def test_redis_ping_rejects_non_redis_urls_before_network() -> None:
    with pytest.raises(ValueError, match="redis"):
        redis_ping("http://registry.example:6379")
