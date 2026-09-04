from __future__ import annotations

import asyncio
import json
from pathlib import Path
import socket
import sys

import pytest

from literegistry.client import RegistryClient
from literegistry.coop.artifacts import (
    IncompleteArtifactError,
    ensure_directory_artifact,
)
from literegistry.coop.endpoints import (
    FilesystemEndpointRegistry,
    main as endpoints_main,
    prepare_endpoint_registry_storage,
    run,
    wait_for_endpoint,
)
from literegistry.coop.ports import parse_assignments, port_candidates
from literegistry.coop.redis import redis_ping
from literegistry.redis import start_redis_server


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


def test_filesystem_endpoint_registry_publishes_ttl_records(tmp_path: Path) -> None:
    async def scenario() -> None:
        registry = FilesystemEndpointRegistry(tmp_path / "nested" / "coordination")
        record = await registry.publish(
            "redis",
            "redis://node.example:6379/",
            publisher_id="redis-1",
            ttl_seconds=0.1,
            metadata={"role": "registry"},
        )

        assert record.uri == "redis://node.example:6379"
        assert await registry.get("redis") == record
        assert not await registry.delete("redis", publisher_id="redis-2")
        assert await registry.get("redis") == record

        await asyncio.sleep(0.15)
        assert await registry.get("redis") is None
        await registry.close()

    asyncio.run(scenario())


def test_prepare_endpoint_registry_storage_supports_file_and_sqlite(tmp_path: Path) -> None:
    file_root = tmp_path / "file-head"
    sqlite_parent = tmp_path / "sqlite-head"

    assert prepare_endpoint_registry_storage(f"file://{file_root}") == file_root.as_uri()
    assert prepare_endpoint_registry_storage(
        f"sqlite://{sqlite_parent / 'head.sqlite3'}"
    ) == f"sqlite://{sqlite_parent / 'head.sqlite3'}"
    assert file_root.is_dir()
    assert sqlite_parent.is_dir()
    assert file_root.stat().st_mode & 0o777 == 0o777
    assert sqlite_parent.stat().st_mode & 0o777 == 0o777


def test_old_publisher_cannot_delete_resumed_endpoint(tmp_path: Path) -> None:
    async def scenario() -> None:
        registry = FilesystemEndpointRegistry(tmp_path)
        await registry.publish(
            "gateway",
            "http://old.example:8080",
            publisher_id="old",
            ttl_seconds=5,
        )
        replacement = await registry.publish(
            "gateway",
            "http://new.example:8080",
            publisher_id="new",
            ttl_seconds=5,
        )

        assert await registry.delete("gateway", publisher_id="old")
        assert await registry.get("gateway") == replacement
        await registry.close()

    asyncio.run(scenario())


def test_wait_for_endpoint_returns_only_a_published_record(tmp_path: Path) -> None:
    async def scenario() -> None:
        root = tmp_path / "coordination"

        async def delayed_publish() -> None:
            await asyncio.sleep(0.05)
            registry = FilesystemEndpointRegistry(root)
            await registry.publish(
                "gateway",
                "http://gateway.example:8080",
                publisher_id="gateway-1",
                ttl_seconds=1,
            )
            await registry.close()

        publisher = asyncio.create_task(delayed_publish())
        record = await wait_for_endpoint(
            root,
            "gateway",
            timeout=1,
            poll_interval=0.01,
        )
        await publisher

        assert record.name == "gateway"
        assert record.uri == "http://gateway.example:8080"

    asyncio.run(scenario())


def test_endpoint_supervisor_publishes_refreshes_and_cleans_up(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1", 0))
            port = reservation.getsockname()[1]

        script = (
            "import socket,time;"
            "s=socket.socket();"
            "s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);"
            f"s.bind(('127.0.0.1',{port}));"
            "s.listen();"
            "time.sleep(2.0)"
        )
        managed = asyncio.create_task(
            asyncio.to_thread(
                run,
                str(tmp_path),
                "gateway",
                f"http://127.0.0.1:{port}",
                json.dumps([sys.executable, "-c", script]),
                "tcp",
                2,
                0.2,
                0.4,
                0.1,
                "gateway-test",
            )
        )

        record = await wait_for_endpoint(
            tmp_path,
            "gateway",
            timeout=2,
            poll_interval=0.02,
            healthcheck="tcp",
        )
        assert record.publisher_id == "gateway-test"
        registry = FilesystemEndpointRegistry(tmp_path)
        await asyncio.sleep(0.6)
        refreshed = await registry.get("gateway")
        assert refreshed is not None
        assert refreshed.published_at > record.published_at

        await asyncio.wait_for(managed, timeout=4)
        assert await registry.get("gateway") is None
        await registry.close()

    asyncio.run(scenario())


def test_redis_server_owns_filesystem_endpoint_lifecycle(tmp_path: Path) -> None:
    fake_redis = tmp_path / "fake-redis-server"
    fake_redis.write_text(
        """#!/usr/bin/env python3
import socket
import sys
import time

port = int(sys.argv[sys.argv.index("--port") + 1])
deadline = time.monotonic() + 1.0
with socket.socket() as server:
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", port))
    server.listen()
    server.settimeout(0.1)
    while time.monotonic() < deadline:
        try:
            connection, _ = server.accept()
        except TimeoutError:
            continue
        with connection:
            connection.recv(4096)
            connection.sendall(b"+PONG\\r\\n")
""",
        encoding="utf-8",
    )
    fake_redis.chmod(0o755)

    async def scenario() -> None:
        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1", 0))
            port = reservation.getsockname()[1]

        coordination = tmp_path / "redis-coordination"
        managed = asyncio.create_task(
            asyncio.to_thread(
                start_redis_server,
                port=port,
                redis_server_path=str(fake_redis),
                runtime="local",
                foreground=True,
                advertise_host="127.0.0.1",
                coordination_dir=coordination,
                coordination_ttl_seconds=0.4,
                coordination_refresh_interval=0.1,
                coordination_startup_timeout=2.0,
                coordination_healthcheck_timeout=0.2,
            )
        )

        record = await wait_for_endpoint(
            coordination,
            "redis",
            timeout=2.0,
            poll_interval=0.02,
            healthcheck="redis",
            healthcheck_timeout=0.2,
        )
        assert record.uri == f"redis://127.0.0.1:{port}"
        assert record.name == "redis"

        registry = FilesystemEndpointRegistry(coordination)
        service_registry = RegistryClient(registry.store)
        assert await service_registry.models(force=True) == {}
        await asyncio.wait_for(managed, timeout=3.0)
        assert await registry.get("redis") is None
        await registry.close()

    asyncio.run(scenario())


def test_endpoint_fire_cli_publishes_and_resolves(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    endpoints_main(
        [
            "publish",
            f"--root={tmp_path}",
            "--name=gateway",
            "--uri=http://gateway.example:8080",
            "--publisher-id=cli-test",
            "--ttl-seconds=5",
        ]
    )
    endpoints_main(
        [
            "wait",
            f"--root={tmp_path}",
            "--name=gateway",
            "--timeout=1",
        ]
    )

    assert capsys.readouterr().out.strip() == "http://gateway.example:8080"


def test_endpoint_fire_cli_accepts_decoded_command_json(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    endpoints_main(
        [
            "run",
            f"--root={tmp_path}",
            "--name=gateway",
            "--uri=http://gateway.example:8080",
            f"--command-json={json.dumps([sys.executable, '-c', 'pass'])}",
            "--healthcheck=none",
            "--ttl-seconds=1",
            "--refresh-interval=0.1",
            "--publisher-id=cli-supervisor-test",
        ]
    )

    assert (
        capsys.readouterr().out.strip()
        == "LITEREGISTRY_ENDPOINT_GATEWAY=http://gateway.example:8080"
    )


def test_endpoint_supervisor_accepts_empty_non_executable_arguments(
    tmp_path: Path,
) -> None:
    run(
        str(tmp_path),
        "redis",
        "redis://127.0.0.1:6379",
        [sys.executable, "-c", "import time; time.sleep(0.1)", ""],
        healthcheck="none",
        ttl_seconds=1,
        refresh_interval=0.1,
    )
