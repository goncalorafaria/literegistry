from __future__ import annotations

import asyncio
import inspect
from typing import Any

from aiohttp import web
import pytest

from literegistry_podman_client import cli
from literegistry_podman_client import (
    CommandResult,
    PodmanCommandError,
    PodmanContainerLostError,
    PodmanClient,
    PodmanGatewayError,
    PodmanSession,
)


class RecordingClient(PodmanClient):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__("http://gateway.example:8080", **kwargs)
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.handshake_count = 0

    async def _post(
        self, endpoint: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        self.calls.append((endpoint, dict(payload)))
        await asyncio.sleep(0)
        if endpoint == "affinity/handshake":
            self.handshake_count += 1
            affinity_id = chr(ord("a") + self.handshake_count - 1) * 64
            return {
                "container_id": affinity_id,
                "affinity_id": affinity_id,
                "instance_id": f"podman-{self.handshake_count}",
                "client_id": payload.get("client_id"),
                "image": payload.get("image") or "ubuntu:24.04",
            }
        if endpoint == "affinity/podman":
            return {
                "container_id": payload["affinity_id"],
                "affinity_id": payload["affinity_id"],
                "stdout": "ai2 hello\n",
                "stderr": "",
                "success": True,
                "exit_code": 0,
                "execution_time": 0.01,
                "timed_out": False,
                "stdout_truncated": True,
                "stderr_truncated": False,
            }
        if endpoint == "affinity/close":
            return {
                "container_id": payload["affinity_id"],
                "affinity_id": payload["affinity_id"],
                "removed": True,
            }
        raise AssertionError(endpoint)


def test_all_lifecycle_operations_are_async() -> None:
    for name in ("open", "handshake", "execute", "close", "aclose"):
        assert inspect.iscoroutinefunction(getattr(PodmanClient, name))
    assert inspect.iscoroutinefunction(PodmanSession.execute)
    assert inspect.iscoroutinefunction(PodmanSession.close)


def test_handshake_execute_and_idempotent_close() -> None:
    async def scenario() -> None:
        client = RecordingClient(workdir="/tmp")
        session = await client.handshake(
            image="python:3.12",
            client_id="trajectory-1",
        )
        result = await session.execute("cat hello.txt", check=True)
        closed = await session.close()

        assert result.stdout == "ai2 hello\n"
        assert result.stdout_truncated is True
        assert result.stderr_truncated is False
        assert closed is not None and closed["removed"] is True
        assert await session.close() is None
        assert [endpoint for endpoint, _ in client.calls] == [
            "affinity/handshake",
            "affinity/podman",
            "affinity/close",
        ]
        assert client.calls[0][1] == {
            "service": "podman",
            "image": "python:3.12",
            "client_id": "trajectory-1",
        }
        assert client.calls[1][1]["affinity_id"] == "a" * 64
        assert client.calls[1][1]["workdir"] == "/tmp"

    asyncio.run(scenario())


def test_session_context_closes_after_user_exception() -> None:
    async def scenario() -> None:
        client = RecordingClient()
        session = await client.handshake()
        with pytest.raises(RuntimeError, match="trajectory failed"):
            async with session:
                raise RuntimeError("trajectory failed")

        assert session.closed is True
        assert [endpoint for endpoint, _ in client.calls] == [
            "affinity/handshake",
            "affinity/close",
        ]

    asyncio.run(scenario())


def test_shared_client_uses_explicit_ids_for_concurrent_sessions() -> None:
    async def scenario() -> None:
        client = RecordingClient()
        first, second = await asyncio.gather(
            client.handshake(client_id="one"),
            client.handshake(client_id="two"),
        )
        results = await asyncio.gather(
            first.execute("echo first"),
            second.execute("echo second"),
        )

        assert first.affinity_id != second.affinity_id
        assert results[0].affinity_id == first.affinity_id
        assert results[1].affinity_id == second.affinity_id
        await asyncio.gather(first.close(), second.close())

    asyncio.run(scenario())


def test_http_pool_open_and_aclose() -> None:
    async def scenario() -> None:
        client = PodmanClient("http://gateway.example:8080")
        assert not client.is_open
        assert await client.open() is client
        assert client.is_open
        await client.aclose()
        assert not client.is_open

    asyncio.run(scenario())


def test_command_error_is_opt_in() -> None:
    result = CommandResult(
        container_id="b" * 64,
        affinity_id="b" * 64,
        stdout="",
        stderr="boom",
        success=False,
        exit_code=7,
        execution_time=0.2,
        timed_out=False,
    )

    with pytest.raises(PodmanCommandError) as exc_info:
        result.check_returncode()

    assert exc_info.value.result is result
    assert "code 7" in str(exc_info.value)


def test_invalid_command_response_is_rejected() -> None:
    with pytest.raises(PodmanGatewayError, match="invalid Podman"):
        CommandResult.from_payload({"success": True})


def test_affinity_owner_loss_is_a_terminal_container_error() -> None:
    async def scenario() -> None:
        response = {
            "error": "strict affinity server is no longer registered",
            "code": "affinity_owner_lost",
            "recoverable": False,
        }

        async def owner_lost(_request):
            return web.json_response(response, status=410)

        app = web.Application()
        app.router.add_post("/affinity/podman", owner_lost)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        assert site._server is not None
        port = site._server.sockets[0].getsockname()[1]
        client = PodmanClient(f"http://127.0.0.1:{port}")
        try:
            with pytest.raises(PodmanContainerLostError) as exc_info:
                await client.execute("a" * 64, "echo hello")
        finally:
            await client.aclose()
            await runner.cleanup()

        error = exc_info.value
        assert error.status_code == 410
        assert error.response == response
        assert error.recoverable is False
        assert "container died" in str(error)
        assert "cannot be recovered" in str(error)

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"gateway_url": "gateway.example:8080"}, "absolute HTTP"),
        ({"gateway_url": ""}, "absolute HTTP"),
        ({"gateway_url": "http://gateway", "service": ""}, "service"),
        ({"gateway_url": "http://gateway", "workdir": ""}, "workdir"),
        ({"gateway_url": "http://gateway", "request_timeout": 0}, "request_timeout"),
    ],
)
def test_invalid_configuration(kwargs: dict[str, Any], error: str) -> None:
    with pytest.raises((TypeError, ValueError), match=error):
        PodmanClient(**kwargs)


def test_mirror_uses_same_gateway_root() -> None:
    client = PodmanClient("https://gateway.example:8443/")
    assert client.gateway_url == "https://gateway.example:8443"
    assert client.mirror_url == client.gateway_url


def test_fire_cli_dispatches_ai2_hello(monkeypatch) -> None:
    calls = []

    async def fake_run(gateway, image, workdir, client_id):
        calls.append((gateway, image, workdir, client_id))

    monkeypatch.setattr(cli, "_run", fake_run)
    cli.main(
        [
            "ai2-hello",
            "--gateway=http://gateway.example:8080",
            "--image=ubuntu:24.04",
            "--workdir=/work",
            "--client-id=fire-test",
        ]
    )

    assert calls == [
        ("http://gateway.example:8080", "ubuntu:24.04", "/work", "fire-test")
    ]
