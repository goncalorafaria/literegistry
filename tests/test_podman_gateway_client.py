import asyncio
import importlib.util
import inspect
from pathlib import Path
import sys

import pytest


CLIENT_PATH = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "podman_gateway_client.py"
)
SPEC = importlib.util.spec_from_file_location("podman_gateway_client", CLIENT_PATH)
assert SPEC is not None and SPEC.loader is not None
client_module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = client_module
SPEC.loader.exec_module(client_module)


class RecordingClient(client_module.PodmanGatewayClient):
    def __init__(self):
        super().__init__("http://gateway.example:8080")
        self.calls = []
        self.handshake_count = 0

    async def _post(self, endpoint, payload):
        self.calls.append((endpoint, dict(payload)))
        await asyncio.sleep(0)
        if endpoint == "affinity/handshake":
            self.handshake_count += 1
            affinity_id = chr(ord("a") + self.handshake_count - 1) * 64
            return {
                "container_id": affinity_id,
                "affinity_id": affinity_id,
                "instance_id": "podman-2",
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
            }
        if endpoint == "affinity/close":
            return {
                "container_id": payload["affinity_id"],
                "affinity_id": payload["affinity_id"],
                "removed": True,
            }
        raise AssertionError(endpoint)


def test_open_handshake_execute_and_close_are_async():
    for name in ("open", "handshake", "execute", "close", "aclose"):
        assert inspect.iscoroutinefunction(
            getattr(client_module.PodmanGatewayClient, name)
        )
    assert inspect.iscoroutinefunction(client_module.PodmanSession.execute)
    assert inspect.iscoroutinefunction(client_module.PodmanSession.close)


def test_client_performs_async_lifecycle_and_idempotent_close():
    async def scenario():
        client = RecordingClient()
        session = await client.handshake(
            image="python:3.12",
            client_id="trajectory-1",
        )
        result = await session.execute(
            "cat /workspace/hello.txt",
            check=True,
        )
        closed = await session.close()

        assert result.stdout == "ai2 hello\n"
        assert closed["removed"] is True
        assert await session.close() is None
        assert [endpoint for endpoint, _ in client.calls] == [
            "affinity/handshake",
            "affinity/podman",
            "affinity/close",
        ]
        assert client.calls[1][1]["affinity_id"] == "a" * 64
        assert client.calls[1][1]["workdir"] == "/workspace"

    asyncio.run(scenario())


def test_async_context_manager_closes_after_user_exception():
    async def scenario():
        client = RecordingClient()
        podman_session = await client.handshake()
        with pytest.raises(RuntimeError, match="trajectory failed"):
            async with podman_session:
                raise RuntimeError("trajectory failed")

        assert podman_session.closed is True
        assert [endpoint for endpoint, _ in client.calls] == [
            "affinity/handshake",
            "affinity/close",
        ]

    asyncio.run(scenario())


def test_shared_client_runs_sessions_concurrently_with_explicit_ids():
    async def scenario():
        client = RecordingClient()
        first, second = await asyncio.gather(
            client.handshake(client_id="one"),
            client.handshake(client_id="two"),
        )
        first_result, second_result = await asyncio.gather(
            client.execute(first.affinity_id, "echo first"),
            client.execute(second.affinity_id, "echo second"),
        )

        assert first.affinity_id != second.affinity_id
        assert first_result.affinity_id == first.affinity_id
        assert second_result.affinity_id == second.affinity_id
        await asyncio.gather(first.close(), second.close())

    asyncio.run(scenario())


def test_client_open_and_aclose_manage_aiohttp_pool():
    async def scenario():
        client = client_module.PodmanGatewayClient("http://gateway.example:8080")
        assert not client.is_open
        opened = await client.open()
        assert opened is client
        assert client.is_open
        await client.aclose()
        assert not client.is_open

    asyncio.run(scenario())


def test_nonzero_result_raises_only_when_checked():
    result = client_module.CommandResult(
        container_id="b" * 64,
        affinity_id="b" * 64,
        stdout="",
        stderr="boom",
        success=False,
        exit_code=7,
        execution_time=0.2,
        timed_out=False,
    )

    with pytest.raises(client_module.PodmanCommandError) as exc_info:
        result.check_returncode()

    assert exc_info.value.result is result
    assert "code 7" in str(exc_info.value)
