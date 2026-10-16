"""Exercise live subprocess output, lifecycle locking, and pinned HTTP polling.

The shim executes the actual Bash/deadline wrapper locally. Podman/CRIU are
not needed; container namespace behavior remains a deployment smoke check.
"""

import asyncio
import sys

import httpx
import pytest

from literegistry.affinity import StrictAffinityBindingStore
from literegistry.gateway import Gateway
from literegistry.gateway.affinity import StrictAffinityGateway
from literegistry.http import HTTPResponseError
from literegistry.kvstore import FileSystemKVStore
from literegistry.services.podman import (
    CompletedPodmanCommand, PodmanAffinityConfig, PodmanAffinityService,
    PodmanSessionBackend, create_app,
)
from literegistry.services.podman_streaming import CommandStreamError


CID = "a" * 64
OTHER_CID = "b" * 64


@pytest.fixture
def backend(tmp_path):
    shim = tmp_path / "podman-shim"
    shim.write_text(
        f"#!{sys.executable}\n"
        "import os,sys,subprocess\n"
        "args=sys.argv[1:]\n"
        "i=args.index('--workdir')\n"
        "os.chdir(args[i+1])\n"
        "rc=subprocess.run(args[i+3:]).returncode\n"
        "sys.exit(rc if rc >= 0 else 128-rc)\n"
    )
    shim.chmod(0o755)
    instance = PodmanSessionBackend(PodmanAffinityConfig(podman_binary=str(shim)))
    instance._owned_container_ids.update({CID, OTHER_CID})
    return instance


async def start(backend, command, **options):
    return await backend.streams.start(
        options.pop("container_id", CID), command=command, stdin=options.pop("stdin", ""),
        timeout=options.pop("timeout", 5), workdir="/",
        merge_output=options.pop("merge_output", False),
        request_id=options.pop("request_id", "start-1"), **options,
    )


async def poll(backend, command_id, cursor=0, **options):
    return await backend.streams.output(
        CID, command_id=command_id, cursor=cursor,
        wait_ms=options.pop("wait_ms", 1000),
        max_chunks=options.pop("max_chunks", 64), **options,
    )


async def finish(backend, command_id):
    async def collect():
        cursor = 0
        chunks = []
        while True:
            batch = await poll(backend, command_id, cursor)
            chunks.extend(batch["chunks"])
            cursor = batch["next_cursor"]
            if batch["done"] and not batch["has_more"]:
                return batch, chunks
    return await asyncio.wait_for(collect(), 10)


def test_output_arrives_before_exit_and_ack_releases_bytes(backend, tmp_path):
    async def scenario():
        gate = tmp_path / "continue"
        try:
            job = await start(backend, f"printf first; while ! test -f {gate}; do sleep .02; done; printf second; exit 7")
            first = await poll(backend, job["command_id"])
            assert first["running"] and not first["done"]
            assert "".join(c["data"] for c in first["chunks"]) == "first"
            # Retrying a lost response is possible before advancing the cursor.
            assert (await poll(backend, job["command_id"]))["chunks"] == first["chunks"]
            await poll(backend, job["command_id"], first["next_cursor"], wait_ms=0)
            assert backend.streams.commands[job["command_id"]].buffered_bytes == 0
            with pytest.raises(CommandStreamError) as error:
                await poll(backend, job["command_id"], 0)
            assert error.value.status == 410
            gate.touch()
            await backend.streams.commands[job["command_id"]].task
            final = await poll(backend, job["command_id"], first["next_cursor"])
            assert "".join(c["data"] for c in final["chunks"]) == "second"
            assert final["done"] and final["exit_code"] == 7 and not final["success"]
        finally:
            gate.touch()
            await backend.streams.shutdown()
    asyncio.run(scenario())


@pytest.mark.parametrize("merged", [False, True])
def test_channels_merge_stdin_and_utf8(backend, merged):
    async def scenario():
        try:
            job = await start(backend, "cat; printf '\\342'; sleep .05; printf '\\202\\254'; printf error >&2",
                              stdin="input", merge_output=merged)
            final, chunks = await finish(backend, job["command_id"])
            stdout = "".join(c["data"] for c in chunks if c["stream"] == "stdout")
            stderr = "".join(c["data"] for c in chunks if c["stream"] == "stderr")
            assert stdout == ("input€error" if merged else "input€")
            assert stderr == ("" if merged else "error")
            assert final["success"] and final["exit_code"] == 0
        finally:
            await backend.streams.shutdown()
    asyncio.run(scenario())


def test_duplicate_start_conflict_busy_and_normal_execute_lock(backend, tmp_path):
    async def scenario():
        gate = tmp_path / "finish"
        command = f"while ! test -f {gate}; do sleep .02; done"
        try:
            one, two = await asyncio.gather(start(backend, command), start(backend, command))
            assert one["command_id"] == two["command_id"]
            assert len(backend.streams.commands) == 1
            other = await start(backend, "echo independent", container_id=OTHER_CID)
            await asyncio.wait_for(backend.streams.commands[other["command_id"]].task, 5)
            assert not backend.streams.commands[one["command_id"]].done
            with pytest.raises(CommandStreamError) as error:
                await start(backend, "echo conflict")
            assert error.value.detail["code"] == "request_id_conflict"
            with pytest.raises(CommandStreamError) as error:
                await start(backend, "echo busy", request_id="other")
            assert error.value.detail["code"] == "session_busy"
            normal = asyncio.create_task(backend.execute(CID, "printf normal", workdir="/"))
            await asyncio.sleep(0)
            assert not normal.done()
            gate.touch()
            assert (await asyncio.wait_for(normal, 10)).stdout == b"normal"
        finally:
            gate.touch()
            await backend.streams.shutdown()
    asyncio.run(scenario())


def test_overflow_expiry_capacity_and_session_isolation(backend):
    async def scenario():
        # Retain one completed command, making capacity deterministic.
        object.__setattr__(backend.config, "stream_buffer_bytes", 12)
        object.__setattr__(backend.config, "max_stream_commands", 1)
        object.__setattr__(backend.config, "stream_retention_seconds", 0.1)
        try:
            job = await start(backend, "printf '0123456789abcdefghijklmnopqrstuvwxyz'")
            record = backend.streams.commands[job["command_id"]]
            await record.task
            assert record.buffered_bytes <= 12
            with pytest.raises(CommandStreamError) as error:
                await poll(backend, job["command_id"], 0)
            assert error.value.detail["code"] == "output_expired"
            oldest = error.value.detail["oldest_cursor"]
            tail = await poll(backend, job["command_id"], oldest, max_chunks=1)
            assert tail["chunks"] and tail["has_more"]
            with pytest.raises(CommandStreamError) as error:
                await backend.streams.output(OTHER_CID, command_id=job["command_id"],
                                             cursor=0, wait_ms=0, max_chunks=64)
            assert error.value.status == 404
            with pytest.raises(CommandStreamError) as error:
                await start(backend, "echo full", request_id="full")
            assert error.value.status == 503
            await asyncio.sleep(0.15)
            assert not backend.streams.commands and not backend.streams.requests
            with pytest.raises(CommandStreamError) as error:
                await poll(backend, job["command_id"])
            assert error.value.detail["code"] == "command_expired"
        finally:
            await backend.streams.shutdown()
    asyncio.run(scenario())


def test_chunk_overhead_is_bounded_and_close_reclaims_completed_output(backend):
    async def scenario():
        try:
            job = await start(backend, "printf done")
            record = backend.streams.commands[job["command_id"]]
            await record.task
            for _ in range(2000):
                backend.streams._append(record, "stdout", "x")
            assert len(record.chunks) <= 1024
            assert record.buffered_bytes <= backend.config.stream_buffer_bytes

            async def remove(args, **kwargs):
                assert "rm" in args
                return CompletedPodmanCommand(tuple(args), 0, b"", b"")

            backend._run = remove
            await backend.remove_session(CID)
            assert not backend.streams.commands and not backend.streams.requests
            assert record.expiry.cancelled()
        finally:
            await backend.streams.shutdown()
    asyncio.run(scenario())


def test_deadline_kills_command_and_shutdown_releases_reserved_lock(backend):
    async def scenario():
        try:
            job = await start(backend, "sleep 30", timeout=0.1)
            final, _ = await finish(backend, job["command_id"])
            assert final["timed_out"] and final["exit_code"] == 137
            assert not (await backend._lock_for(CID)).locked()
            # Also cover cancellation before the spawned task's first turn.
            await start(backend, "sleep 30", request_id="shutdown")
            await backend.streams.shutdown()
            assert not (await backend._lock_for(CID)).locked()
        finally:
            await backend.streams.shutdown()
    asyncio.run(scenario())


def test_worker_api_and_gateway_poll_are_pinned_and_uncached(backend, tmp_path):
    class Registry:
        async def models(self, force=False):
            return {"podman": [{"server_id": "worker-a", "uri": "http://worker-a"},
                               {"server_id": "worker-b", "uri": "http://worker-b"}]}

    async def scenario():
        worker = create_app(PodmanAffinityService(backend), None)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=worker),
                                      base_url="http://worker-a") as upstream:
            class Transport:
                calls = []

                async def post(self, service, server_uri, endpoint, payload, retry):
                    self.calls.append((server_uri, endpoint))
                    response = await upstream.post("/" + endpoint, json=payload)
                    if response.status_code >= 400:
                        raise HTTPResponseError(response.status_code, response.json(), endpoint)
                    return response.json()

            transport = Transport()
            bindings = StrictAffinityBindingStore(FileSystemKVStore(tmp_path / "bindings"))
            await bindings.bind("podman", CID, "worker-a", "http://worker-a")
            gateway = Gateway(Registry(), routes=[], strict_affinity=StrictAffinityGateway(
                Registry(), bindings, transport=transport,
            ))
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=gateway.app),
                                          base_url="http://gateway") as client:
                try:
                    response = await client.post("/affinity/podman/start", json={
                        "service": "podman", "affinity_id": CID,
                        "command": "printf hello", "workdir": "/", "request_id": "http-1",
                    })
                    assert response.status_code == 202, response.text
                    job = response.json()
                    params = {"service": "podman", "affinity_id": CID,
                              "command_id": job["command_id"], "wait_ms": 1000}
                    response = await client.get("/affinity/podman/output", params=params)
                    assert response.status_code == 200, response.text
                    assert response.headers["cache-control"] == "no-store"
                    assert response.json()["chunks"][0]["data"] == "hello"
                    direct = await upstream.get("/podman/output", params={
                        k: v for k, v in params.items() if k != "service"
                    })
                    assert direct.status_code == 200
                    bad = await client.get("/affinity/podman/output", params={**params, "cursor": -1})
                    assert bad.status_code == 422
                    expired = await client.get("/affinity/podman/output", params={
                        **params, "command_id": "0" * 32,
                    })
                    assert expired.status_code == 410
                    assert expired.headers["cache-control"] == "no-store"
                    assert all(uri == "http://worker-a" for uri, _ in transport.calls)
                finally:
                    await backend.streams.shutdown()
    asyncio.run(scenario())
