import asyncio
from contextlib import asynccontextmanager

from aiohttp import web
import pytest

from literegistry_tool_client import PodmanExecutionClient, PodmanCommand, PodmanStreamError


@asynccontextmanager
async def gateway():
    calls = {"handshakes": 0, "starts": [], "polls": [], "closes": 0}

    async def handshake(request):
        calls["handshakes"] += 1
        return web.json_response({"affinity_id": "a" * 64, "container_id": "a" * 64})

    async def start(request):
        body = await request.json()
        assert body["affinity_id"] == "a" * 64 and body["service"] == "podman"
        calls["starts"].append(body)
        return web.json_response({"command_id": f"{len(calls['starts']):032x}"}, status=202)

    async def output(request):
        params = request.query
        assert request.method == "GET" and not await request.read()
        assert params["affinity_id"] == "a" * 64 and params["service"] == "podman"
        cursor = int(params["cursor"])
        calls["polls"].append((params["command_id"], cursor))
        chunks = [
            {"cursor": 0, "stream": "stdout", "data": "first"},
            {"cursor": 1, "stream": "stderr", "data": "warning"},
        ] if cursor == 0 else (
            [{"cursor": 2, "stream": "stdout", "data": "last"}] if cursor == 2 else []
        )
        return web.json_response({
            "chunks": chunks, "next_cursor": 2 if cursor == 0 else 3,
            "done": cursor > 0, "running": cursor == 0, "has_more": False,
            "exit_code": 7 if cursor else None, "success": False if cursor else None,
            "timed_out": False, "error": None,
        })

    async def close(request):
        calls["closes"] += 1
        return web.json_response({"removed": True})

    app = web.Application()
    app.router.add_post("/affinity/handshake", handshake)
    app.router.add_post("/affinity/podman/start", start)
    app.router.add_get("/affinity/podman/output", output)
    app.router.add_post("/affinity/close", close)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    try:
        yield f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}", calls
    finally:
        await runner.cleanup()


def test_command_handle_hides_ids_and_manages_final_acknowledgement():
    async def scenario():
        async with gateway() as (url, calls):
            async with PodmanExecutionClient(url) as client:
                job = await client.start_command(command="build", merge_output=True)
                assert isinstance(job, PodmanCommand)
                chunks = [chunk async for chunk in job.stream()]
                assert chunks == [{"stream": "stdout", "data": "first"},
                                  {"stream": "stderr", "data": "warning"},
                                  {"stream": "stdout", "data": "last"}]
                assert (await job.wait())["exit_code"] == 7
                assert calls["polls"] == [("1".zfill(32), c) for c in (0, 2, 3)]
                assert calls["starts"][0]["merge_output"] is True
                assert len(calls["starts"][0]["request_id"]) == 32
            # The result belongs to this command, even after closing its session.
            assert (await job.wait())["success"] is False
            assert calls["handshakes"] == calls["closes"] == 1
    asyncio.run(scenario())


def test_read_wait_and_direct_stream_need_no_protocol_fields():
    async def scenario():
        async with gateway() as (url, calls):
            async with PodmanExecutionClient(url) as client:
                job = await client.start_command(command="build")
                batch = await job.read()
                assert batch["stdout"] == "first" and batch["stderr"] == "warning"
                assert not {"cursor", "command_id", "affinity_id", "next_cursor"} & batch.keys()
                assert (await job.wait())["exit_code"] == 7
                chunks = [chunk async for chunk in client.stream(command="again")]
                assert "".join(c["data"] for c in chunks) == "firstwarninglast"
                assert calls["starts"][0]["request_id"] != calls["starts"][1]["request_id"]
    asyncio.run(scenario())


def test_expired_cursor_keeps_session_and_failed_reads_do_not_advance():
    class Client(PodmanExecutionClient):
        def __init__(self):
            super().__init__()
            self._affinity_id = "a" * 64
            self.reads = []

        async def _stream_request(self, method, endpoint, payload):
            if endpoint == "start":
                return {"command_id": "c" * 32}
            self.reads.append(payload["cursor"])
            raise PodmanStreamError(410, {"detail": {"code": "output_expired"}})

    async def scenario():
        client = Client()
        job = await client.start_command(command="build")
        for _ in range(2):
            with pytest.raises(PodmanStreamError):
                await job.read()
        assert client.reads == [0, 0] and client.started
        client._affinity_id = "b" * 64
        with pytest.raises(RuntimeError, match="closed"):
            await job.read()
        assert client.reads == [0, 0]
    asyncio.run(scenario())
