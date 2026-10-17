import asyncio

from aiohttp import web
import pytest

from literegistry_podman_client import (
    PodmanClient, PodmanCommand, PodmanContainerLostError, PodmanGatewayError,
    PodmanSession,
)


def test_client_starts_and_polls_get_until_final_acknowledgement():
    async def scenario():
        starts, cursors = [], []

        async def start(request):
            starts.append(await request.json())
            return web.json_response({"command_id": "c" * 32, "cursor": 0}, status=202)

        async def output(request):
            assert request.method == "GET" and not await request.read()
            assert request.query["affinity_id"] == "a" * 64
            assert request.query["command_id"] == "c" * 32
            assert request.query["service"] == "podman"
            cursor = int(request.query["cursor"])
            cursors.append(cursor)
            return web.json_response({
                "next_cursor": cursor + 1 if cursor < 2 else cursor,
                "chunks": [{"stream": "stdout", "data": str(cursor)}] if cursor < 2 else [],
                "done": cursor >= 1, "has_more": False,
                "exit_code": 0 if cursor >= 1 else None,
            })

        app = web.Application()
        app.router.add_post("/affinity/podman/start", start)
        app.router.add_get("/affinity/podman/output", output)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        try:
            async with PodmanClient(f"http://127.0.0.1:{port}") as client:
                session = PodmanSession(client=client, affinity_id="a" * 64,
                                        container_id="a" * 64, instance_id="replica-a", image="test")
                job = await session.start_command("echo hello", merge_output=True,
                                                  request_id="retry-1", timeout=20)
                assert isinstance(job, PodmanCommand)
                batches = [batch async for batch in job.output()]
                assert len(batches) == 2
                assert cursors == [0, 1, 2]
                assert starts[0]["request_id"] == "retry-1"
                assert starts[0]["merge_output"] is True
                assert starts[0]["timeout"] == 20
                assert (await job.wait())["exit_code"] == 0
                assert cursors == [0, 1, 2]  # Cached result: no extra request.
                chunks = [chunk async for chunk in session.stream("echo direct")]
                assert chunks == [{"stream": "stdout", "data": "0"},
                                  {"stream": "stdout", "data": "1"}]
                assert len(starts[1]["request_id"]) == 32
                reader = await session.start_command("echo batches")
                batch = await reader.read()
                assert batch["stdout"] == "0" and batch["stderr"] == ""
                assert not {"cursor", "command_id", "affinity_id", "next_cursor"} & batch.keys()
                assert (await reader.wait())["exit_code"] == 0
        finally:
            await runner.cleanup()
    asyncio.run(scenario())


def test_final_status_survives_expiry_during_final_acknowledgement():
    class Client(PodmanClient):
        calls = 0

        async def _post(self, endpoint, payload):
            return {"command_id": "c" * 32}

        async def command_output(self, *args, **kwargs):
            self.calls += 1
            if self.calls > 1:
                raise PodmanGatewayError("command expired", status_code=410,
                                         response={"detail": {"code": "command_expired"}})
            return {"next_cursor": 0, "chunks": [], "done": True, "has_more": False,
                    "exit_code": 0, "success": True, "timed_out": False, "error": None}

    async def scenario():
        client = Client("http://gateway")
        job = await client.start_command("a" * 64, "true")
        assert (await job.wait())["success"] is True
        assert (await job.wait())["success"] is True
        assert client.calls == 2
    asyncio.run(scenario())


def test_expired_output_does_not_lose_session_or_advance_cursor():
    class Client(PodmanClient):
        async def _post(self, endpoint, payload):
            return {"command_id": "c" * 32}

        async def command_output(self, *args, **kwargs):
            raise PodmanGatewayError("output expired", status_code=410,
                                     response={"detail": {"code": "output_expired", "oldest_cursor": 5}})

    async def scenario():
        client = Client("http://gateway")
        session = PodmanSession(client=client, affinity_id="a" * 64, container_id="a" * 64,
                                instance_id="worker", image="test")
        job = await session.start_command("echo hi")
        with pytest.raises(PodmanGatewayError) as error:
            await job.poll()
        assert not isinstance(error.value, PodmanContainerLostError)
        assert job.cursor == 0 and not session.closed
    asyncio.run(scenario())
