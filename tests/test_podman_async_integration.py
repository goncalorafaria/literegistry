"""New client + actual buffered worker implementation + real local subprocesses.

The Podman executable is a shim: this does not claim fleet deployment coverage.
"""
import asyncio
import json
import os
from pathlib import Path
import shlex
import time

from aiohttp import web
import pytest

from literegistry_tool_client import PodmanAsyncClient, PodmanStreamError
from literegistry.services.podman_streaming import CommandStreamError
from tests.test_podman_streaming import backend, CID


def test_buffered_writer_returns_intermediate_output_before_exit(backend, tmp_path):
    async def scenario():
        async def handshake(request):
            return web.json_response({'affinity_id': CID, 'container_id': CID})

        async def start(request):
            body = await request.json()
            assert body.pop('affinity_id') == CID
            body.pop('service')
            try:
                return web.json_response(await backend.streams.start(CID, **body), status=202)
            except CommandStreamError as exc:
                return web.json_response({'detail': exc.detail}, status=exc.status)

        async def output(request):
            q = request.query
            try:
                result = await backend.streams.output(CID, command_id=q['command_id'],
                    cursor=int(q['cursor']), wait_ms=int(q['wait_ms']), max_chunks=64)
                return web.json_response(result)
            except CommandStreamError as exc:
                return web.json_response({'detail': exc.detail}, status=exc.status)

        async def close(request):
            await backend.streams.shutdown()
            return web.json_response({'removed': True})

        app = web.Application()
        app.router.add_post('/affinity/handshake', handshake)
        app.router.add_post('/affinity/podman/start', start)
        app.router.add_get('/affinity/podman/output', output)
        app.router.add_post('/affinity/close', close)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, '127.0.0.1', 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        observations = []
        try:
            async with PodmanAsyncClient(f'http://127.0.0.1:{port}') as client:
                began = time.monotonic()
                # Six writes, five-second gaps, buffered stdout read before exit.
                job = await client.start(command=f'for i in 1 2 3 4 5 6; do printf "%s\\n" "$i" > {shlex.quote(str(tmp_path))}/"$i"; printf "%s\\n" "$i"; sleep 5; done', timeout=40, workdir='/')
                assert time.monotonic() - began < 2 and not job.done
                with pytest.raises(PodmanStreamError) as error:
                    await client.start(command='printf second')
                assert error.value.code == 'session_busy'
                while not job.done:
                    batch = await job.read(wait_ms=1000)
                    if batch['stdout']:
                        observations.append({'elapsed_seconds': round(time.monotonic()-began, 3),
                                             'stdout': batch['stdout'], 'running': batch['running'],
                                             'files': [i for i in range(1,7) if (tmp_path/str(i)).exists()]})
                final = await job.wait()
                assert final['success'] and final['stdout'] == ''
                assert [o['stdout'] for o in observations] == [str(i)+'\n' for i in range(1,7)]
                assert all(o['running'] for o in observations)
                assert [o['files'] for o in observations] == [list(range(1,i+1)) for i in range(1,7)]
                record = {'ok': True, 'environment': 'real worker stream implementation and subprocesses; Podman shim, not live fleet',
                          'observations': observations, 'final': final, 'second_command': '409 session_busy'}
                if os.environ.get('PODMAN_ASYNC_PROBE_REPORT'):
                    Path(os.environ['PODMAN_ASYNC_PROBE_REPORT']).write_text(json.dumps(record,indent=2)+'\n')
        finally:
            await backend.streams.shutdown()
            await runner.cleanup()
    asyncio.run(scenario())
