import asyncio
import pytest

from literegistry_tool_client import PodmanAsyncClient, PodmanStreamingUnavailable, PodmanStreamError
from .test_podman_streaming import gateway


def test_start_returns_handle_without_reading_and_reads_only_new_output():
    async def scenario():
        async with gateway() as (url, calls):
            async with PodmanAsyncClient(url) as client:
                job = await client.start(command='build')
                assert calls['polls'] == [] and not job.done
                first = await job.read()
                assert first['stdout'] == 'first' and first['stderr'] == 'warning'
                assert not first['done']
                final = await job.wait()
                assert final['stdout'] == 'last' and final['stderr'] == ''
                assert final['exit_code'] == 7 and not final['output_truncated']
                assert job.done
                assert (await job.read())['stdout'] == ''
                assert (await job.wait())['stdout'] == ''
            assert calls['closes'] == 1
            assert (await job.read())['exit_code'] == 7
    asyncio.run(scenario())


def test_wait_bounds_local_memory_but_drains_and_reports_truncation():
    async def scenario():
        async with gateway() as (url, calls):
            async with PodmanAsyncClient(url) as client:
                job = await client.start(command='large-output')
                result = await job.wait(max_output_chars=7)
                assert result['stdout'] == 'first' and result['stderr'] == 'wa'
                assert result['dropped_output_chars'] == 9
                assert result['output_truncated'] and result['exit_code'] == 7
                assert [cursor for _, cursor in calls['polls']] == [0, 2, 3]
    asyncio.run(scenario())


def test_batches_consumes_incrementally_and_close_drains_accepted_work():
    async def scenario():
        async with gateway() as (url, calls):
            async with PodmanAsyncClient(url) as client:
                job = await client.start(command='one')
                batches = [b async for b in job.batches()]
                assert [b['stdout'] for b in batches] == ['first', 'last']
                pending = await client.start(command='two')
                assert not pending.done
            assert pending.done and calls['closes'] == 1
            assert (await pending.wait())['exit_code'] == 7
    asyncio.run(scenario())


@pytest.mark.parametrize('status,code', [(404,'missing'),(409,'session_busy'),(410,'output_expired')])
def test_errors_are_explicit_no_execute_fallback_or_resubmission(status, code):
    async def scenario():
        async with gateway() as (url, calls):
            async with PodmanAsyncClient(url) as client:
                requests = []
                async def fail(method, endpoint, payload):
                    requests.append(endpoint)
                    raise PodmanStreamError(status, {'detail': {'code': code}})
                client._client._stream_request = fail
                kind = PodmanStreamingUnavailable if status == 404 else PodmanStreamError
                with pytest.raises(kind) as raised:
                    await client.start(command='write-once')
                assert raised.value.code == ('streaming_unavailable' if status == 404 else code)
                assert requests == ['start']
    asyncio.run(scenario())


def test_failed_read_keeps_cursor_and_cleanup_still_attempted():
    async def scenario():
        async with gateway() as (url, calls):
            client = PodmanAsyncClient(url)
            await client.open()
            job = await client.start(command='one')
            original = client._client._stream_request
            seen = []
            async def fail(method, endpoint, payload):
                if endpoint == 'output':
                    seen.append(payload['cursor'])
                    raise PodmanStreamError(410, {'detail': {'code': 'output_expired'}})
                return await original(method, endpoint, payload)
            client._client._stream_request = fail
            for _ in range(2):
                with pytest.raises(PodmanStreamError):
                    await job.read()
            with pytest.raises(PodmanStreamError):
                await client.close()
            assert seen == [0, 0, 0] and calls['closes'] == 1 and not client.opened
    asyncio.run(scenario())


@pytest.mark.parametrize("wait_ms", [-1, 10001, 30000, True, 0.5])
def test_invalid_wait_rejected_before_network(wait_ms):
    from literegistry_tool_client.podman_async import _validate_wait
    with pytest.raises(ValueError):
        _validate_wait(wait_ms)


def test_concurrent_readers_consume_each_chunk_once():
    async def scenario():
        async with gateway() as (url, calls):
            async with PodmanAsyncClient(url) as client:
                job = await client.start(command="read-race")
                batches = await asyncio.gather(*(job.read() for _ in range(4)))
                assert "".join(b["stdout"] for b in batches) == "firstlast"
                assert "".join(b["stderr"] for b in batches) == "warning"
                assert (await job.wait())["exit_code"] == 7
                assert [cursor for _, cursor in calls["polls"]] == [0, 2, 3]
    asyncio.run(scenario())
