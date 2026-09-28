import asyncio
import json

import aiohttp
import pytest

from literegistry_tool_client import BotbookClient, ToolClient


class Response:
    def __init__(self, status, body):
        self.status, self.body = status, body

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def text(self):
        return json.dumps(self.body)


class Session:
    def __init__(self, *results):
        self.results = list(results)
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return Response(*result)


def test_session_lifecycle_payloads_and_pagination():
    async def check():
        session = Session(
            (200, {'affinity_id': 'session'}),
            (200, {'message': {'id': 'session:1'}}),
            (200, {'message': {'id': 'session:2'}}),
            (200, {'messages': [{'id': 'session:2'}], 'has_more': True}),
            (200, {'closed': True}),
        )
        client = BotbookClient('http://gateway/', user_id='A', http_session=session)
        assert isinstance(client, ToolClient)
        assert not client.started
        async with client:
            assert client.affinity_id == 'session'
            assert (await client.post('hello'))['message']['id'] == 'session:1'
            await client.dm('B', 'private')
            page = await client.get_unread(limit=5)
            assert page['has_more'] is True
        assert not client.started
        assert await client.close() is None
        assert [url for url, _ in session.calls] == [
            'http://gateway/affinity/handshake',
            *['http://gateway/affinity/botbook'] * 3,
            'http://gateway/affinity/close',
        ]
        assert [kwargs['json'] for _, kwargs in session.calls] == [
            {'service': 'botbook'},
            {'service': 'botbook', 'affinity_id': 'session', 'user_id': 'A',
             'action': 'post_general', 'text': 'hello', 'board': 'general'},
            {'service': 'botbook', 'affinity_id': 'session', 'user_id': 'A',
             'action': 'post_direct', 'recipient_id': 'B', 'text': 'private'},
            {'service': 'botbook', 'affinity_id': 'session', 'user_id': 'A',
             'action': 'get_unread', 'limit': 5},
            {'service': 'botbook', 'affinity_id': 'session'},
        ]
    asyncio.run(check())


def test_attached_participant_leaves_session_open_unless_explicitly_closed():
    async def check():
        session = Session((200, {'messages': [], 'has_more': False}), (200, {'closed': True}))
        bob = BotbookClient(user_id='B', affinity_id='shared', http_session=session)
        async with bob:
            await bob.get_unread()
        assert bob.started and len(session.calls) == 1
        assert session.calls[0][1]['json']['user_id'] == 'B'
        await bob.close()
        assert not bob.started and len(session.calls) == 2
    asyncio.run(check())


@pytest.mark.parametrize('result', [
    (503, {'error': 'unknown execution outcome'}),
    (410, {'error': 'session expired'}),
    aiohttp.ServerDisconnectedError('response lost'),
    asyncio.TimeoutError(),
])
@pytest.mark.parametrize('operation', ['post_general', 'get_unread'])
def test_side_effecting_requests_are_never_replayed(result, operation):
    async def check():
        session = Session(result)
        client = BotbookClient(user_id='A', affinity_id='shared', http_session=session)
        with pytest.raises(RuntimeError):
            if operation == 'post_general':
                await client.post('once')
            else:
                await client.get_unread()
        assert len(session.calls) == 1
    asyncio.run(check())


def test_invalid_handshake_and_unconfirmed_close_preserve_local_state():
    async def check():
        session = Session((200, {}), (200, {'affinity_id': 's'}), (200, {'closed': False}))
        client = BotbookClient(user_id='A', http_session=session)
        with pytest.raises(RuntimeError, match='affinity_id'):
            await client.start()
        assert not client.started
        await client.start()
        with pytest.raises(RuntimeError, match='deletion'):
            await client.close()
        assert client.affinity_id == 's'
        with pytest.raises(RuntimeError, match='already'):
            await client.start()
        assert len(session.calls) == 3
    asyncio.run(check())


def test_operations_require_session_and_validate_inputs_before_sending():
    async def check():
        session = Session()
        client = BotbookClient(user_id='A', http_session=session)
        with pytest.raises(RuntimeError, match='not started'):
            await client.get_unread()
        for payload in [
            {'action': 'unknown'},
            {'action': 'post_general', 'text': ''},
            {'action': 'post_direct', 'text': 'hi'},
            {'action': 'post_general', 'text': 'hi', 'recipient_id': 'B'},
            {'action': 'get_unread', 'limit': 0},
            {'action': 'get_unread', 'limit': True},
            {'action': 'get_unread', 'text': 'wrong'},
            {'action': 'post_general', 'text': 'hi', 'board': ''},
            {'action': 'get_unread', 'board': 'x' * 257},
            {'action': 'post_direct', 'text': 'hi', 'recipient_id': 'B', 'board': 'research'},
        ]:
            with pytest.raises(ValueError):
                await client.execute(**payload)
        assert session.calls == []
    asyncio.run(check())


def test_body_exception_survives_failed_context_cleanup():
    async def check():
        session = Session((200, {'affinity_id': 's'}), (503, {}))
        client = BotbookClient(user_id='A', http_session=session)
        with pytest.raises(ValueError, match='original'):
            async with client:
                raise ValueError('original')
        assert client.affinity_id == 's'
        assert len(session.calls) == 2
    asyncio.run(check())


def test_named_board_is_forwarded_for_posts_and_reads():
    async def check():
        session = Session((200, {'message': {'board': 'research'}}), (200, {'messages': []}))
        client = BotbookClient(user_id='A', affinity_id='shared', http_session=session)
        await client.post('finding', board='research')
        await client.get_unread(board='research')
        assert [kwargs['json']['board'] for _, kwargs in session.calls] == ['research', 'research']
        assert session.calls[0][1]['json']['text'] == 'finding'
    asyncio.run(check())
