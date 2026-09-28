"""Exercise actual Redis Lua semantics and the gateway's strict owner routing."""
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime
import os
import shutil
import subprocess
import time

import httpx
import pytest
from redis import Redis as SyncRedis
from redis.asyncio import Redis
from fastapi import HTTPException

from literegistry.affinity import StrictAffinityBindingStore
from literegistry.gateway import Gateway
from literegistry.gateway.affinity import StrictAffinityGateway
from literegistry.http import HTTPResponseError
from literegistry.kvstore import FileSystemKVStore
from literegistry.services.botbook_server import (
    BotbookService, DirectRequest, GeneralRequest, UnreadRequest, create_app,
    build_registered_app,
)


@pytest.fixture(scope='module')
def redis_url(tmp_path_factory):
    executable = os.environ.get('REDIS_SERVER') or shutil.which('redis-server')
    if not executable:
        pytest.skip('Set REDIS_SERVER or install redis-server to run Redis integration tests')
    directory = tmp_path_factory.mktemp('botbook-redis')
    sock = directory / 'redis.sock'
    process = subprocess.Popen(
        [executable, '--port', '0', '--unixsocket', str(sock), '--save', '',
         '--appendonly', 'no', '--dir', str(directory)],
        stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
    )
    client = SyncRedis(unix_socket_path=str(sock))
    try:
        for _ in range(100):
            try:
                if client.ping():
                    break
            except Exception:
                time.sleep(0.05)
        else:
            pytest.fail('test Redis did not start')
        yield f'unix://{sock}'
    finally:
        client.close()
        process.terminate()
        process.wait(timeout=5)


def general(session, user, text, board="general"):
    return GeneralRequest(affinity_id=session, action='post_general', user_id=user, text=text, board=board)


def direct(session, user, recipient, text):
    return DirectRequest(affinity_id=session, action='post_direct', user_id=user,
                         recipient_id=recipient, text=text)


def unread(session, user, limit=100, board=None):
    return UnreadRequest(affinity_id=session, action='get_unread', user_id=user, limit=limit, board=board)


def test_visibility_read_tracking_and_pagination(redis_url):
    async def check():
        async with Redis.from_url(redis_url) as redis:
            service = BotbookService(redis)
            session = (await service.start())['affinity_id']
            assert 1799 <= await redis.ttl(f'literegistry:botbook:{{{session}}}') <= 1800
            first = (await service.execute(general(session, 'A', 'hello everyone')))['message']
            second = (await service.execute(direct(session, 'A', 'B', 'private')))['message']
            third = (await service.execute(general(session, 'B', 'hello back')))['message']
            assert len({first['id'], second['id'], third['id']}) == 3
            assert datetime.fromisoformat(first['sent_at']).tzinfo is not None
            page = await service.execute(unread(session, 'B', 1))
            assert page['messages'] == [first] and page['has_more']
            assert (await service.execute(unread(session, 'B')))['messages'] == [second, third]
            assert (await service.execute(unread(session, 'B')))['messages'] == []
            # Late joiners see broadcast history, never other users' DMs.
            assert (await service.execute(unread(session, 'C')))['messages'] == [first, third]
            assert (await service.execute(unread(session, 'A')))['messages'] == [first, third]
            fourth = (await service.execute(direct(session, 'C', 'B', 'new')))['message']
            assert (await service.execute(unread(session, 'B')))['messages'] == [fourth]
    asyncio.run(check())


def test_concurrent_posts_and_reads_are_atomic(redis_url):
    async def check():
        async with Redis.from_url(redis_url) as redis:
            service = BotbookService(redis)
            session = (await service.start())['affinity_id']
            posts = await asyncio.gather(*[
                service.execute(general(session, 'A', str(i))) for i in range(80)
            ])
            pages = await asyncio.gather(*[
                service.execute(unread(session, 'B', 10)) for _ in range(10)
            ])
            ids = [m['id'] for page in pages for m in page['messages']]
            assert len(ids) == len(set(ids)) == 80
            assert set(ids) == {post['message']['id'] for post in posts}
            assert len((await service.execute(unread(session, 'C')))['messages']) == 80
    asyncio.run(check())


def test_session_isolation_owner_close_expiry_and_capacity(redis_url):
    async def check():
        async with Redis.from_url(redis_url) as redis:
            service = BotbookService(redis, instance_id='owner', max_messages=1)
            other = BotbookService(redis, instance_id='other')
            session = (await service.start())['affinity_id']
            separate = (await service.start())['affinity_id']
            await service.execute(general(session, 'A', 'one'))
            assert (await service.execute(unread(separate, 'A')))['messages'] == []
            for request in [general(session, 'A', 'bad'), unread(session, 'A')]:
                with pytest.raises(HTTPException) as error:
                    await other.execute(request)
                assert error.value.status_code == 409
            with pytest.raises(HTTPException):
                await other.close(session)
            with pytest.raises(HTTPException) as error:
                await service.execute(general(session, 'A', 'full'))
            assert error.value.detail['error'] == 'session_message_limit'
            await service.close(session)
            assert not await redis.exists(f'literegistry:botbook:{{{session}}}')
            with pytest.raises(HTTPException) as error:
                await service.execute(unread(session, 'A'))
            assert error.value.status_code == 410
            await redis.pexpire(f'literegistry:botbook:{{{separate}}}', 1)
            await asyncio.sleep(0.02)
            with pytest.raises(HTTPException) as error:
                await service.execute(general(separate, 'A', 'expired'))
            assert error.value.status_code == 410
    asyncio.run(check())


def test_http_validation_and_lifecycle(redis_url, tmp_path):
    async def check():
        app = build_registered_app(redis_url=redis_url, registry=f'file://{tmp_path}')
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url='http://test') as client:
                assert (await client.get('/health')).status_code == 200
                session = (await client.post('/handshake', json={})).json()['affinity_id']
                for payload in [
                    {'action': 'post_direct', 'user_id': 'A', 'text': 'missing recipient'},
                    {'action': 'post_general', 'user_id': '', 'text': 'bad'},
                    {'action': 'get_unread', 'user_id': 'B', 'limit': 0},
                    {'action': 'unknown', 'user_id': 'B'},
                    {'action': 'post_general', 'user_id': 'A', 'text': 'bad', 'board': ''},
                    {'action': 'get_unread', 'user_id': 'B', 'board': 'x' * 257},
                ]:
                    response = await client.post('/botbook', json={**payload, 'affinity_id': session})
                    assert response.status_code == 422
                assert (await client.post('/close', json={'affinity_id': session})).json()['closed']
    asyncio.run(check())


def test_gateway_pins_botbook_and_releases_binding(redis_url, tmp_path):
    async def check():
        async with Redis.from_url(redis_url) as redis:
            services = {f'http://replica-{name}': BotbookService(redis, instance_id=name)
                        for name in ('a', 'b')}

            class Registry:
                def __init__(self):
                    self.records = [dict(server_id=name, uri=uri, metadata={'model_path': 'botbook'})
                                    for name, uri in zip(('a', 'b'), services)]

                async def models(self, force=False):
                    return {'botbook': self.records}

                async def sample_servers(self, service, n, force=False):
                    return [(record['uri'], 1) for record in self.records[:n]]

            class Transport:
                def __init__(self):
                    self.calls = []

                async def post(self, service, uri, endpoint, payload, retry):
                    self.calls.append((uri, endpoint))
                    async with httpx.AsyncClient(transport=httpx.ASGITransport(create_app(services[uri])),
                                                 base_url=uri) as client:
                        response = await client.post('/' + endpoint, json=payload)
                        if response.status_code >= 400:
                            raise HTTPResponseError(response.status_code, response.json(), uri)
                        return response.json()

            registry, transport = Registry(), Transport()
            store = FileSystemKVStore(str(tmp_path))
            bindings = StrictAffinityBindingStore(store)
            strict = StrictAffinityGateway(registry, bindings, transport=transport)
            gateway = Gateway(registry=registry, strict_affinity=strict)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(gateway.app), base_url='http://gateway') as client:
                # Exercise the standalone aiohttp-style client against the real
                # ASGI gateway and Redis service without binding a TCP port.
                from literegistry_tool_client import BotbookClient

                class ClientResponse:
                    def __init__(self, response):
                        self.response = response
                        self.status = response.status_code

                    async def text(self):
                        return self.response.text

                class ClientSession:
                    @asynccontextmanager
                    async def post(self, url, **kwargs):
                        response = await client.post(url, json=kwargs['json'])
                        yield ClientResponse(response)

                alice = BotbookClient('http://gateway', user_id='A', http_session=ClientSession())
                session = (await alice.start())['affinity_id']
                bob = BotbookClient('http://gateway', user_id='B', affinity_id=session,
                                    http_session=ClientSession())
                payload = {'service': 'botbook', 'affinity_id': session}
                registry.records.reverse()  # Changing selection preference cannot migrate this session.
                await alice.post('hello')
                await alice.dm('B', 'private')
                posted = await alice.post('findings', board='research')
                assert posted['message']['board'] == 'research'
                assert [m['text'] for m in (await bob.get_unread(board='research'))['messages']] == ['findings']
                assert [m['text'] for m in (await bob.get_unread())['messages']] == ['hello', 'private']
                assert (await bob.get_unread())['messages'] == []
                assert [m['text'] for m in (await alice.get_unread())['messages']] == ['hello']
                assert [m['text'] for m in (await alice.get_unread(board='research'))['messages']] == ['findings']
                assert {uri for uri, _ in transport.calls} == {'http://replica-a'}
                owner = registry.records.pop()  # b remains live; a is absent.
                count = len(transport.calls)
                response = await client.post('/affinity/botbook', json={**payload, 'action': 'get_unread', 'user_id': 'B'})
                assert response.status_code >= 400
                assert len(transport.calls) == count
                registry.records.append(owner)
                assert (await alice.close())['closed'] is True
                assert await bindings.resolve('botbook', session) is None
                response = await client.post('/affinity/botbook', json={**payload, 'action': 'get_unread', 'user_id': 'B'})
                assert response.status_code == 404
            await store.close()
    asyncio.run(check())


def test_default_inbox_leaves_custom_boards_unread(redis_url):
    async def check():
        async with Redis.from_url(redis_url) as redis:
            service = BotbookService(redis)
            session = (await service.start())['affinity_id']
            messages = []
            for request in [
                general(session, 'A', 'default'),
                general(session, 'A', 'research 1', board='research'),
                direct(session, 'A', 'B', 'private'),
                general(session, 'A', 'research 2', board='research'),
                general(session, 'A', 'planning', board='planning'),
            ]:
                messages.append((await service.execute(request))['message'])
            assert messages[0]['board'] == 'general'
            assert 'board' not in messages[2]
            page = await service.execute(unread(session, 'B', limit=1, board='research'))
            assert page['messages'] == [messages[1]] and page['has_more']
            # The default inbox reads only general posts and personal DMs.
            # Even older custom-board posts must remain unread.
            page = await service.execute(unread(session, 'B'))
            assert page['messages'] == [messages[i] for i in (0, 2)]
            assert not page['has_more']
            assert (await service.execute(unread(session, 'B', board='research')))['messages'] == [messages[3]]
            assert (await service.execute(unread(session, 'B', board='planning')))['messages'] == [messages[4]]
            # Another participant can independently read both boards.
            assert (await service.execute(unread(session, 'C', board='planning')))['messages'] == [messages[4]]
            assert (await service.execute(unread(session, 'C', board='research')))['messages'] == [messages[1], messages[3]]
            assert (await service.execute(unread(session, 'C')))['messages'] == [messages[0]]
            assert (await service.execute(unread(session, 'B', board='new')))['messages'] == []
            new = (await service.execute(general(session, 'A', 'new post', board='new')))['message']
            assert (await service.execute(unread(session, 'B', board='new')))['messages'] == [new]
            assert (await service.execute(unread(session, 'B')))['messages'] == []
    asyncio.run(check())


def test_concurrent_board_and_default_reads_never_duplicate_messages(redis_url):
    async def check():
        async with Redis.from_url(redis_url) as redis:
            service = BotbookService(redis)
            session = (await service.start())['affinity_id']
            posts = await asyncio.gather(*[
                service.execute(general(session, 'A', str(i), board=('general' if i % 2 else 'research')))
                for i in range(40)
            ])
            pages = await asyncio.gather(*[
                service.execute(unread(session, 'B', limit=5, board=board))
                for board in ['research', None, 'general', None] * 5
            ])
            ids = [m['id'] for page in pages for m in page['messages']]
            assert len(ids) == len(set(ids)) == 40
            assert set(ids) == {post['message']['id'] for post in posts}
            await service.close(session)
            assert not await redis.exists(f'literegistry:botbook:{{{session}}}')
    asyncio.run(check())


def test_explicit_general_and_default_inbox_share_post_read_state(redis_url):
    async def check():
        async with Redis.from_url(redis_url) as redis:
            service = BotbookService(redis)
            session = (await service.start())['affinity_id']
            post = (await service.execute(general(session, 'A', 'public')))['message']
            dm = (await service.execute(direct(session, 'A', 'B', 'personal')))['message']
            assert (await service.execute(unread(session, 'B', board='general')))['messages'] == [post]
            assert (await service.execute(unread(session, 'B')))['messages'] == [dm]
            later = (await service.execute(general(session, 'A', 'later')))['message']
            assert (await service.execute(unread(session, 'B')))['messages'] == [later]
            assert (await service.execute(unread(session, 'B', board='general')))['messages'] == []
    asyncio.run(check())
