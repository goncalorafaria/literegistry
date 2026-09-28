"""Redis-backed messaging sessions owned by strict-affinity replicas."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import json
import logging
import secrets
import socket
from typing import Annotated, Literal

import fire
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from redis.asyncio import Redis
import uvicorn

from literegistry import get_kvstore
from literegistry.registry import ServerRegistry

# All state for a session shares one Redis key. Lua makes owner checks, writes,
# read cursors, expiry, and close atomic even across concurrent HTTP requests.
_SCRIPT = """
local key, owner, op = KEYS[1], ARGV[1], ARGV[2]
local ttl = tonumber(ARGV[3])
local data = cjson.decode(ARGV[4])
if op == 'start' then
    if redis.call('EXISTS', key) == 1 then return {'collision'} end
    redis.call('HSET', key, 'owner', owner, 'next', 0)
    redis.call('EXPIRE', key, ttl)
    return {'ok'}
end
local actual = redis.call('HGET', key, 'owner')
if not actual then return {'missing'} end
if actual ~= owner then return {'owner'} end
if op == 'close' then
    redis.call('DEL', key)
    return {'ok'}
end
local last = tonumber(redis.call('HGET', key, 'next'))
if op == 'post_general' or op == 'post_direct' then
    if last >= tonumber(ARGV[5]) then return {'full'} end
    local seq = last + 1
    local now = redis.call('TIME')
    local message = {id=data.affinity_id .. ':' .. seq, user_id=data.user_id,
                     text=data.text, sent_at=tonumber(now[1])+tonumber(now[2])/1000000,
                     kind=op == 'post_general' and 'general' or 'direct'}
    if op == 'post_direct' then
        message.recipient_id = data.recipient_id
    else
        message.board = data.board
    end
    local encoded = cjson.encode(message)
    redis.call('HSET', key, 'next', seq, 'message:' .. seq, encoded)
    redis.call('EXPIRE', key, ttl)
    return {'ok', encoded}
end
-- Keep the former all-board cursor as a read-only floor for existing sessions.
local legacy_cursor = tonumber(redis.call('HGET', key, 'cursor:' .. data.user_id) or '0')
local inbox_field = 'inbox_cursor:' .. data.user_id
local inbox_cursor = tonumber(redis.call('HGET', key, inbox_field) or '0')
local filtered = type(data.board) == 'string'
local function board_field(board)
    return 'board_cursor:' .. cjson.encode({data.user_id, board})
end
local field = filtered and board_field(data.board) or inbox_field
local cursor = math.max(legacy_cursor, tonumber(redis.call('HGET', key, field) or '0'))
local general_cursor = tonumber(redis.call('HGET', key, board_field('general')) or '0')
if filtered and data.board == 'general' then
    cursor = math.max(cursor, inbox_cursor)
end
local result = {'ok', '0'}
local count, scanned = 0, 0
while cursor < last and count < data.limit and scanned < 1000 do
    cursor = cursor + 1
    scanned = scanned + 1
    local encoded = redis.call('HGET', key, 'message:' .. cursor)
    local message = cjson.decode(encoded)
    local visible = false
    if message.kind == 'general' then
        local board = message.board or 'general'
        if filtered then
            visible = board == data.board
        else
            visible = board == 'general' and cursor > general_cursor
        end
    elseif not filtered then
        visible = message.recipient_id == data.user_id
    end
    if visible then
        table.insert(result, encoded)
        count = count + 1
    end
end
redis.call('HSET', key, field, cursor)
redis.call('EXPIRE', key, ttl)
if cursor < last then result[2] = '1' end
return result
"""

Identifier = Annotated[str, Field(min_length=1, max_length=256)]


class SessionRequest(BaseModel):
    affinity_id: Identifier


class GeneralRequest(SessionRequest):
    action: Literal['post_general']
    board: Identifier = 'general'
    user_id: Identifier
    text: str = Field(min_length=1, max_length=65536)


class DirectRequest(SessionRequest):
    action: Literal['post_direct']
    user_id: Identifier
    recipient_id: Identifier
    text: str = Field(min_length=1, max_length=65536)


class UnreadRequest(SessionRequest):
    action: Literal['get_unread']
    board: Identifier | None = None
    user_id: Identifier
    limit: int = Field(default=100, ge=1, le=1000)


Command = Annotated[GeneralRequest | DirectRequest | UnreadRequest, Field(discriminator='action')]


class BotbookService:
    def __init__(self, redis: Redis, *, instance_id: str | None = None,
                 session_ttl_seconds: int = 1800, max_messages: int = 10000):
        if session_ttl_seconds < 1 or max_messages < 1:
            raise ValueError('session TTL and max_messages must be positive')
        self.redis = redis
        self.instance_id = instance_id or secrets.token_hex(16)
        self.session_ttl_seconds = session_ttl_seconds
        self.max_messages = max_messages

    async def _run(self, affinity_id: str, operation: str, payload: dict):
        result = await self.redis.eval(
            _SCRIPT, 1, f'literegistry:botbook:{{{affinity_id}}}',
            self.instance_id, operation, self.session_ttl_seconds,
            json.dumps(payload), self.max_messages,
        )
        result = [item.decode() if isinstance(item, bytes) else item for item in result]
        errors = {
            'missing': (410, 'session_closed_or_expired'),
            'owner': (409, 'affinity_miss'),
            'full': (409, 'session_message_limit'),
            'collision': (409, 'session_id_collision'),
        }
        if result[0] in errors:
            status, error = errors[result[0]]
            raise HTTPException(status, detail={'error': error, 'instance_id': self.instance_id})
        return result

    async def start(self):
        affinity_id = secrets.token_urlsafe(32)
        await self._run(affinity_id, 'start', {})
        return {'affinity_id': affinity_id, 'instance_id': self.instance_id, 'service': 'botbook'}

    @staticmethod
    def _message(raw):
        message = json.loads(raw)
        if message['kind'] == 'general':
            message.setdefault('board', 'general')
        message['sent_at'] = datetime.fromtimestamp(message['sent_at'], timezone.utc).isoformat()
        return message

    async def execute(self, request: GeneralRequest | DirectRequest | UnreadRequest):
        result = await self._run(request.affinity_id, request.action, request.model_dump())
        if request.action == 'get_unread':
            return {'affinity_id': request.affinity_id,
                    'messages': [self._message(raw) for raw in result[2:]],
                    'has_more': result[1] == '1'}
        return {'affinity_id': request.affinity_id, 'message': self._message(result[1])}

    async def close(self, affinity_id):
        await self._run(affinity_id, 'close', {})
        return {'affinity_id': affinity_id, 'closed': True}


def create_app(service: BotbookService, *, lifespan=None):
    app = FastAPI(title='LiteRegistry Botbook', lifespan=lifespan)
    app.state.botbook_service = service

    @app.get('/health')
    async def health():
        await service.redis.ping()
        return {'status': 'healthy', 'service': 'botbook', 'instance_id': service.instance_id}

    @app.post('/handshake')
    async def handshake():
        return await service.start()

    @app.post('/botbook')
    async def command(request: Command):
        return await service.execute(request)

    @app.post('/close')
    async def close(request: SessionRequest):
        return await service.close(request.affinity_id)

    return app


def build_registered_app(*, redis_url: str, registry: str | None = None,
                         head_registry: str | None = None,
                         advertise_host: str = '127.0.0.1', port: int = 8093,
                         heartbeat_interval: float = 10,
                         session_ttl_seconds: int = 1800, max_messages: int = 10000):
    if heartbeat_interval <= 0:
        raise ValueError('heartbeat_interval must be positive')
    store = get_kvstore(registry, head_registry=head_registry, raise_on_error=True)
    registration = ServerRegistry(store)
    redis = Redis.from_url(redis_url, socket_connect_timeout=5, socket_timeout=10)
    service = BotbookService(redis, instance_id=registration.server_id,
                             session_ttl_seconds=session_ttl_seconds, max_messages=max_messages)
    host = f'[{advertise_host}]' if ':' in advertise_host else advertise_host
    url = f'http://{host}'
    metadata = {'model_path': 'botbook', 'backend': 'botbook-redis',
                'instance_id': service.instance_id,
                'affinity': {'enabled': True, 'handshake_endpoint': 'handshake',
                             'command_endpoint': 'botbook', 'close_endpoint': 'close',
                             'id_field': 'affinity_id'},
                'authentication': {'type': 'none'}}

    async def heartbeat():
        while True:
            await asyncio.sleep(heartbeat_interval)
            try:
                await registration.heartbeat(url, port)
            except Exception:
                logging.exception('Botbook registry heartbeat failed')

    @asynccontextmanager
    async def lifespan(app):
        task = None
        registered = False
        try:
            await redis.ping()
            await registration.register_server(url, port, metadata)
            registered = True
            task = asyncio.create_task(heartbeat())
            yield
        finally:
            if task is not None:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            try:
                if registered:
                    await asyncio.wait_for(registration.deregister(), timeout=5)
            finally:
                try:
                    await store.close()
                finally:
                    # redis-py 4.5 exposes async close(); newer releases prefer aclose().
                    await getattr(redis, "aclose", redis.close)()

    return create_app(service, lifespan=lifespan)


def main(redis_url: str, registry: str | None = None, head_registry: str | None = None,
         host: str = '127.0.0.1', port: int = 8093, advertise_host: str | None = None,
         heartbeat_interval: float = 10, session_ttl_seconds: int = 1800,
         max_messages: int = 10000):
    """Run a Botbook replica; redis_url stores messages, registry discovers services."""
    app = build_registered_app(
        redis_url=redis_url, registry=registry, head_registry=head_registry,
        advertise_host=advertise_host or (socket.getfqdn() if host in {'0.0.0.0', '::'} else host),
        port=port, heartbeat_interval=heartbeat_interval,
        session_ttl_seconds=session_ttl_seconds, max_messages=max_messages,
    )
    uvicorn.run(app, host=host, port=port, workers=1)


if __name__ == '__main__':
    fire.Fire(main)
