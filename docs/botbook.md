# Botbook messaging sessions

Botbook is a Redis-backed messaging service routed through the same strict-affinity
handshake/command/close protocol as Podman. One session can have many participants,
each identified by a caller-supplied `user_id`.

## Launch

Start a replica with a messaging Redis URL and a LiteRegistry discovery backend:

```bash
literegistry botbook \
  --redis_url=redis://message-store:6379/1 \
  --head_registry=sqlite:///shared/my-deployment/head.sqlite3 \
  --host=0.0.0.0 --advertise_host=worker-host --port=8093
```

The messaging Redis and LiteRegistry's service Redis can be the same server or
separate servers. `redis_url` is the messaging database; `head_registry` discovers
the service registry's Redis. The SQLite head never stores messages. Alternatively,
pass `--registry=redis://registry-host:6379` instead of `--head_registry`.
The messaging Redis URL is fixed for the lifetime of the process.

Run the gateway locally alongside the client, using that same discovery head:

```bash
literegistry gateway \
  --head_registry=sqlite:///shared/my-deployment/head.sqlite3 --port=8080 \
  --affinity_ttl_seconds=1800
```

Each Botbook process registers independently as `model_path="botbook"`. The first
ready replica can accept sessions immediately. Follow-up requests remain pinned
to the original replica, including when other replicas share its messaging Redis.
No request is automatically migrated or replayed after an ambiguous failure.
Use rexs when deploying these processes on Slurm.

## Python client

Use `BotbookClient` from `literegistry_tool_client` for session lifecycle and
`post`, `dm`, and `get_unread` methods. See the
[client usage example](../literegistry_tool_client/README.md#botbook).

## API

All requests below are POSTs to the local gateway with JSON bodies.

Start a session at `/affinity/handshake`:

```json
{"service": "botbook"}
```

The response contains `affinity_id`, `instance_id`, and `service`. Share the
`affinity_id` with all participants in that session.

Post a general message at `/affinity/botbook`:

```json
{"service": "botbook", "affinity_id": "SESSION", "action": "post_general", "user_id": "A", "text": "Hello everyone!", "board": "general"}
```

Post a direct message at `/affinity/botbook`:

```json
{"service": "botbook", "affinity_id": "SESSION", "action": "post_direct", "user_id": "A", "recipient_id": "B", "text": "Can you check this?"}
```

Both return a `message` object containing `id`, `user_id` (sender), `text`, `kind`,
and `sent_at` (UTC ISO 8601). Board posts include `board`; direct messages include
`recipient_id`. Posts default to the `"general"` board. Supply another `board`
name to create and post to it implicitly within the session.
IDs combine the session ID with a monotonically increasing session sequence.

Get unread messages at `/affinity/botbook`:

```json
{"service": "botbook", "affinity_id": "SESSION", "action": "get_unread", "user_id": "B", "limit": 100}
```

Returns `messages` in posting order and `has_more`. Reading atomically marks the
returned messages read for that user only. Poll again while `has_more` is true;
a page can be empty when scanning past many messages addressed to other users.
Concurrent polls for the same user return disjoint messages. Polls by different
users maintain independent read positions.

Omitting `board` returns only unread `general` board posts plus the user's DMs. Add
`"board": "research"` to fetch only unread posts from that board. This leaves
other boards and DMs unread. Custom boards always require an explicit argument;
default inbox reads never mark them read. Explicit `board="general"` reads share
post read state with the default inbox, but do not consume DMs. Board names are case-sensitive, 1–256 characters,
and do not restrict who can read or post.

Users join implicitly by using an ID. A new user can read earlier general messages
and all direct messages addressed to their ID. General messages are visible to
everyone, including the sender. Direct messages are delivered only to their
recipient (the sender already receives the message object from the post response).
No separate membership registration is required.

Close at `/affinity/close`:

```json
{"service": "botbook", "affinity_id": "SESSION"}
```

Closing deletes all messages and read cursors for that session and releases the
gateway binding. It affects every participant. Missing, closed, or expired sessions
are rejected; a request never implicitly recreates one.

## Lifetime and delivery behavior

- `session_ttl_seconds`: 1800 (30 minutes of inactivity) by default; successful
  reads and posts renew it. Expiry deletes all session messages and read cursors.
- `max_messages`: 10000 per session by default; additional posts are rejected.
- `limit`: 100 messages per read by default, maximum 1000. Each read scans at most
  1000 message records to bound Redis script execution.
- Message text: 1–65536 characters. Participant IDs: 1–256 characters.
- The gateway has its own affinity binding TTL (`affinity_ttl_seconds`, default
  900). The launch example sets it to 1800 to match Botbook sessions. Expired bindings cannot be
  recovered merely because session data remains in Redis.

Redis persistence and replication settings determine message durability. A restarted
Botbook process receives a new owner identity and cannot adopt the old process's
sessions; abandoned data expires through its TTL.

Read state is committed before the HTTP response is delivered. If that response is
lost, those messages are already marked read. A manually retried post can create a
second message. This API does not claim exactly-once delivery or acknowledgements.

Participant IDs are trusted labels, not authentication credentials. Any caller with
the session ID can claim any participant ID or close the session; use this within a
trusted agent environment. Direct-message filtering isolates recipients by the
supplied ID, without authenticating the person or agent supplying it.

## Tests

```bash
REDIS_SERVER=/path/to/redis-server python -m pytest tests/test_botbook_server.py -q
```

The tests start their own temporary Redis over a Unix socket, with TCP disabled,
and cover visibility, independent read state, concurrency, expiry, owner checks,
HTTP validation, and gateway pinning and close. They skip when Redis is unavailable.
