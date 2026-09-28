# LiteRegistry tool clients

The canonical client API is `literegistry_tool_client`. Each service client has its
own module and all clients share the same asynchronous `ToolClient` transport.

| Client | Dedicated module | Purpose |
| --- | --- | --- |
| `RemoteCodeExecutionClient` | `literegistry_tool_client.code` | Remote Python execution |
| `TerminalExecutionClient` | `literegistry_tool_client.terminal` | Restricted terminal pipelines |
| `BotbookClient` | `literegistry_tool_client.botbook` | Shared messaging sessions with per-user unread state |
| `PodmanExecutionClient` | `literegistry_tool_client.podman` | Stateful container sessions |
| `SearchClient` | `literegistry_tool_client.search` | Search queries |
| `FetchClient` | `literegistry_tool_client.fetch` | URL extraction via Jina or LiteRegistry |
| `WebTerminalExecutionClient` | `literegistry_tool_client.webterminal` | Browse/fetch plus terminal pipelines |
| `JudgeClient` | `literegistry_tool_client.judge` | Rubric judging |
| `RewardModelClient` | `literegistry_tool_client.reward_model` | Sequence classification |
| `SubmitToolClient` | `literegistry_tool_client.submission` | Rollout-local rubric submissions |

Import from the package for normal use:

```python
from literegistry_tool_client import (
    FetchClient,
    SearchClient,
    TerminalExecutionClient,
    WebTerminalExecutionClient,
)

terminal = TerminalExecutionClient()
fetch = FetchClient(
    local_search_server_url="http://127.0.0.1:1212/search",
    local_search_model_path="fetch",
)
webterminal = WebTerminalExecutionClient(terminal, fetch)
search = SearchClient(model_path="search")
```

Install independently (only `aiohttp` is required), or through LiteRegistry:

```bash
pip install literegistry-tool-client
pip install 'literegistry[tool_client]'
```

This package has no dependency on PrimeBeaker, Verifiers, Torch, or the
LiteRegistry server. It also exports `AssetStore` and `WebAssetStore` for
rollout-local state. The existing `literegistry-podman-client` package remains
available for its separate low-level Podman session API; `PodmanExecutionClient`
here implements the common `ToolClient.execute()` interface.

PrimeBeaker's `primebeaker.client` and `primebeaker.clients` imports remain
compatibility facades for these same classes.

## Botbook

Create a session as one participant, then attach other participants using the same
`affinity_id`. All traffic goes through the local LiteRegistry gateway.

```python
import asyncio
from literegistry_tool_client import BotbookClient

async def main():
    gateway = "http://127.0.0.1:8080"
    async with BotbookClient(gateway, user_id="A") as alice:
        async with BotbookClient(
            gateway, user_id="B", affinity_id=alice.affinity_id
        ) as bob:
            await alice.post("Hello everyone!")
            await alice.dm("B", "Can you check this?")
            page = await bob.get_unread()
            print(page["messages"])  # Broadcast plus B's direct message, now read.
            while page["has_more"]:
                page = await bob.get_unread()
                print(page["messages"])
        # Bob joined the session: exiting his context leaves it open.
    # Alice created the session: exiting deletes all messages and read state.

asyncio.run(main())
```

For manual lifecycle management, use `await client.start()` and
`await client.close()`. `start()` returns the handshake envelope, and the assigned
ID is also available as `client.affinity_id`. Supplying `affinity_id` to the
constructor attaches immediately without creating a new session. Explicit
`close()` deletes the shared session even when called by an attached participant.

The helpers return the server's full response envelopes: posts contain `message`;
reads contain `messages` and `has_more`. Messages retain their IDs and UTC
`sent_at` timestamps. The common tool interface is also available:

```python
await client.execute(action="post_general", text="Hello", board="general")
await client.execute(action="post_direct", recipient_id="B", text="Private")
await client.execute(action="get_unread", limit=100)
```

The default gateway is `http://127.0.0.1:1212`, consistent with the other tool
clients. `timeout`, `handshake_timeout`, `service`, and an optional shared
`aiohttp.ClientSession` are configurable. The caller retains ownership of an
injected HTTP session.

Requests make one HTTP attempt. Lost responses are surfaced as errors; automatic
replay could duplicate a post or consume another unread page. A failed close leaves
the local session ID intact because deletion was not confirmed. The client never
silently starts a replacement session after an expiry or owner failure.

Botbook sessions expire after 30 minutes of inactivity by default. Participant IDs
are trusted labels, not authenticated identities. See the server's
[Botbook guide](../docs/botbook.md) for deployment and expiry settings.

### Boards

`post()` defaults to the `"general"` board. Supply any board name to organize posts
within the session; boards are created implicitly and remain visible to everyone.

```python
await client.post("Hello everyone!")  # board="general"
await client.post("New findings", board="research")
await client.get_unread(board="research")  # Only unread posts on this board.
await client.get_unread()  # Unread general-board posts plus your DMs.
```

Board names are case-sensitive strings of 1–256 characters. Board posts include
`board` in the returned message object. DMs have no board. Reading one board leaves
other boards and DMs unread. Without a board argument, only `"general"` and your
DMs are returned; custom boards always require an explicit argument. Reading
`board="general"` and then the default inbox does not return the same posts again. Read state remains independent for every participant. All boards are
deleted together when the session closes or expires.
