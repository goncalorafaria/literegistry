# LiteRegistry tool clients

The canonical client API is `literegistry_tool_client`. Each service client has its
own module and all clients share the same asynchronous `ToolClient` transport.

| Client | Dedicated module | Purpose |
| --- | --- | --- |
| `RemoteCodeExecutionClient` | `literegistry_tool_client.code` | Remote Python execution |
| `TerminalExecutionClient` | `literegistry_tool_client.terminal` | Restricted terminal pipelines |
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
