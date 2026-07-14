# LiteRegistry Docs

Usage-focused guides for running and operating LiteRegistry. Start with the
[main README](../README.md) for a full end-to-end workflow, then use these pages
for argument details and behavior.

| Guide | What it covers |
|-------|----------------|
| [CLI](cli.md) | Full `literegistry` subcommand reference and Fire usage |
| [Registry](registry.md) | Redis vs filesystem backends, how entries are stored, inspecting the roster |
| [Gateway](gateway.md) | OpenAI-compatible proxy, CLI args, endpoints, retries |
| [vLLM & SGLang](vllm-sglang.md) | Launching model servers, registry registration, passthrough flags |
| [Code & Terminal](code-and-terminal.md) | Python executor and restricted log pipelines |
| [Load balancing](load-balancing.md) | Exp3 bandit routing, failover, latency feedback |
| [Runtimes](runtimes.md) | `local` vs `apptainer`, binds, env, image pull |
| [Console](console.md) | Streamlit dashboard for gateway / vLLM / registry |

## Typical flow

```text
literegistry redis          →  registry backend
literegistry vllm / sglang  →  model workers (heartbeat into registry)
literegistry code / terminal→  tool workers (optional)
literegistry gateway        →  single HTTP front door
literegistry console        →  live ops view (optional)
```

All CLI commands go through `literegistry <subcommand>` (Fire). Registry URLs
are either `redis://host:port` or a filesystem path.
