# LiteRegistry Docs

Usage-focused guides for running and operating LiteRegistry. Start with the
[main README](https://github.com/goncalorafaria/literegistry/blob/master/README.md)
for a full end-to-end workflow, then use these pages for argument details and
behavior.

| Guide | What it covers |
|-------|----------------|
| [CLI](cli.md) | Full `literegistry` subcommand reference and Fire usage |
| [Registry](registry.md) | Redis vs filesystem backends, how entries are stored, inspecting the roster |
| [Gateway](gateway.md) | OpenAI-compatible proxy, CLI args, endpoints, retries |
| [Package layout](package-layout.md) | Core modules, runnable services, gateway routes, and compatibility aliases |
| [vLLM & SGLang](vllm-sglang.md) | Launching model servers, registry registration, passthrough flags |
| [Code & Terminal](code-and-terminal.md) | Python executor and restricted log pipelines |
| [Podman](podman.md) | Rootless stateful containers, handshake affinity, commands, and cleanup |
| [Docker mirror](docker-mirror.md) | Docker Hub pull-through cache, warming, discovery, and gateway use |
| [Load balancing](load-balancing.md) | Exp3 bandit routing, failover, latency feedback |
| [Runtimes](runtimes.md) | `local` vs `apptainer`, binds, env, image pull |
| [Console](console.md) | Streamlit dashboard for gateway / vLLM / registry |

## Typical flow

```text
literegistry redis          →  registry backend
literegistry vllm / sglang  →  model workers (heartbeat into registry)
literegistry code / terminal→  stateless tool workers (optional)
literegistry bm25          →  local Lucene search workers (optional)
literegistry podman         →  stateful rootless container workers (optional)
literegistry docker-mirror  →  stateless Docker Hub pull-through cache (optional)
literegistry gateway        →  single HTTP front door + strict affinity
literegistry console        →  live ops view (optional)
```

All CLI commands go through `literegistry <subcommand>` (Fire). Registry URLs
are either `redis://host:port` or a filesystem path.

This site is built with MkDocs and published via GitHub Pages from `master`.
