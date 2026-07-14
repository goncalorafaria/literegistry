# Registry

The registry is a shared key-value store of live servers. Workers register with
metadata (usually `model_path`), send heartbeats, and the gateway discovers them
by reading that store.

## Choosing a backend

`get_kvstore(registry)` picks the backend from the string you pass:

| `registry` value | Backend | When to use |
|------------------|---------|-------------|
| `redis://host:port` | Redis | Multi-node / production; preferred |
| Any other path string | Filesystem | Single node or shared NFS |

```python
from literegistry import get_kvstore, RegistryClient

store = get_kvstore("redis://login-node:6379")
# or
store = get_kvstore("/shared/fs/registry")

client = RegistryClient(store, service_type="model_path")
```

### Redis

Start Redis with the CLI (see also [Runtimes](runtimes.md)):

```bash
literegistry redis --port 6379
# prints: REDIS_URL=redis://hostname:6379
```

| Argument | Default | Meaning |
|----------|---------|---------|
| `port` | `6379` | Redis listen port |
| `runtime` | `apptainer` | `apptainer` or `local` |
| `foreground` | `False` | Keep Redis attached to this process |
| `log` | `None` | Optional log file path |
| `image` | `redis_7-alpine.sif` | Apptainer SIF name/path |
| `image_source` | `docker://redis:7-alpine` | Pull source if image missing |
| `pull_image` | `True` | Pull when SIF is absent |
| `redis_server_path` | `None` | Host `redis-server` binary when `runtime=local` |
| `bind` / `env` | `None` | Extra Apptainer binds / `KEY=VALUE` env |
| `apptainer_cleanenv` | `True` | Pass `--cleanenv` |
| `apptainer_executable` | `apptainer` | Binary name |
| `apptainer_extra_args` | `None` | Extra Apptainer flags |

Host Redis without Apptainer:

```bash
literegistry redis --runtime local --foreground --port 6379
```

### Filesystem store (the “file thing”)

`FileSystemKVStore` uses a directory as the database:

- **Root directory** = store (`mkdir`’d if missing)
- **Each key** = one file under that directory
- **Each value** = file contents (UTF-8 JSON for registry entries)

```text
/path/to/registry/
  server_hostname-1718….json-ish-key   # one file per registered server
```

Implementation details that matter in practice:

- Keys are filenames from `root.glob("*")` (files only, not subdirs).
- Reads/writes go through a thread executor so they do not block the event loop.
- Works well on shared filesystems (NFS/WeKa) for small clusters.
- Can bottleneck under high concurrency (many simultaneous roster scans). Prefer Redis for busy gateways.

Default path in some helpers is `/gscratch/ark/graf/registry`; always pass an
explicit `--registry` in real deployments.

```bash
# Inspect a file-backed registry
literegistry summary --registry /shared/fs/registry
literegistry detail  --registry /shared/fs/registry
```

## What a registration looks like

On register, LiteRegistry writes roughly:

```json
{
  "server_id": "hostname-1718...",
  "host": "node01",
  "port": 8123,
  "last_heartbeat": 1710000000.0,
  "status": "active",
  "uri": "http://node01.fqdn:8123",
  "metadata": {
    "model_path": "meta-llama/Llama-3.1-8B-Instruct",
    "backend": "vllm",
    "runtime": "apptainer",
    "...": "..."
  }
}
```

Key naming: `server_{server_id}`.

`service_type` on `RegistryClient` (default `"model_path"`) is the metadata field
used to group servers into “models”. Code workers use `model_path="python"`;
terminal workers use `model_path="terminal"`.

## Heartbeats and liveness

| Concept | Default (client) | Meaning |
|---------|------------------|---------|
| Heartbeat interval (workers) | ~10s (vLLM/SGLang), 4s (code), 30s (terminal) | How often the worker refreshes `last_heartbeat` |
| `max_heartbeat_interval` | `240` s on `RegistryClient` | Servers older than this are treated as inactive and dropped from the roster |

If a key disappears, the next heartbeat re-registers with the same metadata.

## Inspecting the registry

```bash
# Counts per model_path
literegistry summary --registry redis://login-node:6379

# URI + metadata per replica
literegistry detail --registry redis://login-node:6379
```

| Command | Args | Output |
|---------|------|--------|
| `summary` | `registry` | `model_path : replica_count` |
| `detail` | `registry` | Per-server `uri` and `metadata` |

## Python API sketch

```python
from literegistry import RegistryClient, get_kvstore
import asyncio

async def main():
    client = RegistryClient(get_kvstore("redis://localhost:6379"))

    await client.register_server(
        url="http://myhost.fqdn",
        port=8000,
        metadata={"model_path": "meta-llama/Llama-3.1-8B-Instruct"},
    )

    models = await client.models()          # {model_path: [server_info, ...]}
    uris = await client.get_all("meta-llama/Llama-3.1-8B-Instruct")
    best = await client.get("meta-llama/Llama-3.1-8B-Instruct")

asyncio.run(main())
```

### Useful `RegistryClient` constructor args

| Arg | Default | Meaning |
|-----|---------|---------|
| `store` | required | KV backend |
| `service_type` | `"model_path"` | Metadata key used for grouping |
| `cache_ttl` | half of heartbeat interval | How long roster/model lists are cached |
| `max_heartbeat_interval` | `240` | Inactive cutoff (seconds) |
| `penalty_latency` | `60.0` | Latency reported on failures (feeds the bandit) |
| `bandit_gamma` | `0.2` | Exp3 exploration (see [Load balancing](load-balancing.md)) |
| `bandit_l_max` | same as `penalty_latency` | Latency normalization for Exp3 |

Next: [Gateway](gateway.md) · [Load balancing](load-balancing.md)
