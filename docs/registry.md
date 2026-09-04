# Registry

The registry is a shared key-value store of live servers. Workers register with
metadata (usually `model_path`), send heartbeats, and the gateway discovers them
by reading that store.

## Choosing a backend

`get_kvstore(registry)` picks the backend from the string you pass:

| `registry` value | Backend | When to use |
|------------------|---------|-------------|
| `redis://host:port` | Redis | Multi-node / production; preferred |
| `sqlite:///absolute/path/registry.sqlite3` | SQLite | One durable database file; low/moderate concurrency |
| `file:///absolute/path/registry` | Filesystem | Single node or shared NFS/WeKa |

```python
from literegistry import get_kvstore, RegistryClient

store = get_kvstore("redis://login-node:6379")
# or
store = get_kvstore("sqlite:///shared/fs/registry.sqlite3")
# or
store = get_kvstore("file:///shared/fs/registry")

client = RegistryClient(store, service_type="model_path")
```

### Redis

Start Redis with the CLI (see also [Runtimes](runtimes.md)):

```bash
literegistry redis --port 6379
# prints both:
# LITEREGISTRY_HEAD_REGISTRY=file:///tmp/literegistry-redis-coordination-...
# LITEREGISTRY_REDIS_DATA_DIR=/tmp/literegistry-redis-coordination-.../redis-data
# REDIS_URL=redis://hostname:6379
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
| `advertise_host` | node FQDN | Host placed in the published Redis URL |
| `head_registry` | new temporary `file://` registry | Stable file, SQLite, or Redis endpoint registry used for Redis discovery and failover |
| `coordination_dir` | `None` | Backward-compatible name for `head_registry` |
| `coordination_ttl_seconds` | `60` | Lifetime of the healthy Redis endpoint record |
| `coordination_refresh_interval` | `30` | Redis PING and record refresh interval |
| `coordination_startup_timeout` | `600` | Maximum wait for Redis to answer PING |
| `coordination_healthcheck_timeout` | `2` | Timeout for each Redis PING |
| `persistence` | `True` | Enable Redis append-only-file persistence |
| `data_dir` | derived from file/SQLite head | Redis AOF storage directory; pass explicitly for a Redis head |
| `appendfsync` | `everysec` | AOF durability policy: `always`, `everysec`, or `no` |
| `bind` / `env` | `None` | Extra Apptainer binds / `KEY=VALUE` env |
| `apptainer_cleanenv` | `True` | Pass `--cleanenv` |
| `apptainer_executable` | `apptainer` | Binary name |
| `apptainer_extra_args` | `None` | Extra Apptainer flags |

Host Redis without Apptainer:

```bash
literegistry redis --runtime local --foreground --port 6379
```

Use an explicit head backend when other machines must discover Redis:

```bash
literegistry redis \
  --runtime=local \
  --foreground=True \
  --port=6379 \
  --head_registry=file:///shared/my-deployment/registry-bootstrap
```

SQLite and a separate Redis instance are also supported:

```bash
# One shared SQLite control-plane file
literegistry redis ... \
  --head_registry=sqlite:///shared/my-deployment/head.sqlite3

# A separate, stable Redis control plane
literegistry redis ... \
  --head_registry=redis://head-registry.example:6379 \
  --data_dir=/shared/my-deployment/redis-data
```

Redis is published there as a TTL-backed endpoint named `redis`. This is not a
normal `server_*` registration and therefore does not appear as a routable
service in `literegistry summary`. The record is refreshed only after a Redis
PING succeeds, removed on clean shutdown, and expires after a crash.

Persistence is enabled by default with Redis AOF and `appendfsync=everysec`.
With a file head, AOF defaults to `<head path>/redis-data`. With SQLite it
defaults beside the database as `<database>.redis-data`. A Redis-backed head
has no associated filesystem, so pass `data_dir` on shared storage for
cross-node resumption. This normally loses at most roughly one second of
acknowledged writes during an abrupt machine loss. Use
`--persistence=False` for an intentionally ephemeral registry.

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
- Values and TTL metadata are published with same-directory atomic replacements.
- Works well on shared filesystems (NFS/WeKa) for small clusters.
- Can bottleneck under high concurrency (many simultaneous roster scans). Prefer Redis for busy gateways.

Default path in some helpers is `/gscratch/ark/graf/registry`; always pass an
explicit `--registry` in real deployments.

```bash
# Inspect a file-backed registry
literegistry summary --registry /shared/fs/registry
literegistry detail  --registry /shared/fs/registry
```

The canonical spelling is:

```bash
literegistry summary --registry=file:///shared/fs/registry
```

Plain paths remain accepted for backward compatibility.

### SQLite store

`SQLiteKVStore` keeps all keys, values, and expiration timestamps in one
SQLite database file. It implements the same async API and TTL semantics as
the filesystem and Redis stores, using only Python's standard-library
`sqlite3` module. It is included in the core `literegistry` installation; no
SQLite extra or third-party PyPI dependency is required.

```bash
literegistry gateway \
  --registry=sqlite:///shared/my-deployment/registry.sqlite3 \
  --port=8080
```

```python
from literegistry import SQLiteKVStore, get_kvstore

store = get_kvstore("sqlite:///shared/my-deployment/registry.sqlite3")
# Equivalent:
store = SQLiteKVStore("/shared/my-deployment/registry.sqlite3")
```

Use the same absolute SQLite URI for every service and gateway. SQLite uses
file locks and the portable rollback journal rather than WAL, so a shared
filesystem must provide correct POSIX locking. It is a good middle ground for
a small deployment that wants one inspectable, persistent file. Redis remains
the recommended backend for high-concurrency clusters. SQLite can be used
directly or as the small control-plane backend in a `head+sqlite://` URI.

Affinity storage is optimized automatically; callers still use the normal
`StrictAffinityBindingStore` and `SoftAffinityBindingStore` APIs. Valid
bindings live in a dedicated `literegistry_affinity` table rather than the
generic KV table. Exact route resolution uses the table's primary key,
service/type listing and server-owner cleanup use dedicated indexes, and
server cleanup is one SQL deletion instead of one transaction per binding.
Expired rows are excluded immediately while physical TTL cleanup is
amortized, avoiding a write lock on every read. Existing affinity rows in an
older `literegistry_kv` table are migrated transactionally when the database
is opened.

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

## Endpoint bootstrap and head backends

Redis cannot advertise its own address inside Redis before clients know that
address. Every `literegistry redis` process therefore owns a small endpoint
lifecycle. The head backend may be a filesystem KV directory, a SQLite file, or
a separate Redis. With no `head_registry`, it creates a unique filesystem
registry under the system temporary directory and prints its `file://` URI.

Services can use that stable directory directly as a head registry:

```bash
literegistry gateway \
  --head_registry=file:///shared/.literegistry-coop/EXPERIMENT \
  --port=8080

# Equivalent form accepted by every existing --registry option:
literegistry terminal \
  --registry=head+file:///shared/.literegistry-coop/EXPERIMENT
```

The complete mapping is:

| Head backend passed to the Redis publisher | Registry URI passed to services |
|---|---|
| `file:///shared/head` | `head+file:///shared/head` |
| `sqlite:///shared/head.sqlite3` | `head+sqlite:///shared/head.sqlite3` |
| `redis://stable-head:6379` | `head+redis://stable-head:6379` |

The old `head:///shared/head` and raw filesystem paths remain accepted as
compatibility aliases.

The resulting `HeadRegistryKVStore` keeps normal roster and affinity data in
Redis. It checks the head record at most once every five seconds during
healthy operation. If Redis fails, the operation remains cancellable but waits
and retries once per second, rereading the head registry until the endpoint
recovers or a replacement Redis publisher appears. Heartbeats then re-register
the service in the replacement database.

```bash
GATEWAY_URL="$(python -m literegistry.coop.endpoints wait \
  --root=file:///shared/.literegistry-coop/EXPERIMENT \
  --name=gateway \
  --healthcheck=http)"
```

Resolve Redis itself from the bootstrap directory with:

```bash
REDIS_URL="$(python -m literegistry.coop.endpoints wait \
  --root=file:///shared/.literegistry-coop/EXPERIMENT \
  --name=redis \
  --healthcheck=redis)"
```

The selected head store is only the bootstrap and failover channel;
request-path registry data remains in the data-plane Redis. File and SQLite
heads must be on storage mounted by every participant with compatible
permissions. A `head+redis://` head should point to a separate stable Redis;
using the data-plane Redis as its own head cannot recover from that Redis
failing. A static `redis://` URL remains supported when failover is not needed.
