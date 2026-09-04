# Package layout

LiteRegistry separates reusable registry primitives from runnable services and
gateway proxy routes.

```text
literegistry/
├── client.py, registry.py, kvstore.py, affinity.py   # reusable core
├── redis.py                                          # Redis store and server launcher
├── coop/                                             # cross-process coordination
│   ├── ports.py                                      # collision-safe host ports
│   ├── artifacts.py                                  # atomic warm/build materialization
│   ├── endpoints.py                                  # file/SQLite/Redis endpoint bootstrap
│   └── redis.py                                      # Redis protocol readiness probe
├── services/                                         # runnable servers/wrappers
│   ├── vllm_wrapper.py and sglang_wrapper.py
│   ├── code_server.py, terminal_server.py, search_server.py, bm25_server.py
│   ├── podman.py
│   ├── podman_server.py
│   └── docker_mirror_server.py
└── gateway/                                          # HTTP front door
    ├── __init__.py       # gateway application and shared routing
    ├── affinity.py       # handshake/execute/close proxy routes
    ├── mirror.py         # Docker Registry V2 proxy routes
    ├── basic.py          # older basic route implementation
    └── legacy.py         # pre-composable gateway
tools/                                                  # repository-only utilities
└── bandit_tuning.py                                  # offline parameter tuning
```

## Cooperation

`literegistry.coop` owns reusable coordination that is neither a backend service
nor gateway routing. `ports.py` serializes host-port selection until a child has
bound its ports; `artifacts.py` provides single-writer atomic directory builds
for warmed or extracted assets; `endpoints.py` provides TTL-backed endpoint
publication, health-aware waiting, and supervised cleanup on top of
`FileSystemKVStore`; `redis.py` provides the Redis protocol readiness probe.
Deployment packages call these modules directly and contain no compatibility
copies.

## Redis

Redis provides the registry storage and discovery substrate. Its adapter and
process launcher live directly in `literegistry/redis.py`. The public
`literegistry redis` command remains unchanged.

## Services

Everything that starts or manages a backend process belongs in `services/`.
The reusable runtime abstraction predates those wrappers and remains a core
feature in `literegistry/runtime.py`; service-specific executable-wrapper code
stays under `services/`.

The public Fire commands remain stable:

```bash
literegistry redis
literegistry vllm
literegistry code
literegistry terminal
literegistry search
literegistry bm25
literegistry podman
literegistry docker-mirror
```

For direct Python imports, use canonical service paths such as
`literegistry.services.podman_server`.

## Gateway

`literegistry.gateway` owns the Starlette application, common load-balanced
routing, retry policy, and gateway metrics. Route families are separate modules:

- `literegistry.gateway.affinity` implements stateful affinity forwarding.
- `literegistry.gateway.mirror` implements Docker Registry V2 forwarding.
- `literegistry.gateway.legacy` retains the old gateway during migration.

Start the canonical application with either `literegistry gateway` or
`python -m literegistry.gateway`.

## Repository tooling

Bandit parameter tuning is an offline development concern, so it lives at
`tools/bandit_tuning.py` and is deliberately excluded from the installed
`literegistry` runtime package. From a repository checkout, run it with
`python tools/bandit_tuning.py` or import helpers from `tools.bandit_tuning`.
