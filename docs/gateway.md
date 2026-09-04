# Gateway

The gateway is the single HTTP front door. Clients talk OpenAI-style (and a few
extra endpoints); the gateway looks up replicas in the registry and forwards
with load balancing + retries.

```text
Client → Gateway → RegistryHTTPClient → sampled replica
```

## Start it

```bash
literegistry gateway \
  --registry redis://login-node:6379 \
  --port 8080
```

Multi-worker (production):

```bash
ulimit -n 65536
literegistry gateway --registry redis://login-node:6379 --port 8080 --workers 4
```

Or via uvicorn directly:

```bash
REGISTRY_PATH=redis://login-node:6379 \
  uvicorn literegistry.gateway:create_app --factory --host 0.0.0.0 --port 8080 --workers 4
```

## CLI arguments

| Argument | Default | Meaning |
|----------|---------|---------|
| `registry` | cluster Redis URL | `redis://…` or filesystem path |
| `host` | `0.0.0.0` | Listen address |
| `port` | `8080` | Listen port |
| `advertise_host` | node FQDN | Host used in the printed `GATEWAY_URL` |
| `workers` | `1` | Uvicorn workers (`>1` uses factory mode) |
| `registry_cache_ttl_seconds` | `5` | Registry roster cache lifetime |
| `timeout` | `300` | Default and Podman affinity request timeout |
| `docker_mirror_service` | `docker-mirror` | Registry service used for mirror discovery |
| `docker_mirror_connect_timeout` | `3` | Mirror connection timeout |
| `docker_mirror_read_timeout` | `300` | Mirror streaming read timeout |
| `docker_mirror_max_retries` | `3` | Replicas tried before streaming starts |
| `docker_mirror_soft_affinity` | `True` | Enable experimental repository-derived soft affinity |
| `affinity_ttl_seconds` | `900` | Sliding lifetime of strict and mirror soft-affinity bindings |
| `log_level` | `info` | Uvicorn log level |
| `access_log` | `False` | Enable Uvicorn access logs |
| `reload` | `False` | Development reload mode; requires one worker |

When `workers > 1`, registry and affinity settings are exported for each
factory-created worker process.

## Endpoints

| Method | Path | Routes to |
|--------|------|-----------|
| `GET` | `/health` | Registry force-refresh; returns model count |
| `GET` | `/session-stats` | Shared aiohttp session / connector stats |
| `GET` | `/v1/models` | Distinct `model_path` values (+ metadata) |
| `POST` | `/v1/completions` | Replica with matching `model` |
| `POST` | `/v1/chat/completions` | Replica with matching `model` |
| `POST` | `/classify` | Replica with matching `model` |
| `POST` | `/python` | Workers registered as `model_path=python` |
| `POST` | `/terminal` | Workers registered as `model_path=terminal` |
| `POST` | `/affinity/handshake` | Select a stateful replica and create a binding |
| `POST` | `/affinity/podman` | Run a command on the bound Podman container |
| `POST` | `/affinity/close` | Delete the container and release its binding |
| `GET/HEAD` | `/v2` | Docker Registry V2 mirror root |
| `GET/HEAD` | `/v2/` | Docker Registry V2 mirror root |
| `GET/HEAD` | `/v2/{path}` | A discovered `model_path=docker-mirror` replica |

Podman routes are stateful: handshake creates an affinity binding and every
command/close request carries that ID. Before forwarding, the gateway checks
that the bound replica is still in the registry roster (cached, then refreshed
from Redis). Because the roster lags reality — a replica whose heartbeats were
delayed, or a registry hiccup, drops it for a while although it still holds
the session containers — an owner missing from the refreshed roster is then
probed directly (`GET /health` on its exact URI, bounded by
`GATEWAY_AFFINITY_OWNER_PROBE_TIMEOUT_SECONDS`, default `3`). A reachable owner
keeps receiving its pinned requests (logged as
`event=owner_off_roster_alive`); an unreachable one fails the request with
HTTP 503 `strict affinity server is no longer registered`. A replica that was
*restarted* at the same address answers the forwarded request with its own
HTTP 404 (the container is gone), which clients should treat as session loss
and handshake again. Strict affinity never substitutes another replica.
Mirror soft affinity is experimental, optional, and enabled by default. Disable it to use normal mirror
load balancing. When enabled, affinity is inferred from the repository path
without a handshake.

### Completions

```bash
curl -X POST http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "prompt": "Hello",
    "max_tokens": 64
  }'
```

- **Required body field:** `model` — must match a registered `model_path`.
- All other fields are forwarded to the backend as-is.

### Chat completions

~~~bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 64
  }'
~~~

The gateway requires model, forwards messages and all other fields unchanged,
and sends the request to POST /v1/chat/completions on the selected replica.


### Classify

Same routing: body must include `model`. Forwarded to `POST /classify` on the
chosen replica.

### Python

```bash
curl -X POST http://localhost:8080/python \
  -H "Content-Type: application/json" \
  -d '{"code": "print(2 + 2)", "max_runtime": 1.0}'
```

- **Required:** `code`
- Gateway always looks up `model_path="python"` (no `model` field needed).
- Uses the shorter python retry/timeout knobs above.

### Terminal

```bash
curl -X POST http://localhost:8080/terminal \
  -H "Content-Type: application/json" \
  -d '{
    "contents": "INFO ok\nERROR disk full\n",
    "command": "rg ERROR | head -n 1",
    "max_runtime": 5
  }'
```

Routes to `model_path="terminal"`. See [Code & Terminal](code-and-terminal.md).

### Health / session stats

```bash
curl http://localhost:8080/health
curl http://localhost:8080/session-stats
```

Healthy response includes `models_count`. Session stats should show
`shared_session_initialized: true` (LiteLLM-style single shared aiohttp session).

## How routing works (short)

1. Parse JSON body; read `model` (or hardcode `python` / `terminal`).
2. Build `RegistryHTTPClient(registry, model, …)`.
3. Call `request_with_rotation(endpoint, payload)`.
4. Client samples replicas via the Exp3 bandit, tries until success / retries /
   budget exhausted, and reports latency back for the next request.

Details: [Load balancing](load-balancing.md).

## Ops tips

- Raise `ulimit -n` (e.g. `65536`) before busy gateways.
- Prefer one shared gateway process family with `--workers` rather than many
  independent gateways fighting for the same FDs.
- Watch logs for `Request counts (last 5.0s): …` and `Probs: …` — those are the
  console’s main signal sources.
- Failures return HTTP 500 with `{"error": "...", "status": "failed"}`; missing
  `model` / `code` returns 400.

Next: [vLLM & SGLang](vllm-sglang.md) · [Load balancing](load-balancing.md)
