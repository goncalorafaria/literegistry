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
| `head_registry` | `None` | Shared directory that publishes the current Redis endpoint; overrides `registry` |
| `host` | `0.0.0.0` | Listen address |
| `port` | `8080` | Listen port |
| `advertise_host` | node FQDN | Host used in the printed `GATEWAY_URL` |
| `instance_id` | advertised host + port | Stable identity for this gateway deployment |
| `workers` | `1` | Uvicorn workers (`>1` uses factory mode) |
| `register` | `True` | Register and heartbeat the gateway as `model_path=gateway` |
| `heartbeat_interval` | `10` | Seconds between gateway health updates |
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

## Gateway registry lifecycle

By default the gateway registers itself under `model_path="gateway"` in the
same Redis or filesystem registry that it uses for service discovery. The
record publishes its advertised URI, `/health` endpoint, configured worker
count, and capabilities. It is refreshed every `heartbeat_interval` seconds
and removed during a clean shutdown.

A multi-worker gateway publishes **one record for the entire gateway**, not one
record per Uvicorn worker. One worker owns that heartbeat through a local
leader lock; another worker takes over the same stable record within one second
if the owner exits. Use a distinct `instance_id` for each independently
deployed gateway. Registration can be disabled with `--register=False`.

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
command/close request carries that ID. Mirror soft affinity is experimental, optional, and enabled by default. Disable it to use normal mirror
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


## Strict-affinity transport failures

A pinned command is sent once. If its response is lost, the gateway does not
replay it: side effects may already have occurred. Upstream disconnects and
incomplete responses return structured HTTP 502 (`affinity_upstream_disconnected`),
timeouts return 504 (`affinity_upstream_timeout`), and connection failures return
503 (`affinity_owner_unavailable`). `execution_outcome` is `unknown` unless a
connection was never established or the request was blocked before forwarding.
An upstream application's HTTP error, including 503, is preserved without
triggering transport-failure probes.

After a transport failure, the gateway confirms registration and probes the
exact owner's `/health`. Each registry/HTTP probe has a one-second budget.
Concurrent requests for one owner share its probe, and completed results have a
two-second cooldown. During a confirmed reachability failure, later requests
return 503 with `execution_outcome=not_sent` without hitting the owner. A failed
health check alone never declares the session permanently lost. Probe state is
bounded to 256 owners per gateway process; at capacity, additional checks remain
inconclusive rather than guessing that an owner is dead.

If the owner is reachable, Podman command/close failures also trigger the
read-only `GET /sessions/{container_id}` endpoint. It reports whether the exact
container exists and is running; it does not execute a command, restart a
container, or create a replacement. Session checks share a 16-request concurrency
limit and a one-second budget including queue time. Confirmed loss returns 410
with `code=sandbox_lost` and `recoverable=false`; a stopped container has reason
`container_stopped`, an absent container has reason `unknown`, and retained
watchdog reasons are preserved. Loss of the registered owner retains the existing
`410 affinity_owner_lost` contract.

Old servers without the session endpoint, registry failures, ambiguous responses,
and probe timeouts remain inconclusive. The original transport error is returned.
Strict affinity never changes owners. These checks do not diagnose filesystem
corruption or choose a training reward/episode policy. The client exception is the
boundary for callers handling confirmed loss.
