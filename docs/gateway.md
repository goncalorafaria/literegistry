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
  uvicorn literegistry.gateway:app --host 0.0.0.0 --port 8080 --workers 4
```

## CLI arguments

| Argument | Default | Meaning |
|----------|---------|---------|
| `registry` | cluster Redis URL | `redis://…` or filesystem path |
| `port` | `8080` | Listen port |
| `workers` | `1` | Uvicorn workers (`>1` switches to multi-process mode) |
| `timeout` | `61` | Seconds for model completion / classify requests |
| `python_timeout` | `20` | Timeout for `/python` proxied calls |
| `python_max_retries` | `3` | Max replica attempts for `/python` |
| `python_retry_budget_seconds` | `20` | Wall-clock budget for `/python` retries |
| `terminal_timeout` | `20` | Timeout for `/terminal` |
| `terminal_max_retries` | `2` | Max replica attempts for `/terminal` |
| `terminal_retry_budget_seconds` | `20` | Wall-clock budget for `/terminal` retries |

When `workers > 1`, the same values are exported as env vars for child
processes: `REGISTRY_PATH`, `TIMEOUT`, `PYTHON_TIMEOUT`, etc.

## Endpoints

| Method | Path | Routes to |
|--------|------|-----------|
| `GET` | `/health` | Registry force-refresh; returns model count |
| `GET` | `/session-stats` | Shared aiohttp session / connector stats |
| `GET` | `/v1/models` | Distinct `model_path` values (+ metadata) |
| `POST` | `/v1/completions` | Replica with matching `model` |
| `POST` | `/classify` | Replica with matching `model` |
| `POST` | `/python` | Workers registered as `model_path=python` |
| `POST` | `/terminal` | Workers registered as `model_path=terminal` |

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
