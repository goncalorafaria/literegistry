# Code and Terminal servers

Two optional workers that register into the same registry and are reached
through the gateway:

| Service | Registry `model_path` | Gateway path |
|---------|----------------------|--------------|
| Python code executor | `python` | `POST /python` |
| Restricted terminal pipelines | `terminal` | `POST /terminal` |

---

## Code executor (`literegistry code`)

Stateless sandboxed Python in a process pool. No arbitrary `import`/`open`/
`eval`/`exec` from user code — only allowlisted tools.

### Start

```bash
literegistry code --registry redis://login-node:6379
```

### CLI arguments

| Argument | Default | Meaning |
|----------|---------|---------|
| `host` | `0.0.0.0` | Bind address |
| `port` | `1212` | Listen port |
| `registry` | Redis URL | Registration target |
| `heartbeat_interval` | `4` | Seconds between heartbeats |
| `pool_size` | CPU count | ProcessPoolExecutor workers |
| `max_queue_size` | `None` | Admission queue cap (`None` = derived) |
| `default_tools` | `json,math,time,re,pandas,…` | Tools injected into the exec namespace by default |
| `preimport_tools` | all known tool specs | Modules pre-imported at worker boot |
| `title` | `Stateless Python Code Executor` | Service title |
| `outer_timeout_grace_seconds` | `1` | Extra seconds beyond `max_runtime` before pool kill |
| `log_level` | `INFO` | Logging level |

`default_tools` / `preimport_tools` accept a CSV string or a Fire list.

### Request body (`POST /python`)

| Field | Required | Default | Meaning |
|-------|----------|---------|---------|
| `code` | yes | — | Python source to run |
| `max_runtime` | no | `5.0` | Seconds (also accepts legacy `timeout` / `code_timeout`, or `"500ms"` / `"2s"`) |
| `context_payload` | no | `None` | Exposed as `context` (often a JSON string) |
| `setup_code` | no | `None` | Run before `code` in the same namespace |
| `custom_tools` | no | `None` | Extra tool names from the server allowlist |
| `return_locals` | no | `True` | Include serializable locals in the response |

```bash
# Via gateway
curl -X POST http://localhost:8080/python \
  -H "Content-Type: application/json" \
  -d '{
    "code": "data = json.loads(context)\nprint(data[\"score\"] + 1)",
    "context_payload": "{\"score\": 41}",
    "max_runtime": 3
  }'
```

### Response fields

`stdout`, `stderr`, `success`, `locals`, `execution_time`, `final_answer`,
`retryable`.

### Safety notes

- User code cannot import or open files freely.
- Tools are server-side allowlisted (`DEFAULT_TOOL_SPECS`: numpy, pandas, sympy, …).
- Overloaded workers return HTTP 503 (gateway may retry another replica).

---

## Terminal pipeline server (`literegistry terminal`)

Not a shell. You submit **log contents** plus a **pipe of allowlisted commands**.
Stages run with stdin/stdout only — no user file paths, no `&&` / redirects.

### Start

```bash
literegistry terminal --registry redis://login-node:6379
```

### CLI arguments

| Argument | Default | Meaning |
|----------|---------|---------|
| `host` | `0.0.0.0` | Bind address |
| `port` | `1213` | Listen port |
| `registry` | Redis URL | Registration target |
| `heartbeat_interval` | `30` | Heartbeat period (seconds) |
| `max_output_bytes` | `1_048_576` | Cap on captured stdout |
| `max_response_chars` | `None` | Optional response truncation |
| `command_path` | `None` | Optional directory prepended to `PATH` for tools |

### Allowed commands

`rg`, `grep`, `awk`, `sed`, `jq`, `xsv`, `pandoc`, `sort`, `uniq`, `tr`, `cut`,
`head`, `tail`, `wc`, `cat`, `nl`, `echo`

Constraints:

- Max **8** pipeline stages.
- No `;`, `&&`, `||`, redirects, subshells.
- Arguments that look like paths (`/…`, `~`, `..`) are rejected.
- Dangerous awk/sed/jq features (e.g. `system(`, sed `e`/`w`, jq `include`) rejected.
- Pandoc accepts only stdin readers `markdown`, `commonmark`, `commonmark_x`, `gfm`, or `html` and writers `json`, `native`, or `plain`; `--sandbox` is added automatically. Filters, Lua, templates, defaults, output files, resource paths, and file operands are rejected.
- `sort` and `uniq` accept stdin-only ordering, deduplication, and counting options; file operands and sort output/temp/random-source options are rejected.
- `tr` performs stdin-only translation, deletion, and character squeezing.
- `cut` accepts byte, character, or field ranges from stdin. File operands are rejected; delimiters are limited to one byte.

Pandoc can expose document structure directly or feed its JSON AST into `jq`:

```bash
pandoc -f markdown -t native
pandoc -f markdown -t json | jq '[.blocks[] | {type: .t}]'
rg -o 'https?://[^) ]+' | sort | uniq -c
grep -o -E '20[0-9]{2}' | sort | uniq -c
cut -d, -f1,3
```

### Request body (`POST /terminal`)

| Field | Required | Default | Limits | Meaning |
|-------|----------|---------|--------|---------|
| `contents` | yes | — | ≤ 2 MiB | Input text (stdin for the pipeline) |
| `command` | yes | — | ≤ 16 KiB | Pipeline string, e.g. `rg ERROR \| head -n 5` |
| `max_runtime` | no | `5.0` | 0.01–60 s | Wall time for the whole pipeline |
| `truncation` | no | `None` | ≥ 1 | Optional char truncate of stdout |

```bash
curl -X POST http://localhost:8080/terminal \
  -H "Content-Type: application/json" \
  -d '{
    "contents": "INFO started\nERROR disk full\nERROR retrying\n",
    "command": "rg ERROR | head -n 1",
    "max_runtime": 5
  }'
```

### Response fields

`stdout`, `stderr`, `success`, `exit_code`, `execution_time`, `truncated`,
`truncated_characters`.

---

## Scaling tip

Run multiple `code` / `terminal` workers on different nodes with the same
`--registry`. The gateway’s bandit load-balances across them the same way as
model replicas (with tighter retry budgets for these endpoints).

Next: [Load balancing](load-balancing.md) · [Gateway](gateway.md)

### Podman capacity-aware handshake routing

Podman publishes `data.podman_capacity` with each registry heartbeat: active
sessions, pending creates, maximum sessions, available slots, observation time,
and validity duration. Counts are captured under the backend lock; reporting
never runs a Podman subprocess. Active counts are the backend's admission
accounting, including reservations retained until cleanup completes.

Before a new strict-affinity handshake, the gateway excludes workers whose
fresh snapshot has `active_sessions + pending_sessions >= max_sessions`, then
samples the remaining workers through the load balancer. Fallback candidates
are shuffled to avoid concentrating rejected handshakes on the first registry
entry. `event=capacity_skip` records each skipped worker.

Missing, malformed, unlimited, or expired snapshots do not exclude a worker.
Workers retain the authoritative atomic capacity check; if a worker fills after
its heartbeat, explicit 503 rejection advances to another candidate. Exhausting
cached candidates triggers one registry refresh before returning failure.
Existing sessions remain pinned to their owner regardless of reported capacity.

Both worker and gateway packages must be upgraded to enable this behavior.

### Session admission under capacity pressure

The gateway retains the original handshake for up to 300 seconds, including container creation. It tries at most eight eligible cached candidates per round, then waits with exponential jittered backoff (initially 0.25–0.5 seconds, capped at 2.5–5 seconds). Normal roster-cache expiry supplies refreshed capacity; exhausted rounds do not force extra Redis scans.

An upstream HTTP 503 containing `Podman session capacity exhausted` advances to another candidate. Rejected workers receive a five-second cooldown shared across handshakes in each gateway process. Workers become eligible again after the cooldown. Cooldowns are process-local, not distributed reservations; worker-side admission remains authoritative. The original trace is not discarded on transient candidate exhaustion.

An upstream HTTP 500/503 containing `image not known` or `layer not known` permits exactly one alternate-worker attempt. The result of that attempt is final. This policy does not retry terminal commands or creation requests with unknown outcomes. Client disconnection is checked before each admission round. A creation timeout reports `execution_outcome=unknown` and is not retried; it can require later session cleanup.

Constructor settings: `admission_timeout`, `admission_backoff_initial`, and `admission_backoff_max`. The external request timeout should exceed the admission timeout. Logs include `handshake_retry` with `reason=capacity_rejected` or `reason=image_storage`, and `admission_wait` with the round, delay, and elapsed time.
