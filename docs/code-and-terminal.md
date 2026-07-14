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

`rg`, `grep`, `awk`, `sed`, `jq`, `xsv`, `head`, `tail`, `wc`, `cat`, `nl`

Constraints:

- Max **8** pipeline stages.
- No `;`, `&&`, `||`, redirects, subshells.
- Arguments that look like paths (`/…`, `~`, `..`) are rejected.
- Dangerous awk/sed/jq features (e.g. `system(`, sed `e`/`w`, jq `include`) rejected.

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
