# LiteRegistry CLI

Entry point installed by the package:

```text
literegistry = literegistry.cli:main
```

All subcommands are [Python Fire](https://github.com/google/python-fire) functions:

```bash
literegistry <subcommand> [--arg=value ...]
```

`--help` works on the root and on each subcommand (`literegistry gateway --help`).

## Subcommands at a glance

| Subcommand | Role | Deeper docs |
|------------|------|-------------|
| `redis` | Start Redis (registry backend) | [Registry](registry.md), [Runtimes](runtimes.md) |
| `vllm` | Launch vLLM + register | [vLLM & SGLang](vllm-sglang.md) |
| `sglang` | Launch SGLang + register | [vLLM & SGLang](vllm-sglang.md) |
| `gateway` | HTTP front door / load balancer | [Gateway](gateway.md) |
| `code` | Stateless Python workers | [Code & Terminal](code-and-terminal.md) |
| `terminal` | Restricted log pipelines | [Code & Terminal](code-and-terminal.md) |
| `console` | Streamlit ops dashboard | [Console](console.md) |
| `summary` | Print replica counts per model | [Registry](registry.md) |
| `detail` | Print URI + metadata per replica | [Registry](registry.md) |

## Registry URL convention

Almost every command takes `--registry`:

| Value | Backend |
|-------|---------|
| `redis://host:port` | Redis |
| `/path/to/dir` (anything without `redis://`) | Filesystem KV store |

Pass the **same** `--registry` to workers, gateway, and inspect commands.

---

## `literegistry redis`

Start a Redis server and print `REDIS_URL=redis://hostname:PORT`.

```bash
literegistry redis --port 6379
literegistry redis --runtime local --foreground --port 6379
```

| Argument | Default | Meaning |
|----------|---------|---------|
| `port` | `6379` | Listen port |
| `runtime` | `apptainer` | `apptainer` or `local` |
| `foreground` | `False` | Stay attached to this process |
| `log` | `None` | Optional log file |
| `image` | `redis_7-alpine.sif` | Apptainer image |
| `image_source` | `docker://redis:7-alpine` | Pull source |
| `pull_image` | `True` | Pull if SIF missing |
| `redis_server_path` | `None` | Host binary when `runtime=local` |
| `workdir` / `bind` / `env` | `None` | Apptainer workdir, binds, env |
| `apptainer_cleanenv` | `True` | `--cleanenv` |
| `apptainer_executable` | `apptainer` | Binary name |
| `apptainer_extra_args` | `None` | Extra Apptainer flags |

---

## `literegistry vllm` / `literegistry sglang`

Start a model server, register under `model_path=<model>`, heartbeat while healthy.
Extra flags are forwarded to the backend (`--tensor-parallel-size`, `--tp-size`, …).

```bash
literegistry vllm \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --registry redis://login-node:6379 \
  --tensor-parallel-size 4

literegistry sglang \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --registry redis://login-node:6379 \
  --tp-size 1
```

| Argument | Default | Meaning |
|----------|---------|---------|
| `model` | Llama 3.1 8B Instruct | Model id/path → registry `model_path` |
| `host` | `0.0.0.0` | Backend bind host |
| `registry` | path or Redis (wrapper default) | Registration target |
| `port` | random `8000–12000` | Backend port |
| `runtime` | `apptainer` | `local` or `apptainer` |
| `image` / `image_source` | backend-specific SIF / Docker URI | Container image |
| `pull_image` | `True` | Pull if missing |
| `bind` / `env` / `workdir` | `None` (+ auto HF binds/env) | Container mounts & env |
| `apptainer_nv` | `True` | GPU (`--nv`) |
| `apptainer_cleanenv` | `True` | `--cleanenv` |
| `apptainer_executable` | `apptainer` | Binary |
| `apptainer_extra_args` | `None` | Extra flags |
| `**kwargs` | — | Passed through as `--kebab-case` flags |

Full flag tables and image names: [vLLM & SGLang](vllm-sglang.md), [Runtimes](runtimes.md).

---

## `literegistry gateway`

OpenAI-compatible proxy with bandit load balancing.

```bash
literegistry gateway --registry redis://login-node:6379 --port 8080
literegistry gateway --registry redis://login-node:6379 --port 8080 --workers 4
```

| Argument | Default | Meaning |
|----------|---------|---------|
| `registry` | cluster Redis URL | Discovery backend |
| `port` | `8080` | Listen port |
| `workers` | `1` | Uvicorn workers (`>1` = multi-process) |
| `timeout` | `61` | Model request timeout (s) |
| `python_timeout` | `20` | `/python` timeout |
| `python_max_retries` | `3` | `/python` attempts |
| `python_retry_budget_seconds` | `20` | `/python` retry wall budget |
| `terminal_timeout` | `20` | `/terminal` timeout |
| `terminal_max_retries` | `2` | `/terminal` attempts |
| `terminal_retry_budget_seconds` | `20` | `/terminal` retry wall budget |

Endpoints: `/health`, `/session-stats`, `/v1/models`, `/v1/completions`,
`/classify`, `/python`, `/terminal` — see [Gateway](gateway.md).

---

## `literegistry code`

Stateless Python executor; registers as `model_path=python`.

```bash
literegistry code --registry redis://login-node:6379 --port 1212
```

| Argument | Default | Meaning |
|----------|---------|---------|
| `host` | `0.0.0.0` | Bind host |
| `port` | `1212` | Listen port |
| `registry` | Redis URL | Registration target |
| `heartbeat_interval` | `4` | Heartbeat period (s) |
| `pool_size` | CPU count | Process pool size |
| `max_queue_size` | `None` | Admission queue cap |
| `default_tools` | csv of common libs | Tools in default namespace |
| `preimport_tools` | all known specs | Pre-imported at boot |
| `title` | `Stateless Python Code Executor` | Service title |
| `outer_timeout_grace_seconds` | `1` | Extra seconds past `max_runtime` |
| `log_level` | `INFO` | Log level |

Request shape via gateway: [Code & Terminal](code-and-terminal.md).

---

## `literegistry terminal`

Restricted stdin pipelines; registers as `model_path=terminal`.

```bash
literegistry terminal --registry redis://login-node:6379 --port 1213
```

| Argument | Default | Meaning |
|----------|---------|---------|
| `host` | `0.0.0.0` | Bind host |
| `port` | `1213` | Listen port |
| `registry` | Redis URL | Registration target |
| `heartbeat_interval` | `30` | Heartbeat period (s) |
| `max_output_bytes` | `1048576` | Stdout capture cap |
| `max_response_chars` | `None` | Optional response truncate |
| `command_path` | `None` | Extra `PATH` dir for tools |

Allowed commands: `rg`, `grep`, `awk`, `sed`, `jq`, `xsv`, `head`, `tail`,
`wc`, `cat`, `nl`, `echo`. Details: [Code & Terminal](code-and-terminal.md).

---

## `literegistry console`

Streamlit dashboard for gateway / vLLM / registry views.

```bash
literegistry console \
  --logs /path/to/logs \
  --registry redis://login-node:6379 \
  --server-address 127.0.0.1 \
  --server-port 8765
```

| Argument | Default | Meaning |
|----------|---------|---------|
| `server_address` / `address` | `127.0.0.1` | Streamlit bind host |
| `server_port` / `port` | `8765` | Streamlit bind port |
| `ngrok` | `True` | Try ngrok tunnel if installed |
| `logs` | derived | Log root(s) to scan |
| `logs_dir` | package gateway logs | Legacy single gateway path |
| `registry` | cluster Redis URL | For summary polling |
| `slurm_logs_dir` / `vllm_logs_dir` | package defaults | vLLM/Slurm logs |
| `seed_recent` | `True` | Seed from recent tails |
| `poll_seconds` | `0.5` | Log poll interval |
| `window` | `"1 hour"` | Gateway chart window |
| `refresh` / `refresh_seconds` | `True` / `5` | UI auto-refresh |
| `poll_registry` / `registry_poll_seconds` | `True` / `5` | Run `summary` periodically |
| `show_vllm` | `True` | vLLM panel |
| `vllm_newest_files` | `80` | Max newest log files |
| `vllm_tail_lines` | `1000` | Tail depth |
| `vllm_window` | `"2 hours"` | vLLM chart window |

Full guide: [Console](console.md).

---

## `literegistry summary`

Print `model_path : replica_count` for active servers.

```bash
literegistry summary --registry redis://login-node:6379
literegistry summary --registry /shared/fs/registry
```

| Argument | Default | Meaning |
|----------|---------|---------|
| `registry` | cluster Redis URL | Backend to inspect |

---

## `literegistry detail`

Same grouping as `summary`, plus each replica’s `uri` and `metadata`.

```bash
literegistry detail --registry redis://login-node:6379
```

| Argument | Default | Meaning |
|----------|---------|---------|
| `registry` | cluster Redis URL | Backend to inspect |

---

## Fire usage tips

```bash
# Flags: both styles usually work
literegistry gateway --port 8080
literegistry gateway --port=8080

# Booleans
literegistry redis --foreground
literegistry console --ngrok=False

# Lists / nested values
literegistry console --logs='["/a/logs","/b/logs"]'
literegistry vllm --bind='["/data:/data","/models:/models"]'

# Underscores become kebab-case for backend passthrough
literegistry vllm --tensor_parallel_size=4   # → --tensor-parallel-size 4
```

Equivalent module forms (same Fire mains):

```bash
python -m literegistry.cli gateway --registry redis://… --port 8080
python -m literegistry.gateway --registry redis://… --port 8080
python -m literegistry.vllm_wrapper --model … --registry redis://…
```

## Minimal cluster recipe

```bash
# 1) Registry
literegistry redis --port 6379
# note REDIS_URL=...

# 2) Workers (on GPU nodes)
literegistry vllm --model meta-llama/Llama-3.1-8B-Instruct \
  --registry redis://LOGIN:6379 --tensor-parallel-size 1
literegistry code --registry redis://LOGIN:6379
literegistry terminal --registry redis://LOGIN:6379

# 3) Front door
literegistry gateway --registry redis://LOGIN:6379 --port 8080 --workers 4

# 4) Inspect / watch
literegistry summary --registry redis://LOGIN:6379
literegistry console --logs /path/to/logs --registry redis://LOGIN:6379
```

Next: [docs index](README.md) · [Registry](registry.md) · [Gateway](gateway.md)
