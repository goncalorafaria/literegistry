# Console

Streamlit dashboard for live gateway traffic, optional registry replica counts,
and vLLM Slurm log telemetry. Launch with:

```bash
literegistry console \
  --logs /path/to/logs \
  --registry redis://login-node:6379 \
  --server-address 127.0.0.1 \
  --server-port 8765
```

By default the launcher also tries to start an **ngrok** tunnel to the Streamlit
port (skipped cleanly if `ngrok` is not on `PATH`). Disable with `--ngrok=False`.

## How it works

```text
gateway *.log  →  listener / seed parse  →  metric queue  →  Streamlit charts
slurm *.log/.err →  vLLM parser (optional panel)
literegistry summary →  registry replica table (optional poll)
```

It looks for gateway summary lines such as:

```text
Request counts (last 5.0s): Qwen/Qwen3-8B: 7, python: 8
Completion stats (last 5.0s): Qwen/Qwen3-8B: 68 reqs, avg: 14.402s, max: 60.860s
```

Initial files are tailed from the end after seeding only the latest ~50 lines
per file, so you do not replay full historical logs. Newly discovered files are
read from the start so startup lines are not missed.

## Launcher arguments (`literegistry console`)

Streamlit bind (either naming style works):

| Argument | Default | Meaning |
|----------|---------|---------|
| `server_address` / `address` | `127.0.0.1` | Streamlit bind host |
| `server_port` / `port` | `8765` | Streamlit bind port |
| `ngrok` | `True` | Open public tunnel if ngrok is installed |

App options (forwarded after `--` to `app.py`):

| Argument | Default | Meaning |
|----------|---------|---------|
| `logs` | derived | One path, or Fire list of log roots to scan |
| `logs_dir` | package `logs/gateway` | Gateway log directory (legacy single-path) |
| `registry` | cluster Redis URL | For `literegistry summary` polling |
| `slurm_logs_dir` | package `logs/slurmcompose` | Slurm / compose log dir |
| `vllm_logs_dir` | `None` | Alias override for vLLM log dir |
| `seed_recent` | `True` | Seed charts from recent tails |
| `poll_seconds` | `0.5` | Log listener poll interval |
| `window` | `"1 hour"` | Gateway chart time window label |
| `refresh` | `True` | Auto-refresh UI |
| `refresh_seconds` | `5` | UI refresh period |
| `poll_registry` | `True` | Shell out to `literegistry summary` |
| `registry_poll_seconds` | `5` | Registry poll period |
| `show_vllm` | `True` | Show vLLM telemetry panel |
| `vllm_newest_files` | `80` | Max newest vLLM log files to consider |
| `vllm_tail_lines` | `1000` | Tail depth per vLLM file |
| `vllm_window` | `"2 hours"` | vLLM chart window label |

### Examples

```bash
# Minimal local UI
literegistry console --ngrok=False --server-port 8765

# Point at your job logs + registry
literegistry console \
  --logs /mmfs1/.../logs \
  --registry redis://klone-login03.hyak.local:6379 \
  --server-address 127.0.0.1 \
  --server-port 8765

# Multiple roots
literegistry console \
  --logs='["/path/gateway","/path/slurmcompose"]' \
  --registry redis://login-node:6379
```

From a checkout without the entrypoint:

```bash
streamlit run literegistry/console/app.py \
  --server.address 127.0.0.1 --server.port 8765 -- \
  --logs /path/to/logs --registry redis://host:6379
```

Needs Python 3.9+ with Streamlit and pandas (see
`literegistry/console/requirements.txt`).

## What you see

- Request / completion rates, backlog pressure, avg & max latency
- Per-model and tool (`python`, etc.) breakdowns
- Registry replica counts (from `literegistry summary`)
- vLLM throughput, KV-cache, and request pressure from Slurm logs
- Recent parsed raw events and listener/queue status

Window labels map to seconds (`"5 min"` → 300, `"all parsed"` → no cutoff).

## Ops checklist

1. Gateway logging to a directory the console can read.
2. Same `--registry` the cluster uses, if you want replica counts.
3. For remote viewing on HPC: either SSH tunnel to `server_port`, or leave
   `ngrok=True` with a configured ngrok binary.

Next: [Gateway](gateway.md) · [docs index](README.md)
