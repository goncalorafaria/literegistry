# literegistry console

Streamlit dashboard for `literegistry.gateway` and vLLM Slurm logs. The app starts
a background listener that tails new gateway lines, parses gateway summary lines
into metric events, puts those events on a queue, and lets the Streamlit render
loop drain the queue for processing and display. It also reads bounded recent tails
from `logs/slurmcompose` for vLLM backend telemetry.

Run after installing `literegistry`:

```bash
literegistry console --server-address 127.0.0.1 --server-port 8765
```

Pass log locations directly to the subcommand. The console scans those locations
for `.log` gateway summaries and `.log`/`.err` vLLM telemetry, then classifies
matching lines by content:

```bash
literegistry console --server-address 127.0.0.1 --server-port 8765 \
  --logs /mmfs1/gscratch/ark/graf/judges-that-code-pro/logs \
  --registry redis://klone-login03.hyak.local:6379
```

You can also pass a Fire list:

```bash
literegistry console --logs '["/path/to/gateway", "/path/to/slurmcompose"]'
```

From a source checkout, the direct Streamlit form still works:

```bash
streamlit run literegistry/console/app.py --server.address 127.0.0.1 --server.port 8765
```

Use a Python environment with Streamlit and pandas installed. The system `python3`
on some login nodes may be too old; a Python 3.9+ conda/venv environment is the
intended target.

Point the sidebar at log locations such as:

```text
/mmfs1/gscratch/ark/graf/judges-that-code-pro/logs/gateway
/mmfs1/gscratch/ark/graf/judges-that-code-pro/logs/slurmcompose
```

The app parses lines like:

```text
Request counts (last 5.0s): Qwen/Qwen3-8B: 7, python: 8
Completion stats (last 5.0s): Qwen/Qwen3-8B: 68 reqs, avg: 14.402s, max: 60.860s
```

It seeds from only the latest 50 log lines per file by default, then prioritizes new
lines from the live listeners. It shows request rate, completion rate, backlog pressure,
average latency, max latency, model/tool breakdowns, workflow source summaries,
queue/listener status, registry replicas, vLLM throughput/KV-cache/request pressure,
and recent raw parsed events. Registry summary polling is optional and shells out to:

```bash
literegistry summary --registry redis://host:6379
```

Runtime shape:

```text
*.log listener -> Queue[metric event] -> Streamlit drain/process -> charts/tables
```

The initial files are tailed from the end after optional seed parsing, so the app does
not replay the same existing log lines twice or scan historical gateway logs. Newly
discovered log files are read from the beginning so startup lines are not missed.
