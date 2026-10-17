# Podman affinity containers

LiteRegistry includes a first-class rootless Podman server and the matching
strict-affinity routes in the LiteRegistry gateway. Datadev is not required.

```text
client → Gateway → selected Podman replica → rootless container
                 └── affinity_id → replica URI (Redis, 15-minute TTL)
```

## Start the gateway and server

```bash
literegistry gateway \
  --registry redis://registry:6379 \
  --host 0.0.0.0 \
  --port 8080 \
  --workers 8

literegistry podman \
  --registry redis://registry:6379 \
  --host 0.0.0.0 \
  --port 8091 \
  --allow-non-loopback=True \
  --advertise-host podman-node \
  --advertise-port 8091 \
  --instance-id podman-0 \
  --registry-mirror http://gateway.example:8080 \
  --image python:3.12-slim \
  --network none \
  --session-memory 4g \
  --session-pids-limit 2048 \
  --session-idle-timeout 7200 \
  --image-prune-until 24h
```

`--registry-mirror` makes Podman use the LiteRegistry gateway as its native
`docker.io` mirror. The server writes a private rootless `registries.conf` and
sets `CONTAINERS_REGISTRIES_CONF` only for its Podman subprocesses. HTTP is
marked insecure while HTTPS remains verified. Podman tries the gateway first
and retains its normal Docker Hub fallback.

The server binds to loopback by default because it has no application token.
For a managed private cluster using host networking, explicitly pass
`--host=0.0.0.0 --allow-non-loopback=True` and enforce access at the network
boundary. Run the outer service as a non-root user; containers are rootless.

The standalone LiteRegistry Podman Beaker stack can launch `docker_mirror_replicas` and one
gateway itself. It publishes the gateway's actual host and dynamically
allocated port, then supplies that URL to every Podman replica automatically.

> **Optional companions:** Applications can install the async gateway client
> with `pip install "literegistry[podman_client]"`. Operators who want the
> complete rootless Podman, Docker mirror, gateway, and Redis Beaker stack can
> use `pip install "literegistry[podman_beaker]"`. These packages are
> convenience extensions; neither is required to run the core server and HTTP
> API documented below.

## Session resource limits

`--max-sessions` places a hard cap on simultaneous containers owned by one
replica. Handshakes reserve a slot before invoking Podman, so concurrent
requests cannot race past the limit. A full replica returns HTTP 503 and the
gateway tries another registered Podman replica. The limit is off by default.

`--session-memory` (e.g. `4g`) applies `--memory` plus an equal
`--memory-swap` to every session container, so a runaway session is OOM-killed
by its own cgroup instead of exhausting the replica host and taking every
colocated session down with it. `--session-pids-limit` guards against fork
bombs the same way. Both budgets are off by default. Native enforcement
requires the host to delegate the corresponding cgroup controllers (standard
on rootful hosts; verify with `cat /sys/fs/cgroup/memory.max` inside a session
when running nested).

On nested hosts where those controllers are unavailable, a userspace watchdog
provides a fallback. It caches each container's host init PID, reads the host
`/proc` table once per sweep, and groups processes by their Linux PID-namespace
inode. It force-removes a session whose aggregate RSS or kernel-task count
exceeds its budget. Namespace membership catches `podman exec` processes even
when the outer namespace reparents them away from the container init process.
Suspected violations are confirmed with a second snapshot, and an active
runaway command can be terminated immediately without waiting for the normal
per-session command lock.

`--resource-watchdog-interval` controls this loop independently from the
slower idle/image janitor and defaults to 5 seconds. Set it to `None` to
disable userspace enforcement and rely only on native cgroups. RSS is an
approximation, not a replacement for kernel cgroup accounting.

## Command execution and deadlines

Every command runs under an in-container deadline. The server uses a
`timeout` executable found on the task image's `PATH` when available; images
without `timeout` use a Bash watchdog that kills the command's complete process
group. The server no longer assumes that `/usr/bin/timeout` exists. A timed-out
command exits `137`, is returned with `timed_out=true`, and has no surviving
child processes. The outer `podman exec` deadline remains as a five-second
failsafe. Session images must provide `/bin/bash` and `sleep`, which are also
required by the long-lived session entrypoint.

## Optional live command output

`POST /affinity/podman/start` starts a command and returns HTTP 202 immediately:

```json
{
  "service": "podman",
  "affinity_id": "<container-id>",
  "request_id": "build-1",
  "command": "make build",
  "timeout": 60,
  "merge_output": false
}
```

The response contains `command_id` and an initial `cursor` of `0`. Poll with
`GET /affinity/podman/output?service=podman&affinity_id=<container-id>&command_id=<command-id>&cursor=0&wait_ms=1000`.
Both requests go to the container's existing owner. The ordinary
`POST /affinity/podman` continues to wait for a complete result.

Each poll returns `chunks` containing `{cursor, stream, data}`, a `next_cursor`,
`running`, `done`, `has_more`, `exit_code`, `success`, `timed_out`, and `error`.
The exit code and success are null until completion. Stop when `done` is true
and `has_more` is false; check `error` as well as the exit code. An empty batch
with `running=true` means no new output arrived during the wait.
`wait_ms` ranges from 0 to 10000. `max_chunks` defaults to 64 (maximum 256).

Pass the returned `next_cursor` in the next poll. That acknowledges and frees
all earlier chunks, so consumed output does not accumulate. Retrying the same
cursor can return the same output until it is acknowledged or evicted; this
is a single-consumer flow. Send one final poll with the last `next_cursor` and
`wait_ms=0` to release the final batch. Responses disable HTTP caching.

With `merge_output=false` (the default), chunks identify stdout or stderr;
their ordering across streams reflects when the worker reads them. With
`merge_output=true`, stderr is redirected into stdout inside the container,
and all chunks use `stream="stdout"`. Output is UTF-8 text, with invalid bytes
replaced. Programs may still buffer their output; use their unbuffered mode
when needed (for example, `python -u`).

The buffer keeps at most 1 MiB or 1024 chunks per command, evicting the oldest
chunks when full. A stale cursor returns HTTP 410 with `code="output_expired"` and
`oldest_cursor`; explicitly resume there to accept the output gap. Results
expire 300 seconds after command completion, returning 410
`code="command_expired"`. Closing a session also removes its completed results.
These errors do not mean the container was lost. State is local to the worker
and does not survive a worker restart.

One worker accepts up to 64 running or retained commands. Configure these
limits with `literegistry podman --max-stream-commands=64
--stream-buffer-bytes=1048576 --stream-retention-seconds=300` or the
`PODMAN_MAX_STREAM_COMMANDS`, `PODMAN_STREAM_BUFFER_BYTES`, and
`PODMAN_STREAM_RETENTION_SECONDS` environment variables. The worker's `/health`
reports the configured limits. When full, starts return 503
`command_stream_capacity`. A start on a busy container returns 409 `session_busy`;
other sessions remain independent. Normal commands and close wait for the
running command's session lock. Existing command deadlines still apply.

Reuse the same `request_id` to retry an uncertain start: while the command is
retained, matching requests return the same handle; different inputs return
409 `request_id_conflict`. After retention expires the key can be reused and
would start a new command. No command is restarted automatically.

The Python client exposes this flow explicitly:

```python
job = await session.start_command(
    "make build", timeout=60, merge_output=True
)
async for chunk in job.stream():
    print(chunk["data"], end="", flush=True)
result = await job.wait()  # exit_code, timed_out, success, and error
```

For periodic reads use `await job.read()`; it returns stdout/stderr text and
status without IDs or cursors. `job.wait()` drains unread output and returns
cached final status without accumulating logs. The canonical
`literegistry_tool_client.PodmanExecutionClient` exposes the same command handle
through `start_command(command=...)`, with the session managed by its context
manager. For raw batch polling use `await job.poll()` on the standalone client;
for reconnecting use
`client.command_output(affinity_id, command_id, cursor=saved_cursor)`.
The worker also exposes `POST /podman/start`, `GET /podman/output`, and the
JSON-body `POST /podman/output` used by the gateway's pinned transport.

## Output truncation

A command whose stdout exceeds 1MB (stderr: 256KB) is no longer aborted.
The replica keeps the first bytes up to the limit, drains the rest, and
returns the command's real exit code with `stdout_truncated` /
`stderr_truncated` set on the response. Previously the exec was killed and the
request failed with HTTP 413, which retry layers could re-run — re-executing a
command that deterministically overflows again and serializing behind the
per-container lock.

## Janitor

`--session-idle-timeout SECONDS` starts a background janitor (sweep interval
`--janitor-interval`, default 300s) that removes session containers with no
activity for that long. This reclaims containers whose client died without
calling close and containers orphaned when a handshake response was lost in
transit — both otherwise persist until the replica restarts. Pick a timeout
comfortably above your longest legitimate gap between commands in one session.
`--image-prune-until 24h` additionally runs `podman image prune --all
--filter until=24h` each sweep; under the `vfs` storage driver every container
copies its whole image, so replicas serving many distinct images fill their
disk without pruning.

## Session lifecycle

Create a container and affinity binding:

```bash
HANDSHAKE=$(curl -fsS http://gateway:8080/affinity/handshake \
  -H 'content-type: application/json' \
  -d '{"service":"podman","image":"python:3.12-slim","client_id":"demo"}')
CONTAINER_ID=$(printf '%s' "$HANDSHAKE" | jq -r .container_id)
```

Run commands. The binding sends every command to the replica that owns the
container:

```bash
curl -fsS http://gateway:8080/affinity/podman \
  -H 'content-type: application/json' \
  -d "{\"service\":\"podman\",\"affinity_id\":\"$CONTAINER_ID\",\"command\":\"echo ai2-hello > /workspace/hello.txt\"}"

curl -fsS http://gateway:8080/affinity/podman \
  -H 'content-type: application/json' \
  -d "{\"service\":\"podman\",\"affinity_id\":\"$CONTAINER_ID\",\"command\":\"cat /workspace/hello.txt\"}"
```

Delete the container and binding:

```bash
curl -fsS http://gateway:8080/affinity/close \
  -H 'content-type: application/json' \
  -d "{\"service\":\"podman\",\"affinity_id\":\"$CONTAINER_ID\"}"
```

The affinity ID is the container ID and is a routing handle, not an
authentication secret. Successful close removes the Redis binding immediately.

## Benchmark Podman throughput

Install the standalone client extra, then run the benchmark tools from the
LiteRegistry checkout:

```bash
python -m pip install "literegistry[podman_client]"

python tools/benchmark_podman_execution_client.py  --gateway-url http://gateway:8080  --replicas 8  --concurrency 16,32,64  --total-sessions 8192  --commands-per-session 4  --output podman-throughput-r8.json
```

`--total-sessions` keeps the workload fixed across concurrency levels. Each
worker completes create, command, and confirmed-delete before taking another
trajectory. Use `--sessions-per-worker` and `--waves` instead for sequential,
phase-isolated waves.

Measure handshake-to-container-ready throughput separately:

```bash
python tools/benchmark_podman_container_creation.py  --gateway-url http://gateway:8080  --replicas 8  --concurrency 1,2,4,8,16,32,64  --output podman-creation-r8.json
```

Compare equivalent throughput results from different replica counts:

```bash
python tools/compare_podman_execution_benchmarks.py  podman-throughput-r1.json podman-throughput-r8.json
```

The comparator accepts both current `literegistry-podman-*` results and legacy
`jtc-podman-*` benchmark JSON.
