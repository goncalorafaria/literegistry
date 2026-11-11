# Live buffered-command validation

32 rootless Podman workers on Jupiter, 2 mirrors, one Redis and gateway; 8 sessions/worker. Six waves of 128 concurrent sessions, six commands/session. Existing images loaded a frozen working-source snapshot; this was not a test of a newly baked image or of this exact PR commit. The streaming engine and underlying tool-client files were byte-identical to the PR base; the async wrapper subsequently received a local wait-limit validation fix (30,000 → 10,000 ms). Snapshot manifest and raw evidence: `/weka/gfaria/records/podman-stream32-20261111/`.

| Measurement | Result |
|---|---:|
| Sessions opened / closed | 768 / 768 |
| Planned commands | 4,608 |
| Exact-output commands passed | 4,602 |
| Command attempts with error | 1 |
| Remaining commands skipped in failed episode | 5 |
| Session error rate | 0.130% |
| Command-attempt error rate | 0.0217% (1/4,603) |
| Cleanup failures / output mismatches | 0 / 0 |
| Elapsed | 252.04 s |
| Verified commands/s, including creation and cleanup | 18.26 |
| Command latency p50 / p95 | 0.205 / 4.071 s |
| Final active / pending sessions | 0 / 0 |

The one error was HTTP 502 `affinity_upstream_disconnected` during a command's output handling. The session's start and surrounding polls succeeded, cleanup succeeded, and worker logs showed no restart or exception. Exact transport cause is unconfirmed. The gateway did not log the failed request's underlying exception. Commands were not resubmitted automatically.

Smoke checks passed: output before exit; read then wait collects only unread output; batches; nonzero exit; timeout; bounded Unicode output; merged streams; eight racing starts on one container (one accepted, seven `session_busy`); four concurrent readers without missing/duplicate output; closing while running; twelve repeated commands.

One initial test incorrectly assumed the first one-second poll must contain output. Empty polls during process startup are valid; the corrected test waits for output or completion and checks output arrives before exit. Deployment startup also required readable source permissions and retrying initial mirror discovery until core registration completed.

## Reproduce against an isolated fleet

```bash
PYTHONPATH=literegistry_tool_client/src python scripts/podman_async_live.py smoke --gateway http://HOST:PORT --output-dir /tmp/stream-smoke
PYTHONPATH=literegistry_tool_client/src python scripts/podman_async_live.py load --gateway http://HOST:PORT --output-dir /tmp/stream-load --waves 6 --sessions 128 --steps 6
```

These commands create and close their own containers. They do not clear other users' sessions. Error records and summaries are written to the supplied output directory. Same-container concurrency is explicitly rejected by the server; different containers execute concurrently.

## Focused PR validation

25 tests passed across the async client, streaming client, subprocess integration, and worker/gateway streaming suites. `git diff --check` passed.
