# Client gateway recovery test

Patched client: `0.1.6+recovery1`. Same private fleet, 31 established sessions (previous test: 30). Gateway moved to 28941, then restored to 28940. Redis was not restarted in this repeat.

| Scenario | Previous failures | New failures / requests | New failure rate | All sessions recovered |
|---|---:|---:|---:|---:|
| Gateway relocation | 299 | 14 / 755 | 1.85% | 18.0s (31/31) |
| Gateway restoration | 88 | 11 / 1032 | 1.07% | 16.2s (31/31) |

## Changes

- Bounded 120-second recovery after discovery or connection-establishment failures; jittered backoff up to 5 seconds. Zero disables recovery.
- Reuses existing generation-aware shared discovery and pool retirement. Also catches discovery errors from `open()`. Session affinity IDs are unchanged.
- Unsent operations retry. Mutating operations with unknown outcomes do not retry; GET transport failures can retry.
- Six targeted checks passed, including a real HTTP response disconnect proving the command is submitted once.

## Remaining gaps

- Remaining caller-visible errors are connection resets/disconnects during in-flight operations. Durable request status/result tracking and create idempotency are **not implemented**; these need a server protocol change.
- One initial creation timed out and left one orphan. This is separate from the outage request counts. See orphan cleanup evidence.
- New replacements started faster: original full recovery was 74.8s/35.2s. These are single uncontrolled trials with different phase durations; counts and rates do not isolate the client effect.
- The test keeps sessions alive after individual errors and checks sentinel contents. It does not prove an RL episode resumes after an unknown-outcome error.
- The preliminary attempt exposed an uncaught discovery path; its failed results remain in the parent directory, excluded from the corrected `repeat/` comparison.

## Evidence recorded in the private test workspace

- `comparison.json`: metrics and exact error breakdown.
- `repeat/events.jsonl`, `repeat/faults.jsonl`, `repeat/probe.log`: raw evidence and retry logs.
- `wheels/literegistry_podman_client-0.1.6+recovery1-py3-none-any.whl`: built client.
- `check-recovery.py`: targeted checks; also retained in the package tests.

Final cleanup: 31 known sessions closed normally; the one log-confirmed orphan was closed directly. Final snapshot: **128/128 healthy, zero active sessions, zero pending**. The patched wheel is installed in the private replay control environment; no RL job was launched.

Full evidence directory: `/weka/gfaria/records/private-fleet-outage-recovery-20261020`. These workspace artifacts are not included in the public repository.
