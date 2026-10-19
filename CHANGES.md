# Changes

## Unreleased — Redis failure diagnostics (#15)

- Log the failing Redis operation name, elapsed operation time, exception class, and exception message before closing/invalidation.
- Keep the connection-identity guard so concurrent failures generate one invalidation warning for that connection.
- Redact the configured Redis URL in exception text and show `<no message>` for empty exceptions. Do not add keys, values, or command payloads to the warning.
- Preserve existing recovery, retry, lock, and cancellation behavior. Diagnostic timing covers the failed operation, excluding endpoint discovery and retry sleep.
- Rebuild gateway/worker images or update installed LiteRegistry code to deploy this logging change. Release 1.0.60 does not include #15.

## Podman replay deployment — merged in #16

### Included in #16

- Default task-session networking to `pasta` in the service, worker image, and Beaker launcher, allowing outbound DNS and package downloads. Set `--podman-session-network=none` to disable it for Beaker deployments, or `--network=none` for the service directly.
- Install `passt` and `util-linux` in the worker image. Prepare a namespace-local `/dev/net/tun`, then drop to the rootless `podman` user before starting the service. Beaker commands explicitly invoke the entrypoint so they retain this setup. The platform must allow device creation and use.
- Default Beaker workers to eight task sessions with 16 GiB per session, a two-hour affinity lifetime, and an 840-second gateway timeout. Worker resource reservations must cover the configured session limits.
- Accept Podman command deadlines up to 300 seconds.
- Add explicit client session renewal and gateway renewal handling.

Rebuild the Podman worker image and select it with `--podman-image` before deploying these entrypoint changes. Existing image references, including the legacy launcher default, are not updated by a source change. Existing workers must be replaced to acquire the new behavior.

### Gateway, registry, and client fixes

| Area | Change | Submission status |
| --- | --- | --- |
| Worker discovery | Index the active Redis worker roster instead of repeatedly scanning every key; bound fallback SCAN batches. | [#4, merged](https://github.com/goncalorafaria/literegistry/pull/4) |
| Session loss | Preserve typed sandbox-loss reasons, including resource-limit failures; terminate a lost client session rather than silently creating a different container. | [#5, merged](https://github.com/goncalorafaria/literegistry/pull/5) |
| Strict affinity | Probe an unavailable owner and its session with bounded liveness checks, including owners absent from the cached roster. Preserve affinity and avoid replaying an executed command on another worker. | [#6, merged](https://github.com/goncalorafaria/literegistry/pull/6) |
| Chat routing | Optional session affinity for standard chat-completions routing. | [#8, merged](https://github.com/goncalorafaria/literegistry/pull/8) |
| Redis discovery | Reuse the healthy Redis connection; consult head discovery again when it fails, rather than polling head storage during normal operations. | [#11, merged](https://github.com/goncalorafaria/literegistry/pull/11) |
| Refresh settings | Remove the obsolete head-registry refresh interval. | [#12, merged](https://github.com/goncalorafaria/literegistry/pull/12), incorporated into #14 |
| Gateway discovery | Share cached gateway selection across Podman clients; keep a healthy gateway and invalidate it on failure. Index endpoint discovery in Redis instead of full-key scans. | [#14, merged](https://github.com/goncalorafaria/literegistry/pull/14); released as LiteRegistry 1.0.59 / Podman client 0.1.5 |
| Redis diagnostics | Log the operation exception before invalidating a connection so endpoint warnings retain the underlying cause. | [#15, open](https://github.com/goncalorafaria/literegistry/pull/15) |
| Session renewal | Expose `/affinity/renew` and client renewal without executing a task command; extend deployment affinity TTL to two hours. | [#16, merged](https://github.com/goncalorafaria/literegistry/pull/16) |
| Gateway deadline | Set the Podman deployment gateway timeout to 840 seconds. | #16, merged |
| Streaming | Live Podman command output and client resume APIs. | [#13, separate open PR](https://github.com/goncalorafaria/literegistry/pull/13) |

A merged PR does not update running gateways or clients automatically. Rebuild the relevant gateway/worker images and update installed clients to deploy those revisions.

### Additional local gateway changes requiring separate submission review

The shared development checkout contains further changes beyond this PR. They are listed here to avoid treating them as already merged or covered by #16:

- New-session handshake handling recognizes worker `HTTPResponseError` responses with status 503 as well as gateway transport errors, allowing candidate selection to continue. The existing branch handles only `GatewayRequestError` in that path.
- Least-outstanding routing, shared Redis reservation accounting, routing statistics, disconnect cancellation, and routing cleanup are local changes whose PR coverage has not been confirmed.
- An opt-in filesystem-head setting bypasses `fsync` for an explicitly selected ephemeral head directory; this durability change is not included here.
- Streaming-specific gateway endpoints and response cache headers remain outside this PR.

### Replay boundaries and outstanding issues

Task-image selection, SWERL Bash/cwd/export persistence, source pytest stop policy, trace sanitization, and final-count filtering belong to the replay/judge harness; this PR does not implement those policies.

Outbound access does not freeze historical package versions. Rust dependency drift can still prevent replay builds. Original user-namespace permission failures and background-service timing or port occupancy can also differ on replay.

Observed image/layer storage errors, memory-limit losses, affinity failures, and intermittent pasta startup errors remain unresolved by this PR. Generated timestamps, addresses, keys, and archive metadata can differ even when final test counts match. Exact count agreement alone does not establish identical intermediate behavior.

No new tests were run for this update.
