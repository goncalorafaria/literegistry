# Changes

## Unreleased — Podman replay deployment

### In this PR (#16)

- Default task-session networking to `pasta` in the service, worker image, and Beaker launcher, allowing outbound DNS and package downloads. Set `--podman-session-network=none` to disable it for Beaker deployments, or `--network=none` for the service directly.
- Install `passt` and `util-linux` in the worker image. Prepare a namespace-local `/dev/net/tun`, then drop to the rootless `podman` user before starting the service. Beaker commands explicitly invoke the entrypoint so they retain this setup. The platform must allow device creation and use.
- Default Beaker workers to eight task sessions with 16 GiB per session, a two-hour affinity lifetime, and an 840-second gateway timeout. Worker resource reservations must cover the configured session limits.
- Accept Podman command deadlines up to 300 seconds.
- Add explicit client session renewal and gateway renewal handling.

Rebuild the Podman worker image and select it with `--podman-image` before deploying these entrypoint changes. Existing image references, including the legacy launcher default, are not updated by a source change. Existing workers must be replaced to acquire the new behavior.

### Related PRs

- [#11, merged](https://github.com/goncalorafaria/literegistry/pull/11): reuse healthy Redis connections without polling the head registry.
- [#14, merged](https://github.com/goncalorafaria/literegistry/pull/14): retain healthy gateway discovery results, index Redis endpoints, and incorporate obsolete head-refresh removal from #12.
- [#15, separate open PR](https://github.com/goncalorafaria/literegistry/pull/15): log the Redis operation exception before invalidating its connection.
- Streaming APIs remain in [#13](https://github.com/goncalorafaria/literegistry/pull/13).

### Replay boundaries and outstanding issues

Task-image selection, SWERL Bash/cwd/export persistence, source pytest stop policy, trace sanitization, and final-count filtering belong to the replay/judge harness; this PR does not implement those policies.

Outbound access does not freeze historical package versions. Rust dependency drift can still prevent replay builds. Original user-namespace permission failures and background-service timing or port occupancy can also differ on replay.

Observed image/layer storage errors, memory-limit losses, affinity failures, and intermittent pasta startup errors remain unresolved by this PR. Generated timestamps, addresses, keys, and archive metadata can differ even when final test counts match. Exact count agreement alone does not establish identical intermediate behavior.

No new tests were run for this update.
