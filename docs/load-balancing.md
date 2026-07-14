# Load balancing

LiteRegistry does not use round-robin alone. The gateway picks replicas with an
**Exp3 bandit** that learns from observed latency and failures, then retries
other replicas on error.

```text
request
  → sample_servers(model)          # Exp3 over live URIs
  → POST to chosen replica
  → on success: report_latency(uri, t, success=True)
  → on failure: report_latency(uri, penalty, success=False), try next
```

## Where it lives

| Layer | Role |
|-------|------|
| `RegistryClient.bandit` | `Exp3Dynamic` instance; holds weights |
| `sample_servers(value, n)` | Syncs active arms, returns `n` (uri, prob) samples |
| `RegistryHTTPClient.request_with_rotation` | Tries sampled servers until success / budget |
| Gateway | Creates the HTTP client per request with timeout / retry knobs |

Default bandit on `RegistryClient`:

```python
Exp3Dynamic(gamma=bandit_gamma, L_max=bandit_l_max or penalty_latency)
# defaults: gamma=0.2, penalty_latency=60.0
```

## Exp3Dynamic in practice

Each active replica URI is an “arm”.

1. **Sync arms** — new URIs get the average existing weight; dead URIs are removed.
2. **Probabilities** — mixture of weight-proportional mass and a uniform floor
   `gamma / K` so every replica keeps some traffic.
3. **Sample** — `random.choices` with those probabilities (gateway asks for
   `k=max_retries` candidates up front).
4. **Update** after each attempt:
   - success → reward from normalized latency: faster ⇒ better
   - failure → worst reward (`-1`), and telemetry records `penalty_latency`

### Parameters that matter

| Parameter | Default | Effect |
|-----------|---------|--------|
| `gamma` (`bandit_gamma`) | `0.2` | Exploration. Higher ⇒ more even traffic; lower ⇒ stick to winners |
| `L_max` (`bandit_l_max`) | `penalty_latency` (60s) | Latency scale for reward. Set near ~P99 of real latencies for sharp learning |
| `penalty_latency` | `60.0` | What failures “cost” in telemetry / learning |
| `init_weight` | `1.0` | New arms start equal (usually leave alone) |

Rough gamma guidance:

| Situation | Suggested `gamma` |
|-----------|-------------------|
| Few replicas, stable | `0.1–0.2` |
| Many similar replicas | `0.2–0.3` |
| Highly dynamic join/leave | `0.2–0.4` |

Tuning cheat sheet in-repo:
[`PARAMETER_CHEATSHEET.md`](https://github.com/goncalorafaria/literegistry/blob/master/PARAMETER_CHEATSHEET.md).

## Failover / rotation (`request_with_rotation`)

Important HTTP client knobs:

| Arg | Typical gateway value | Meaning |
|-----|----------------------|---------|
| `max_retries` | `20` (models), `3` (python), `2` (terminal) | Max attempts |
| `timeout` | `61` / `20` | Per-attempt HTTP timeout |
| `retry_budget_seconds` | set for python/terminal | Stop retrying after this wall time |
| `retry_backoff_seconds` | `0.1` for python | Sleep between attempts when set |

Flow:

1. `sample_servers(model, n=max_retries)` → ordered candidate list with probs.
2. For each attempt, pick next unused candidate (python skips previously failed
   URIs in that request).
3. On HTTP/network error: update bandit with failure, advance, optionally backoff.
4. On success: update bandit with measured latency; return JSON.

If no servers are registered for that `model_path`, the client raises immediately.

## Caching interaction

`RegistryClient` caches the roster / per-model URI lists for `cache_ttl`
(default: half of `max_heartbeat_interval`). Bandit weights are **in-process** —
each gateway worker learns independently. That is why multi-worker mode still
works: each worker explores a bit.

Force a fresh roster:

```python
await registry.models(force=True)
await registry.get_all("my-model", force=True)
```

`/health` and `/v1/models` on the gateway use `force=True`.

## Other algorithms in the tree

The package also contains `UniformBandit`, `LoadAwareExp3`, and capacity helpers
(see [`README_BANDITS.md`](https://github.com/goncalorafaria/literegistry/blob/master/README_BANDITS.md)). The **default production path
for the gateway is `Exp3Dynamic`** wired inside `RegistryClient`.

Uniform behavior (equal weights, no learning) is available if you construct a
client with a custom bandit, but the stock gateway does not expose a CLI switch
for that yet.

## Watching it live

Gateway logs periodically:

```text
Probs: [('http://nodeA:8000', 0.41), ('http://nodeB:8001', 0.35), ...]
Request counts (last 5.0s): meta-llama/...: 12, python: 3
```

The [Console](console.md) scrapes those lines for charts. You can also hit
replicas via `literegistry detail` to confirm who is registered.

## Mental model

- **Discovery** = registry heartbeats (who exists).
- **Selection** = Exp3 over URIs (who gets the next try).
- **Resilience** = rotation + retry budget (survive a bad replica).
- **Feedback** = latency/success updates (tomorrow’s probabilities).

Next: [Gateway](gateway.md) · [Registry](registry.md)
