# Chat-completions soft affinity

The standard LiteRegistry gateway supports best-effort replica reuse on
`POST /v1/chat/completions`. No special route or JTC gateway is required.

**Without an `X-Session-ID` header, requests use normal load-balanced routing.**
Empty/blank session IDs and other endpoints also use normal routing. Existing
clients do not need to change. To opt into reuse, send the same session ID on
successive turns and a different ID for each independent conversation/rollout.

Bindings are scoped by model/service and session, not by chat-template identity.
The first turn chooses a least-loaded registered replica; subsequent turns
prefer the last successful replica. Busy, missing, or failed replicas may be
replaced. Only successful responses update the shared binding. Binding-store
errors do not discard successful model responses. Existing retry budgets apply.

- `CHAT_SOFT_AFFINITY` (default true): set false for ordinary routing everywhere.
- `CHAT_AFFINITY_LOAD_SLACK` (default 2): allowed extra inflight requests on the
  preferred replica compared with the least-loaded replica; 0 allows immediate overflow.
- `AFFINITY_TTL_SECONDS` (default 900): shared preference lifetime.

These settings also exist on `GatewayConfig`. Explicit custom routing policies
take precedence. Bindings are shared across workers; inflight counts are local
to each worker, not cluster-wide capacity estimates. This is a JSON-response
route, not a new streaming implementation. Stateful tool affinity is separate.

Deploy an updated gateway to use this feature; merging code does not update
already-running processes. Clients must continue using the gateway, not direct
model-replica URLs.
