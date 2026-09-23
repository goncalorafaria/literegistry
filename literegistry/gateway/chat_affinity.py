"""Best-effort, per-rollout replica reuse for JSON chat completions."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import random
from collections import Counter
from functools import cached_property

from literegistry.affinity import SoftAffinityBindingStore
from literegistry.gateway import LoadBalancedRouting, RoutingResponse
from literegistry.http import RegistryHTTPClient

logger = logging.getLogger(__name__)


class ChatAffinityRouting:
    """Prefer the last successful replica, with load overflow and retry failover.

    Bindings live in the shared registry. Inflight counts are local to this
    gateway worker, as in the PrimeRL capacity router.
    """

    def __init__(self, registry, *, ttl_seconds=900, load_slack=2):
        if load_slack < 0:
            raise ValueError("chat affinity load slack must be non-negative")
        self.registry = registry
        self.ttl_seconds = ttl_seconds
        self.load_slack = load_slack
        self.active = Counter()
        self.fallback = LoadBalancedRouting(registry)

    @cached_property
    def bindings(self):
        # Ordinary routes must not require an affinity store at construction.
        return SoftAffinityBindingStore(
            self.registry.store, default_ttl_seconds=self.ttl_seconds
        )

    async def forward(self, request):
        session = next(
            (v for k, v in request.headers.items() if k.lower() == "x-session-id"), None
        )
        if (
            request.endpoint.strip("/") != "v1/chat/completions"
            or not session
            or not session.strip()
        ):
            return await self.fallback.forward(request)
        # Keep these bindings separate from strict affinity for stateful tools.
        key = "gateway-chat:" + request.service
        binding = None
        try:
            binding = await asyncio.wait_for(
                self.bindings.resolve(key, session), timeout=2
            )
        except Exception:
            logger.warning(
                "Chat affinity lookup failed; selecting an available replica",
                exc_info=True,
            )
        selection = _Selection(
            self, request.service, binding.server_uri if binding else None
        )
        async with _ChatClient(
            selection, request.service, **request.retry.client_kwargs()
        ) as client:
            body, index = await client.request_with_rotation(
                request.endpoint, request.payload
            )
        # Only successful completions change the preferred replica.
        try:
            await asyncio.wait_for(
                self._save_binding(key, session, selection.successful), timeout=2
            )
        except Exception:
            logger.warning(
                "Chat affinity save failed; returning successful model response",
                exc_info=True,
            )
        logger.info(
            "CHAT_AFFINITY session=%s preferred=%s selected=%s reused=%s",
            hashlib.sha256(session.encode()).hexdigest()[:16],
            binding.server_uri if binding else None,
            selection.successful,
            bool(binding and binding.server_uri == selection.successful),
        )
        return RoutingResponse(body=body, server_index=index)

    async def _save_binding(self, key, session, server):
        if await self.bindings.handoff(key, session, server, server) is None:
            await self.bindings.bind(key, session, server, server)


class _Selection:
    def __init__(self, routing, model, preferred):
        self.routing = routing
        self.model = model
        self.preferred = preferred
        self.failed = set()
        self.successful = None

    async def sample_servers(self, value, n=1, force=False):
        servers = list(
            dict.fromkeys(
                uri.rstrip("/")
                for uri in await self.routing.registry.get_all(value, force=force)
            )
        )
        candidates = [uri for uri in servers if uri not in self.failed]
        if not candidates:
            return []
        loads = self.routing.active
        minimum = min(loads[(value, uri)] for uri in candidates)
        if (
            self.preferred in candidates
            and loads[(value, self.preferred)] <= minimum + self.routing.load_slack
        ):
            selected = self.preferred
        else:
            selected = random.choice(
                [uri for uri in candidates if loads[(value, uri)] == minimum]
            )
        return [(selected, 1.0)]

    def report_latency(self, server, latency, *, prob, success):
        self.routing.registry.report_latency(
            server, latency, prob=prob, success=success
        )
        if success:
            self.successful = server
        else:
            self.failed.add(server)


class _ChatClient(RegistryHTTPClient):
    async def _make_http_request(self, server, endpoint, payload):
        key = (self.value, server)
        self.registry.routing.active[key] += 1
        try:
            return await super()._make_http_request(server, endpoint, payload)
        finally:
            self.registry.routing.active[key] -= 1
            if not self.registry.routing.active[key]:
                del self.registry.routing.active[key]
