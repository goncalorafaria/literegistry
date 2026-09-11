"""Strict-affinity routes and routing for :mod:`literegistry.gateway`."""

from __future__ import annotations

import asyncio
from collections import OrderedDict
import json
import logging
import os
import re
import socket
import time
import uuid
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Protocol
from urllib.parse import urlsplit, urlunsplit

import aiohttp

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from literegistry.affinity import (
    AffinityBindingConflict,
    StrictAffinityBinding,
    StrictAffinityBindingStore,
)
from literegistry.gateway import GatewayRequestError, RetryConfig
from literegistry.http import HTTPResponseError, RegistryHTTPClient


logger = logging.getLogger(__name__)
_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")


def _request_id(request: Request) -> str:
    """Return a log-safe request correlation ID without exposing headers."""
    supplied = request.headers.get("x-request-id", "")
    if _REQUEST_ID_RE.fullmatch(supplied):
        return supplied
    return uuid.uuid4().hex[:16]


def local_hostnames() -> set[str]:
    """Return hostnames that identify the gateway's own network namespace."""
    aliases = {"localhost", "127.0.0.1", "::1"}
    for name in (socket.gethostname(), socket.getfqdn()):
        if name:
            aliases.add(name.rstrip(".").lower())
    aliases.update(
        alias.strip().rstrip(".").lower()
        for alias in os.getenv("GATEWAY_LOCAL_HOST_ALIASES", "").split(",")
        if alias.strip()
    )
    return aliases


def same_host_loopback_uri(
    server_uri: str,
    *,
    aliases: Optional[set[str]] = None,
    loopback_host: str = "127.0.0.1",
) -> str:
    """Rewrite a same-host server URI to loopback while preserving its port."""
    parsed = urlsplit(server_uri)
    hostname = parsed.hostname
    known_local = aliases if aliases is not None else local_hostnames()
    if hostname is None or hostname.rstrip(".").lower() not in known_local:
        return server_uri

    host = f"[{loopback_host}]" if ":" in loopback_host else loopback_host
    netloc = f"{host}:{parsed.port}" if parsed.port is not None else host
    return urlunsplit(parsed._replace(netloc=netloc))


class PinnedTransport(Protocol):
    """Send one request to an exact server URI without load balancing."""

    async def post(
        self,
        service: str,
        server_uri: str,
        endpoint: str,
        payload: dict[str, Any],
        retry: RetryConfig,
    ) -> Any:
        ...


class RegistryPinnedTransport:
    """Exact-server HTTP transport using LiteRegistry's shared session."""

    def __init__(
        self,
        registry,
        client_factory=RegistryHTTPClient,
        *,
        host_aliases: Optional[set[str]] = None,
        loopback_host: str = "127.0.0.1",
    ) -> None:
        self.registry = registry
        self.client_factory = client_factory
        self.host_aliases = host_aliases or local_hostnames()
        self.loopback_host = loopback_host

    async def post(
        self,
        service: str,
        server_uri: str,
        endpoint: str,
        payload: dict[str, Any],
        retry: RetryConfig,
    ) -> Any:
        request_uri = same_host_loopback_uri(
            server_uri,
            aliases=self.host_aliases,
            loopback_host=self.loopback_host,
        )
        if request_uri != server_uri:
            logger.debug("routing same-host server %s via %s", server_uri, request_uri)

        async with self.client_factory(
            self.registry,
            service,
            **retry.client_kwargs(),
        ) as client:
            # Selection is intentionally bypassed: affinity already chose the
            # exact server. RegistryHTTPClient owns session and timeout setup.
            return await client.request_server(
                request_uri,
                endpoint,
                payload,
            )


    async def probe(self, server_uri: str, endpoint: str, timeout: float) -> tuple[int, Any]:
        """Bounded read-only request to the same owner; never follow redirects."""
        request_uri = same_host_loopback_uri(
            server_uri, aliases=self.host_aliases, loopback_host=self.loopback_host
        )
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
            async with session.get(
                f"{request_uri.rstrip('/')}/{endpoint}", allow_redirects=False
            ) as response:
                raw = await response.content.read(16_385)
                if len(raw) > 16_384:
                    return response.status, None
                try:
                    body = json.loads(raw)
                except (ValueError, UnicodeDecodeError):
                    body = None
                return response.status, body


class _TransportFailure(GatewayRequestError):
    """No upstream application response; execution may already have happened."""


@dataclass(frozen=True)
class SelectedServer:
    server_id: str
    server_uri: str


class StrictAffinityGateway:
    """Handshake and pinned forwarding for strict-affinity services."""

    def __init__(
        self,
        registry,
        bindings: StrictAffinityBindingStore,
        retry: Optional[RetryConfig] = None,
        transport: Optional[PinnedTransport] = None,
        *,
        probe_timeout: float = 1.0,
        probe_cooldown: float = 2.0,
        max_owner_probes: int = 256,
    ) -> None:
        self.registry = registry
        self.bindings = bindings
        self.retry = retry or RetryConfig()
        self.transport = transport or RegistryPinnedTransport(registry)
        if not (0 < probe_timeout < float("inf") and 0 < probe_cooldown < float("inf")):
            raise ValueError("probe timeout and cooldown must be positive and finite")
        if max_owner_probes < 1:
            raise ValueError("max_owner_probes must be positive")
        self.probe_timeout = probe_timeout
        self.probe_cooldown = probe_cooldown
        self.max_owner_probes = max_owner_probes
        self._owner_probes: OrderedDict[
            tuple[str, str, str], tuple[float, asyncio.Task[str]]
        ] = OrderedDict()
        self._session_probe_slots = asyncio.Semaphore(16)

    @staticmethod
    async def _payload(request: Request) -> dict[str, Any]:
        try:
            payload = await request.json()
        except Exception as exc:
            raise GatewayRequestError("valid JSON object required") from exc
        if not isinstance(payload, dict):
            raise GatewayRequestError("JSON object required")
        return payload

    @staticmethod
    def _service(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        forwarded = dict(payload)
        service = forwarded.pop("service", None)
        if not isinstance(service, str) or not service:
            raise GatewayRequestError("service must be a non-empty string")
        return service, forwarded

    async def _records(self, service: str, force: bool = False) -> list[Mapping[str, Any]]:
        models = await self.registry.models(force=force)
        records = models.get(service, [])
        return [record for record in records if isinstance(record, Mapping)]

    async def _candidate_servers(
        self,
        service: str,
        *,
        force: bool = False,
    ) -> list[SelectedServer]:
        # RegistryClient owns the short-lived roster cache. Only the fallback
        # path after failed cached candidates bypasses it.
        records = await self._records(service, force=force)
        by_uri: dict[str, SelectedServer] = {}
        for record in records:
            uri = record.get("uri")
            server_id = record.get("server_id")
            if isinstance(uri, str) and uri and isinstance(server_id, str) and server_id:
                normalized = uri.rstrip("/")
                by_uri[normalized] = SelectedServer(server_id, normalized)
        if not by_uri:
            return []

        # _records() populated the models cache, so this does not cause a
        # second roster scan even when _records() was explicitly refreshed.
        preferred = await self.registry.sample_servers(
            service, n=1, force=False
        )
        ordered: list[SelectedServer] = []
        seen: set[str] = set()
        for uri, _ in preferred:
            normalized = uri.rstrip("/")
            selected = by_uri.get(normalized)
            if selected is not None and normalized not in seen:
                ordered.append(selected)
                seen.add(normalized)
        for uri, selected in by_uri.items():
            if uri not in seen:
                ordered.append(selected)
        return ordered

    async def _ensure_active(
        self,
        service: str,
        binding: StrictAffinityBinding,
        *,
        force: bool = False,
    ) -> None:
        for record in await self._records(service, force=force):
            if (
                record.get("server_id") == binding.server_id
                and isinstance(record.get("uri"), str)
                and record["uri"].rstrip("/") == binding.server_uri.rstrip("/")
            ):
                return
        raise GatewayRequestError(
            "strict affinity server is no longer registered",
            status_code=410,
            response_body={
                "error": "strict affinity server is no longer registered",
                "code": "affinity_owner_lost",
                "recoverable": False,
            },
        )

    async def _post(
        self,
        service: str,
        server_uri: str,
        endpoint: str,
        payload: dict[str, Any],
    ) -> Any:
        try:
            return await self.transport.post(
                service,
                server_uri,
                endpoint,
                payload,
                self.retry,
            )
        except HTTPResponseError:
            raise
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as exc:
            if isinstance(exc, asyncio.TimeoutError):
                status, code = 504, "affinity_upstream_timeout"
            elif isinstance(exc, aiohttp.ClientConnectorError) or not isinstance(exc, aiohttp.ClientError):
                status, code = 503, "affinity_owner_unavailable"
            else:
                status, code = 502, "affinity_upstream_disconnected"
            raise _TransportFailure(
                "strict affinity upstream is unavailable",
                status_code=status,
                response_body={
                    "error": "strict affinity upstream is unavailable",
                    "code": code,
                    "execution_outcome": (
                        "not_sent" if isinstance(exc, aiohttp.ClientConnectorError) else "unknown"
                    ),
                },
            ) from exc

    @staticmethod
    def _owner_key(service: str, binding: StrictAffinityBinding) -> tuple[str, str, str]:
        return service, binding.server_id, binding.server_uri

    @staticmethod
    def _owner_lost() -> GatewayRequestError:
        return GatewayRequestError(
            "strict affinity server is no longer registered", status_code=410,
            response_body={"error": "strict affinity server is no longer registered",
                           "code": "affinity_owner_lost", "recoverable": False},
        )

    def _cached_owner_state(self, service: str, binding: StrictAffinityBinding) -> Optional[str]:
        entry = self._owner_probes.get(self._owner_key(service, binding))
        if entry is not None:
            expires, task = entry
            if expires > time.monotonic() and task.done() and not task.cancelled():
                return task.result()
        return None

    async def _probe_owner(self, service: str, binding: StrictAffinityBinding) -> str:
        key = self._owner_key(service, binding)
        now = time.monotonic()
        entry = self._owner_probes.get(key)
        if entry is not None and (entry[0] > now or not entry[1].done()):
            return await asyncio.shield(entry[1])
        self._owner_probes.pop(key, None)
        # Never evict a running probe: waiters must share the same operation.
        for old_key, (expires, task) in list(self._owner_probes.items()):
            if task.done() and (expires <= now or len(self._owner_probes) >= self.max_owner_probes):
                self._owner_probes.pop(old_key)
        if len(self._owner_probes) >= self.max_owner_probes:
            return "unknown"

        async def check() -> str:
            try:
                await asyncio.wait_for(
                    self._ensure_active(service, binding, force=True), self.probe_timeout
                )
            except GatewayRequestError as exc:
                if exc.status_code == 410:
                    return "lost"
                raise
            probe = getattr(self.transport, "probe", None)
            if probe is None:
                return "unknown"  # Compatibility with custom POST-only transports.
            try:
                status, _ = await asyncio.wait_for(
                    probe(binding.server_uri, "health", self.probe_timeout), self.probe_timeout
                )
            except (aiohttp.ClientError, asyncio.TimeoutError, OSError):
                return "unavailable"
            if status == 200:
                return "healthy"
            return "unavailable" if status == 503 else "unknown"

        async def bounded_check() -> str:
            try:
                outcome = await asyncio.wait_for(check(), 2 * self.probe_timeout)
            except Exception:
                # A failed/slow registry read cannot prove permanent owner loss.
                outcome = "unknown"
            # Cooldown starts on completion, not while the probe is running.
            self._owner_probes[key] = (time.monotonic() + self.probe_cooldown, asyncio.current_task())
            logger.warning(
                "gateway_affinity mode=strict event=owner_probe service=%r "
                "server_id=%r server_uri=%r outcome=%s",
                service, binding.server_id, binding.server_uri, outcome,
            )
            return outcome

        task = asyncio.create_task(bounded_check())
        self._owner_probes[key] = (now + self.probe_cooldown, task)
        return await asyncio.shield(task)

    async def _probe_session(
        self, binding: StrictAffinityBinding, container_id: str
    ) -> Optional[dict[str, Any]]:
        probe = getattr(self.transport, "probe", None)
        # Podman IDs are canonical; never interpolate arbitrary affinity IDs into URLs.
        if probe is None or re.fullmatch(r"[0-9a-f]{64}", container_id) is None:
            return None

        async def check():
            async with self._session_probe_slots:
                return await probe(binding.server_uri, f"sessions/{container_id}", self.probe_timeout)

        try:
            status, body = await asyncio.wait_for(check(), self.probe_timeout)
        except Exception:
            return None
        if not isinstance(body, dict):
            return None
        detail = body.get("detail", body)
        if not isinstance(detail, dict):
            return None
        if (status == 410 and detail.get("code") == "sandbox_lost"
                and detail.get("recoverable") is False
                and detail.get("container_id") == container_id):
            return detail
        if (status == 404 and detail.get("error") == "container_not_found"
                and detail.get("container_id") == container_id):
            return {"code": "sandbox_lost", "reason": "unknown",
                    "container_id": container_id, "recoverable": False}
        return None

    async def handshake(self, request: Request) -> Response:
        request_id = _request_id(request)
        started = time.monotonic()
        payload = await self._payload(request)
        service, forwarded = self._service(payload)
        logger.info(
            "gateway_affinity mode=strict event=handshake_start "
            "request_id=%s service=%r",
            request_id,
            service,
        )
        last_unavailable: Optional[GatewayRequestError] = None
        # Use cached discovery first. A complete registry refresh happens only
        # if every cached candidate fails or the cached roster is empty.
        for force in (False, True):
            candidates = await self._candidate_servers(service, force=force)
            logger.info(
                "gateway_affinity mode=strict event=roster "
                "request_id=%s service=%r registry_force=%s candidates=%d",
                request_id,
                service,
                str(force).lower(),
                len(candidates),
            )
            for attempt, selected in enumerate(candidates, start=1):
                try:
                    result = await self._post(
                        service,
                        selected.server_uri,
                        "handshake",
                        forwarded,
                    )
                except GatewayRequestError as exc:
                    if exc.status_code == 503:
                        last_unavailable = exc
                        logger.warning(
                            "gateway_affinity mode=strict event=handshake_retry "
                            "request_id=%s service=%r registry_force=%s attempt=%d "
                            "server_id=%r server_uri=%r reason=unavailable",
                            request_id,
                            service,
                            str(force).lower(),
                            attempt,
                            selected.server_id,
                            selected.server_uri,
                        )
                        continue
                    raise

                affinity_id = (
                    result.get("affinity_id") if isinstance(result, dict) else None
                )
                if not isinstance(affinity_id, str) or not affinity_id:
                    raise GatewayRequestError(
                        "affinity handshake returned no affinity_id",
                        status_code=502,
                    )
                try:
                    await self.bindings.bind(
                        service,
                        affinity_id,
                        selected.server_id,
                        selected.server_uri,
                    )
                except AffinityBindingConflict as exc:
                    logger.warning(
                        "gateway_affinity mode=strict event=bind_conflict "
                        "request_id=%s service=%r affinity_id=%r "
                        "server_id=%r server_uri=%r",
                        request_id,
                        service,
                        affinity_id,
                        selected.server_id,
                        selected.server_uri,
                    )
                    raise GatewayRequestError(str(exc), status_code=409) from exc
                logger.info(
                    "gateway_affinity mode=strict event=bound "
                    "request_id=%s service=%r affinity_id=%r "
                    "server_id=%r server_uri=%r registry_force=%s attempt=%d "
                    "elapsed_ms=%.3f",
                    request_id,
                    service,
                    affinity_id,
                    selected.server_id,
                    selected.server_uri,
                    str(force).lower(),
                    attempt,
                    (time.monotonic() - started) * 1000.0,
                )
                return JSONResponse(result)

        logger.error(
            "gateway_affinity mode=strict event=handshake_failed "
            "request_id=%s service=%r reason=no_available_server "
            "elapsed_ms=%.3f",
            request_id,
            service,
            (time.monotonic() - started) * 1000.0,
        )
        raise GatewayRequestError(
            f"no available servers for affinity service {service}",
            status_code=503,
        ) from last_unavailable

    async def forward(self, request: Request, endpoint: str) -> Response:
        request_id = _request_id(request)
        started = time.monotonic()
        payload = await self._payload(request)
        service, forwarded = self._service(payload)
        affinity_id = forwarded.get("affinity_id")
        if not isinstance(affinity_id, str) or not affinity_id:
            raise GatewayRequestError("affinity_id must be a non-empty string")
        binding = await self.bindings.resolve(service, affinity_id)
        if binding is None:
            logger.warning(
                "gateway_affinity mode=strict event=binding_miss "
                "request_id=%s service=%r affinity_id=%r endpoint=%r",
                request_id,
                service,
                affinity_id,
                endpoint,
            )
            raise GatewayRequestError(
                "strict affinity binding was not found or has expired",
                status_code=404,
            )

        liveness_check = "normal"
        try:
            # This normally checks RegistryClient's short-lived in-process
            # roster cache. A negative cached result is confirmed against
            # Redis before rejecting the pinned request.
            await self._ensure_active(service, binding, force=False)
        except GatewayRequestError:
            logger.info(
                "gateway_affinity mode=strict event=liveness_refresh "
                "request_id=%s service=%r affinity_id=%r endpoint=%r "
                "server_id=%r server_uri=%r reason=owner_not_in_normal_roster",
                request_id,
                service,
                affinity_id,
                endpoint,
                binding.server_id,
                binding.server_uri,
            )
            try:
                await self._ensure_active(service, binding, force=True)
            except GatewayRequestError:
                logger.warning(
                    "gateway_affinity mode=strict event=owner_unavailable "
                    "request_id=%s service=%r affinity_id=%r endpoint=%r "
                    "server_id=%r server_uri=%r liveness=refresh",
                    request_id,
                    service,
                    affinity_id,
                    endpoint,
                    binding.server_id,
                    binding.server_uri,
                )
                raise
            liveness_check = "forced_refresh"

        owner_state = self._cached_owner_state(service, binding)
        if owner_state == "lost":
            raise self._owner_lost()
        if owner_state == "unavailable":
            raise GatewayRequestError(
                "strict affinity owner is temporarily unavailable", status_code=503,
                response_body={"error": "strict affinity owner is temporarily unavailable",
                               "code": "affinity_owner_unavailable",
                               "execution_outcome": "not_sent", "request_id": request_id},
            )
        try:
            # A pinned command is sent once. A lost response is not permission
            # to replay a potentially side-effecting command.
            result = await self._post(service, binding.server_uri, endpoint, forwarded)
        except _TransportFailure as exc:
            owner_state = await self._probe_owner(service, binding)
            if owner_state == "lost":
                raise self._owner_lost() from exc
            if owner_state == "healthy" and endpoint.strip("/") in {"podman", "close"}:
                loss = await self._probe_session(binding, affinity_id)
                if loss is not None:
                    raise GatewayRequestError(
                        "strict affinity sandbox is lost", status_code=410,
                        response_body=loss,
                    ) from exc
            exc.response_body["request_id"] = request_id
            raise

        if endpoint.strip("/") == "close":
            await self.bindings.release(service, affinity_id)
            binding_action = "release"
        else:
            await self.bindings.refresh_binding(binding)
            binding_action = "touch"
        logger.info(
            "gateway_affinity mode=strict event=route_complete "
            "request_id=%s service=%r affinity_id=%r endpoint=%r "
            "server_id=%r server_uri=%r binding=hit action=%s "
            "liveness_check=%s elapsed_ms=%.3f",
            request_id,
            service,
            affinity_id,
            endpoint,
            binding.server_id,
            binding.server_uri,
            binding_action,
            liveness_check,
            (time.monotonic() - started) * 1000.0,
        )
        return JSONResponse(result)

    async def put(self, request: Request) -> Response:
        return await self.forward(request, "kv/put")

    async def get(self, request: Request) -> Response:
        return await self.forward(request, "kv/get")

    async def podman(self, request: Request) -> Response:
        """Execute a command on the replica selected by the handshake."""
        return await self.forward(request, "podman")

    async def close(self, request: Request) -> Response:
        """Close the upstream session and release its affinity binding."""
        return await self.forward(request, "close")

    def routes(self) -> list[Route]:
        return [
            Route("/affinity/handshake", self.handshake, methods=["POST"]),
            Route("/affinity/kv/put", self.put, methods=["POST"]),
            Route("/affinity/kv/get", self.get, methods=["POST"]),
            Route("/affinity/podman", self.podman, methods=["POST"]),
            Route("/affinity/close", self.close, methods=["POST"]),
        ]

    def install(self, app: Starlette) -> None:
        existing = {route.path for route in app.routes}
        for route in self.routes():
            if route.path in existing:
                raise ValueError(f"duplicate gateway route: {route.path}")
            app.router.routes.append(route)
            existing.add(route.path)
        app.state.strict_affinity = self


__all__ = [
    "PinnedTransport",
    "RegistryPinnedTransport",
    "SelectedServer",
    "StrictAffinityGateway",
]
