"""Strict-affinity routes and routing for :mod:`literegistry.gateway`."""

from __future__ import annotations

import asyncio
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
from literegistry.shared_session import get_shared_session


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
    """Send one request to an exact server URI without load balancing.

    Transports may additionally implement ``probe(service, server_uri)``
    returning whether the exact server currently answers ``GET /health``.
    Strict affinity uses it to distinguish a stale roster from a dead owner;
    transports without it are treated as unable to confirm liveness.
    """

    async def post(
        self,
        service: str,
        server_uri: str,
        endpoint: str,
        payload: dict[str, Any],
        retry: RetryConfig,
    ) -> Any:
        ...


# How long a strict-affinity liveness probe of an off-roster owner may take.
# Only paid on the failure path (owner missing from the refreshed roster).
DEFAULT_OWNER_PROBE_TIMEOUT_S = float(
    os.getenv("GATEWAY_AFFINITY_OWNER_PROBE_TIMEOUT_SECONDS", "3")
)


class RegistryPinnedTransport:
    """Exact-server HTTP transport using LiteRegistry's shared session."""

    def __init__(
        self,
        registry,
        client_factory=RegistryHTTPClient,
        *,
        host_aliases: Optional[set[str]] = None,
        loopback_host: str = "127.0.0.1",
        probe_timeout: float = DEFAULT_OWNER_PROBE_TIMEOUT_S,
    ) -> None:
        self.registry = registry
        self.client_factory = client_factory
        self.host_aliases = host_aliases or local_hostnames()
        self.loopback_host = loopback_host
        self.probe_timeout = probe_timeout

    async def probe(self, service: str, server_uri: str) -> bool:
        """Return True if the exact server answers ``GET /health``.

        Any HTTP answer counts (including 401 from token-protected replicas):
        the question is whether the process is alive, not whether it is
        healthy. Transport failures and timeouts mean "not reachable".
        """
        request_uri = same_host_loopback_uri(
            server_uri,
            aliases=self.host_aliases,
            loopback_host=self.loopback_host,
        ).rstrip("/")
        timeout = aiohttp.ClientTimeout(total=self.probe_timeout)
        session = await get_shared_session()
        owns_session = session is None or session.closed
        if owns_session:
            session = aiohttp.ClientSession(timeout=timeout)
        try:
            async with session.get(f"{request_uri}/health", timeout=timeout) as response:
                return response.status < 500
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError):
            return False
        finally:
            if owns_session:
                await session.close()

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
    ) -> None:
        self.registry = registry
        self.bindings = bindings
        self.retry = retry or RetryConfig()
        self.transport = transport or RegistryPinnedTransport(registry)

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
            status_code=503,
        )

    async def _owner_reachable(
        self, service: str, binding: StrictAffinityBinding
    ) -> bool:
        """Ask the bound server directly whether it is alive.

        The roster is a lagging view of the fleet (heartbeat delays, registry
        hiccups, records dropped while a key read fails), so a roster miss on
        its own is not proof that the pinned owner is gone.
        """
        probe = getattr(self.transport, "probe", None)
        if probe is None:
            return False
        try:
            return bool(await probe(service, binding.server_uri))
        except Exception:  # pragma: no cover - defensive, probe already maps errors
            logger.exception("strict affinity owner probe failed for %s", binding.server_uri)
            return False

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
        except (asyncio.TimeoutError, OSError, RuntimeError) as exc:
            raise GatewayRequestError(
                "strict affinity server is unavailable",
                status_code=503,
            ) from exc

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
                liveness_check = "forced_refresh"
            except GatewayRequestError:
                # The refreshed roster does not list the owner either. Before
                # failing every request pinned to it, ask the server itself:
                # a replica whose heartbeat lapsed still holds the session
                # state, and forwarding to it is the only way to keep the
                # affinity contract. A restarted replica answers the forward
                # with its own 404 (container gone), which clients treat as
                # session loss; strict affinity never substitutes a replica.
                if not await self._owner_reachable(service, binding):
                    logger.warning(
                        "gateway_affinity mode=strict event=owner_unavailable "
                        "request_id=%s service=%r affinity_id=%r endpoint=%r "
                        "server_id=%r server_uri=%r liveness=refresh+probe",
                        request_id,
                        service,
                        affinity_id,
                        endpoint,
                        binding.server_id,
                        binding.server_uri,
                    )
                    raise
                logger.warning(
                    "gateway_affinity mode=strict event=owner_off_roster_alive "
                    "request_id=%s service=%r affinity_id=%r endpoint=%r "
                    "server_id=%r server_uri=%r liveness=probe",
                    request_id,
                    service,
                    affinity_id,
                    endpoint,
                    binding.server_id,
                    binding.server_uri,
                )
                liveness_check = "direct_probe"

        try:
            # The binding supplies the exact URI after the cached liveness check.
            # Strict affinity never substitutes another replica.
            result = await self._post(
                service,
                binding.server_uri,
                endpoint,
                forwarded,
            )
        except GatewayRequestError as exc:
            if exc.status_code != 503:
                raise
            # Only a failed cached route forces fresh discovery. If the
            # replica is still registered, retry once for transient failures.
            logger.warning(
                "gateway_affinity mode=strict event=route_retry "
                "request_id=%s service=%r affinity_id=%r endpoint=%r "
                "server_id=%r server_uri=%r reason=backend_unavailable",
                request_id,
                service,
                affinity_id,
                endpoint,
                binding.server_id,
                binding.server_uri,
            )
            liveness_check = "forced_refresh"
            try:
                await self._ensure_active(service, binding, force=True)
                result = await self._post(
                    service,
                    binding.server_uri,
                    endpoint,
                    forwarded,
                )
            except GatewayRequestError:
                logger.warning(
                    "gateway_affinity mode=strict event=route_failed "
                    "request_id=%s service=%r affinity_id=%r endpoint=%r "
                    "server_id=%r server_uri=%r",
                    request_id,
                    service,
                    affinity_id,
                    endpoint,
                    binding.server_id,
                    binding.server_uri,
                )
                raise

        if endpoint.strip("/") == "close":
            await self.bindings.release(service, affinity_id)
            binding_action = "release"
        else:
            await self.bindings.touch(service, affinity_id)
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
