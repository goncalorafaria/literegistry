"""Transparent Docker Registry V2 routes for the canonical LiteRegistry gateway."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
import logging
import re
import time
import uuid
from dataclasses import dataclass
from typing import Any, Mapping

import aiohttp
from starlette.requests import Request
from starlette.responses import Response, StreamingResponse
from starlette.routing import Route

from literegistry.affinity import (
    AffinityBindingConflict,
    SoftAffinityBinding,
    SoftAffinityBindingStore,
)
from literegistry.client import RegistryClient
from literegistry.gateway import GatewayRequestError


logger = logging.getLogger(__name__)
_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")
_DIGEST_RE = re.compile(r"^[A-Za-z0-9_+.-]+:[A-Za-z0-9=_-]+$")


def _request_id(request: Request) -> str:
    """Return a log-safe request correlation ID without exposing headers."""
    supplied = request.headers.get("x-request-id", "")
    if _REQUEST_ID_RE.fullmatch(supplied):
        return supplied
    return uuid.uuid4().hex[:16]

_HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}
_IMAGE_ACTIONS = frozenset({"manifests", "blobs", "tags", "referrers"})
_DOCKER_HUB_NAMESPACES = frozenset(
    {"docker.io", "registry-1.docker.io", "index.docker.io"}
)


@dataclass(frozen=True)
class _MirrorServer:
    server_id: str
    server_uri: str
    probability: float


class DockerMirrorProxy:
    """Stream Docker Registry V2 GET/HEAD requests to discovered mirrors."""

    def __init__(
        self,
        registry: RegistryClient,
        session_manager: Any,
        *,
        service: str = "docker-mirror",
        connect_timeout: float = 3.0,
        read_timeout: float = 300.0,
        max_retries: int = 3,
        bindings: SoftAffinityBindingStore | None = None,
    ) -> None:
        if not service:
            raise ValueError("docker mirror service must be non-empty")
        if connect_timeout <= 0 or read_timeout <= 0:
            raise ValueError("docker mirror timeouts must be positive")
        if max_retries < 1:
            raise ValueError("docker mirror max_retries must be at least one")
        self.registry = registry
        self.session_manager = session_manager
        self.service = service
        self.connect_timeout = connect_timeout
        self.read_timeout = read_timeout
        self.max_retries = max_retries
        self.bindings = bindings
        self.binding_service = f"{service}:image"

    @staticmethod
    def _image_affinity_id(request: Request) -> str | None:
        """Infer a stable object key from a Registry V2 request path."""
        path = request.scope.get("path") or request.url.path
        if not path.startswith("/v2/"):
            return None
        segments = [segment for segment in path[4:].split("/") if segment]
        action_index = next(
            (
                index
                for index, segment in enumerate(segments)
                if index > 0 and segment in _IMAGE_ACTIONS
            ),
            None,
        )
        if action_index is None:
            return None
        repository = "/".join(segments[:action_index]).lower()
        action = segments[action_index].lower()
        reference = "/".join(segments[action_index + 1 :])
        if not reference:
            return None
        object_key = f"{repository}/{action}/{reference}"
        namespace = request.query_params.get("ns", "").strip().lower()
        if namespace in _DOCKER_HUB_NAMESPACES:
            namespace = ""
        return f"{namespace}/{object_key}" if namespace else object_key

    @staticmethod
    def _blob_digest(request: Request) -> str | None:
        """Return the digest for an exact Registry V2 blob download path."""
        if request.method not in {"GET", "HEAD"}:
            return None
        path = request.scope.get("path") or request.url.path
        if not path.startswith("/v2/"):
            return None
        segments = [segment for segment in path[4:].split("/") if segment]
        if len(segments) < 3 or segments[-2] != "blobs":
            return None
        digest = segments[-1]
        return digest if _DIGEST_RE.fullmatch(digest) else None

    async def _live_servers(
        self,
        *,
        force: bool = False,
    ) -> dict[str, _MirrorServer] | None:
        """Return the healthy cached roster, or None for legacy clients."""
        models_method = getattr(self.registry, "models", None)
        if not callable(models_method):
            return None
        try:
            models = await models_method(force=force)
        except TypeError:
            models = await models_method()
        records = models.get(self.service, [])
        live: dict[str, _MirrorServer] = {}
        for record in records:
            if not isinstance(record, Mapping):
                continue
            uri = record.get("uri")
            server_id = record.get("server_id")
            if (
                isinstance(uri, str)
                and uri
                and isinstance(server_id, str)
                and server_id
                and record.get("status", "active") == "active"
            ):
                normalized = uri.rstrip("/")
                live[normalized] = _MirrorServer(server_id, normalized, 1.0)
        return live

    async def _sample_servers(
        self,
        *,
        force: bool = False,
    ) -> list[tuple[str, float]]:
        try:
            return await self.registry.sample_servers(
                self.service,
                n=self.max_retries,
                force=force,
            )
        except TypeError:
            return await self.registry.sample_servers(
                self.service,
                n=self.max_retries,
            )

    async def _candidates(
        self,
        affinity_id: str | None,
        *,
        force: bool = False,
    ) -> tuple[SoftAffinityBinding | None, list[_MirrorServer]]:
        affinity_enabled = self.bindings is not None and affinity_id is not None
        live = (
            await self._live_servers(force=force)
            if affinity_enabled
            else None
        )
        binding = (
            await self.bindings.resolve(self.binding_service, affinity_id)
            if affinity_enabled
            else None
        )

        ordered: list[_MirrorServer] = []
        seen: set[str] = set()
        if binding is not None:
            uri = binding.server_uri.rstrip("/")
            current = live.get(uri) if live is not None else None
            if current is not None and current.server_id == binding.server_id:
                ordered.append(current)
                seen.add(uri)

        for uri, probability in await self._sample_servers(force=force):
            normalized = uri.rstrip("/")
            if normalized in seen:
                continue
            if live is not None:
                selected = live.get(normalized)
                if selected is None:
                    continue
                selected = _MirrorServer(
                    selected.server_id,
                    selected.server_uri,
                    probability,
                )
            else:
                selected = _MirrorServer(normalized, normalized, probability)
            ordered.append(selected)
            seen.add(normalized)
            if len(ordered) >= self.max_retries:
                break
        return binding, ordered

    async def _remember(
        self,
        affinity_id: str | None,
        binding: SoftAffinityBinding | None,
        selected: _MirrorServer,
    ) -> str:
        if self.bindings is None or affinity_id is None:
            return "disabled"
        if binding is None:
            try:
                await self.bindings.bind(
                    self.binding_service,
                    affinity_id,
                    selected.server_id,
                    selected.server_uri,
                )
            except AffinityBindingConflict:
                # Another gateway worker won the first-request race. Keep its
                # valid choice so later requests converge on one mirror.
                return "race"
            return "new"
        if (
            binding.server_id == selected.server_id
            and binding.server_uri.rstrip("/") == selected.server_uri
        ):
            await self.bindings.touch(self.binding_service, affinity_id)
            return "hit"
        else:
            await self.bindings.handoff(
                self.binding_service,
                affinity_id,
                selected.server_id,
                selected.server_uri,
            )
            return "handoff"

    @staticmethod
    def _forward_request_headers(request: Request) -> list[tuple[str, str]]:
        # Keep duplicate Accept headers: containers/image uses them for OCI
        # manifest negotiation and collapsing them can select the wrong object.
        headers: list[tuple[str, str]] = []
        for raw_name, raw_value in request.scope.get("headers", []):
            name = raw_name.decode("latin-1")
            lower = name.lower()
            if lower in _HOP_BY_HOP_HEADERS or lower in {"host", "content-length"}:
                continue
            headers.append((name, raw_value.decode("latin-1")))
        return headers

    @staticmethod
    def _upstream_url(server: str, request: Request) -> str:
        # ASGI's decoded path gives Distribution canonical digest paths such as
        # ``sha256:...`` instead of forwarding Podman's escaped ``%3A``.
        path = request.scope.get("path") or request.url.path
        query = request.scope.get("query_string", b"")
        return f"{server.rstrip('/')}{path}" + (
            "?" + query.decode("ascii") if query else ""
        )

    @staticmethod
    def _response_headers(
        response: aiohttp.ClientResponse,
        *,
        server: str,
        request: Request,
    ) -> dict[str, str]:
        headers: dict[str, str] = {}
        backend = server.rstrip("/")
        gateway = str(request.base_url).rstrip("/")
        for name, value in response.headers.items():
            if name.lower() in _HOP_BY_HOP_HEADERS:
                continue
            if name.lower() == "location" and (
                value == backend or value.startswith(f"{backend}/")
            ):
                value = f"{gateway}{value[len(backend):]}"
            headers[name] = value
        return headers

    @staticmethod
    async def _stream_response(
        response: aiohttp.ClientResponse,
    ) -> AsyncIterator[bytes]:
        try:
            async for chunk in response.content.iter_chunked(1024 * 1024):
                yield chunk
        finally:
            response.release()

    def _report(
        self,
        server: str,
        latency: float,
        probability: float,
        success: bool,
    ) -> None:
        report = getattr(self.registry, "report_latency", None)
        if report is not None:
            report(server, latency, prob=probability, success=success)

    async def forward(self, request: Request) -> Response:
        request_id = _request_id(request)
        request_started = time.monotonic()
        affinity_id = self._image_affinity_id(request)
        binding, candidates = await self._candidates(affinity_id, force=False)
        if self.bindings is None or affinity_id is None:
            binding_state = "disabled"
        elif binding is None:
            binding_state = "miss"
        elif (
            candidates
            and candidates[0].server_id == binding.server_id
            and candidates[0].server_uri == binding.server_uri.rstrip("/")
        ):
            binding_state = "hit"
        else:
            binding_state = "stale"
        logger.info(
            "gateway_affinity mode=soft event=resolve request_id=%s "
            "service=%r method=%s object=%r binding=%s owner_id=%r "
            "owner_uri=%r registry_force=false candidates=%d",
            request_id,
            self.service,
            request.method,
            affinity_id or "-",
            binding_state,
            binding.server_id if binding is not None else "-",
            binding.server_uri if binding is not None else "-",
            len(candidates),
        )
        if not candidates:
            # A negative cached roster is confirmed against Redis before the
            # gateway reports that no mirror is alive.
            logger.info(
                "gateway_affinity mode=soft event=roster_refresh "
                "request_id=%s service=%r object=%r reason=no_candidates",
                request_id,
                self.service,
                affinity_id or "-",
            )
            binding, candidates = await self._candidates(affinity_id, force=True)
        if not candidates:
            logger.error(
                "gateway_affinity mode=soft event=route_failed request_id=%s "
                "service=%r object=%r reason=no_healthy_server elapsed_ms=%.3f",
                request_id,
                self.service,
                affinity_id or "-",
                (time.monotonic() - request_started) * 1000.0,
            )
            raise GatewayRequestError(
                f"No healthy {self.service} servers are registered",
                status_code=503,
            )

        timeout = aiohttp.ClientTimeout(
            total=None,
            connect=self.connect_timeout,
            sock_connect=self.connect_timeout,
            sock_read=self.read_timeout,
        )
        headers = self._forward_request_headers(request)
        session = self.session_manager.get_session()
        last_error: Exception | None = None
        attempted: set[str] = set()
        attempts = 0
        refreshed = False

        while candidates and attempts < self.max_retries:
            selected = candidates.pop(0)
            server = selected.server_uri
            if server in attempted:
                continue
            attempted.add(server)
            attempts += 1
            selection = (
                "binding"
                if binding is not None
                and binding.server_id == selected.server_id
                and binding.server_uri.rstrip("/") == selected.server_uri
                else "sample"
            )

            blob_digest = self._blob_digest(request)
            if blob_digest is not None:
                affinity_result = await self._remember(
                    affinity_id,
                    binding,
                    selected,
                )
                location = self._upstream_url(server, request)
                logger.info(
                    "gateway_affinity mode=soft event=blob_redirect "
                    "request_id=%s service=%r method=%s object=%r "
                    "decision=%s previous_server_id=%r server_id=%r "
                    "server_uri=%r selection=%s location=%r",
                    request_id,
                    self.service,
                    request.method,
                    affinity_id or "-",
                    affinity_result,
                    binding.server_id if binding is not None else "-",
                    selected.server_id,
                    selected.server_uri,
                    selection,
                    location,
                )
                return Response(
                    status_code=307,
                    headers={
                        "Location": location,
                        "Docker-Content-Digest": blob_digest,
                        "Docker-Distribution-Api-Version": "registry/2.0",
                        "Cache-Control": "no-store",
                    },
                )
            started = time.monotonic()
            response: aiohttp.ClientResponse | None = None
            try:
                response = await session.request(
                    request.method,
                    self._upstream_url(server, request),
                    headers=headers,
                    allow_redirects=False,
                    auto_decompress=False,
                    timeout=timeout,
                )
                if response.status >= 500 and attempts < self.max_retries:
                    logger.warning(
                        "gateway_affinity mode=soft event=route_retry "
                        "request_id=%s service=%r object=%r attempt=%d "
                        "server_id=%r server_uri=%r selection=%s "
                        "reason=upstream_5xx status=%d",
                        request_id,
                        self.service,
                        affinity_id or "-",
                        attempts,
                        selected.server_id,
                        selected.server_uri,
                        selection,
                        response.status,
                    )
                    if not candidates and not refreshed:
                        logger.info(
                            "gateway_affinity mode=soft event=roster_refresh "
                            "request_id=%s service=%r object=%r "
                            "reason=candidates_exhausted",
                            request_id,
                            self.service,
                            affinity_id or "-",
                        )
                        binding, fresh = await self._candidates(
                            affinity_id, force=True
                        )
                        candidates.extend(
                            candidate
                            for candidate in fresh
                            if candidate.server_uri not in attempted
                        )
                        refreshed = True
                    if candidates:
                        response.release()
                        self._report(
                            server,
                            time.monotonic() - started,
                            selected.probability,
                            False,
                        )
                        continue
                forwarded_headers = self._response_headers(
                    response,
                    server=server,
                    request=request,
                )
                healthy_response = response.status < 500
                upstream_ready_seconds = time.monotonic() - started
                self._report(
                    server,
                    upstream_ready_seconds,
                    selected.probability,
                    healthy_response,
                )
                if healthy_response:
                    affinity_result = await self._remember(
                        affinity_id,
                        binding,
                        selected,
                    )
                    logger.info(
                        "gateway_affinity mode=soft event=route_complete "
                        "request_id=%s service=%r method=%s object=%r "
                        "decision=%s previous_server_id=%r server_id=%r "
                        "server_uri=%r selection=%s attempt=%d "
                        "roster_refreshed=%s upstream_ready_ms=%.3f "
                        "elapsed_ms=%.3f status=%d",
                        request_id,
                        self.service,
                        request.method,
                        affinity_id or "-",
                        affinity_result,
                        binding.server_id if binding is not None else "-",
                        selected.server_id,
                        selected.server_uri,
                        selection,
                        attempts,
                        str(refreshed).lower(),
                        upstream_ready_seconds * 1000.0,
                        (time.monotonic() - request_started) * 1000.0,
                        response.status,
                    )
                else:
                    logger.warning(
                        "gateway_affinity mode=soft event=route_complete "
                        "request_id=%s service=%r method=%s object=%r "
                        "decision=unchanged server_id=%r server_uri=%r "
                        "selection=%s attempt=%d roster_refreshed=%s "
                        "upstream_ready_ms=%.3f elapsed_ms=%.3f status=%d",
                        request_id,
                        self.service,
                        request.method,
                        affinity_id or "-",
                        selected.server_id,
                        selected.server_uri,
                        selection,
                        attempts,
                        str(refreshed).lower(),
                        upstream_ready_seconds * 1000.0,
                        (time.monotonic() - request_started) * 1000.0,
                        response.status,
                    )
                if request.method == "HEAD":
                    status = response.status
                    response.release()
                    return Response(status_code=status, headers=forwarded_headers)
                return StreamingResponse(
                    self._stream_response(response),
                    status_code=response.status,
                    headers=forwarded_headers,
                )
            except (aiohttp.ClientError, asyncio.TimeoutError) as error:
                if response is not None:
                    response.release()
                last_error = error
                self._report(
                    server,
                    time.monotonic() - started,
                    selected.probability,
                    False,
                )
                logger.warning(
                    "gateway_affinity mode=soft event=backend_failure "
                    "request_id=%s service=%r object=%r attempt=%d "
                    "server_id=%r server_uri=%r selection=%s "
                    "error_type=%s elapsed_ms=%.3f",
                    request_id,
                    self.service,
                    affinity_id or "-",
                    attempts,
                    selected.server_id,
                    selected.server_uri,
                    selection,
                    type(error).__name__,
                    (time.monotonic() - started) * 1000.0,
                )

            if not candidates and not refreshed and attempts < self.max_retries:
                # Only failures exhaust the cached choices and trigger a full
                # roster refresh. The replacement may then receive a handoff.
                logger.info(
                    "gateway_affinity mode=soft event=roster_refresh "
                    "request_id=%s service=%r object=%r "
                    "reason=candidates_exhausted",
                    request_id,
                    self.service,
                    affinity_id or "-",
                )
                binding, fresh = await self._candidates(affinity_id, force=True)
                candidates.extend(
                    candidate
                    for candidate in fresh
                    if candidate.server_uri not in attempted
                )
                refreshed = True

        logger.error(
            "gateway_affinity mode=soft event=route_failed request_id=%s "
            "service=%r object=%r attempts=%d roster_refreshed=%s "
            "reason=all_backends_failed error_type=%s elapsed_ms=%.3f",
            request_id,
            self.service,
            affinity_id or "-",
            attempts,
            str(refreshed).lower(),
            type(last_error).__name__ if last_error is not None else "-",
            (time.monotonic() - request_started) * 1000.0,
        )
        raise GatewayRequestError(
            f"All {self.service} servers failed: {last_error}",
            status_code=502,
        )

    def install(self, app: Any) -> None:
        existing = {route.path for route in app.routes}
        routes = [
            Route("/v2", self.forward, methods=["GET", "HEAD"], name="docker_mirror_root"),
            Route("/v2/", self.forward, methods=["GET", "HEAD"], name="docker_mirror_slash_root"),
            Route("/v2/{path:path}", self.forward, methods=["GET", "HEAD"], name="docker_mirror"),
        ]
        for route in routes:
            if route.path in existing:
                raise ValueError(f"duplicate gateway route: {route.path}")
            app.router.routes.append(route)
            existing.add(route.path)

