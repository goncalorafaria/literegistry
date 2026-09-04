"""Canonical composable Starlette gateway for LiteRegistry.

Route handling, routing decisions, HTTP transport, and strict affinity are
separate components so stateful services such as Podman use the same public
gateway as ordinary model and tool workers.

Run it with uvicorn's factory mode::

    uvicorn literegistry.gateway:create_app --factory --host 0.0.0.0 --port 8080
"""

from __future__ import annotations

import asyncio
import fcntl
import hashlib
import logging
import os
import socket
import tempfile
import time
from collections import Counter, defaultdict, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping, MutableMapping, Optional, Protocol, Sequence

from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route
import uvicorn

from literegistry import get_kvstore, head_registry_uri
from literegistry.client import RegistryClient
from literegistry.http import HTTPResponseError, RegistryHTTPClient
from literegistry.registry import ServerRegistry
from literegistry.shared_session import SharedSessionManager, get_session_manager


logger = logging.getLogger(__name__)


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value")

Payload = dict[str, Any]
PreparedRequest = tuple[str, Payload]
RequestPreparer = Callable[[Payload], PreparedRequest]
ResponseMapper = Callable[[Any], Response]


class GatewayRequestError(ValueError):
    """A client request that cannot be forwarded."""

    def __init__(
        self,
        message: str,
        status_code: int = 400,
        *,
        response_body: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.response_body = dict(response_body) if response_body is not None else None


@dataclass(frozen=True)
class RetryConfig:
    """HTTP and retry settings for one class of upstream request."""

    timeout: float = 62.0
    connect_timeout: float = 10.0
    max_retries: int = 20
    retry_budget_seconds: Optional[float] = None
    retry_backoff_seconds: Optional[float] = None

    def __post_init__(self) -> None:
        if self.timeout <= 0:
            raise ValueError("timeout must be greater than zero")
        if self.connect_timeout <= 0:
            raise ValueError("connect_timeout must be greater than zero")
        if self.max_retries <= 0:
            raise ValueError("max_retries must be greater than zero")
        if self.retry_budget_seconds is not None and self.retry_budget_seconds <= 0:
            raise ValueError("retry_budget_seconds must be greater than zero")
        if self.retry_backoff_seconds is not None and self.retry_backoff_seconds < 0:
            raise ValueError("retry_backoff_seconds cannot be negative")

    def client_kwargs(self) -> dict[str, Any]:
        """Return arguments understood by :class:`RegistryHTTPClient`."""
        return {
            "timeout": self.timeout,
            "connect_timeout": self.connect_timeout,
            "max_retries": self.max_retries,
            "retry_budget_seconds": self.retry_budget_seconds,
            "retry_backoff_seconds": self.retry_backoff_seconds,
            "use_shared_session": True,
        }


def default_retry_configs() -> dict[str, RetryConfig]:
    """Return fresh defaults matching the established gateway behavior."""
    return {
        "default": RetryConfig(),
        "python": RetryConfig(
            timeout=20,
            connect_timeout=3,
            max_retries=3,
            retry_budget_seconds=20,
            retry_backoff_seconds=0.1,
        ),
        "terminal": RetryConfig(
            timeout=20,
            connect_timeout=3,
            max_retries=2,
            retry_budget_seconds=20,
            retry_backoff_seconds=0.1,
        ),
        "search": RetryConfig(
            timeout=65,
            connect_timeout=3,
            max_retries=2,
            retry_budget_seconds=65,
            retry_backoff_seconds=0.1,
        ),
        "affinity": RetryConfig(
            timeout=300,
            connect_timeout=3,
            max_retries=1,
            retry_budget_seconds=300,
            retry_backoff_seconds=0.1,
        ),
    }


@dataclass(frozen=True)
class GatewayConfig:
    """Application configuration, independent of route definitions."""

    host: str = "0.0.0.0"
    port: int = 8080
    retry: Mapping[str, RetryConfig] = field(default_factory=default_retry_configs)
    cors_origins: tuple[str, ...] = ("*",)
    stats_window_seconds: float = 5.0
    affinity_ttl_seconds: float = 900.0
    docker_mirror_affinity_ttl_seconds: float = 604800.0
    docker_mirror_soft_affinity: bool = True
    advertise_host: Optional[str] = None
    registration_enabled: bool = True
    registration_instance_id: Optional[str] = None
    registration_heartbeat_interval: float = 10.0

    def __post_init__(self) -> None:
        if self.stats_window_seconds <= 0:
            raise ValueError("stats_window_seconds must be greater than zero")
        if self.docker_mirror_affinity_ttl_seconds <= 0:
            raise ValueError(
                "docker_mirror_affinity_ttl_seconds must be greater than zero"
            )
        if self.affinity_ttl_seconds <= 0:
            raise ValueError("affinity_ttl_seconds must be greater than zero")
        if self.registration_enabled and self.registration_heartbeat_interval <= 0:
            raise ValueError(
                "registration_heartbeat_interval must be greater than zero"
            )
        if self.advertise_host is not None and not self.advertise_host.strip():
            raise ValueError("advertise_host must be non-empty when supplied")
        if (
            self.registration_instance_id is not None
            and not self.registration_instance_id.strip()
        ):
            raise ValueError(
                "registration_instance_id must be non-empty when supplied"
            )

    def retry_config(self, name: str) -> RetryConfig:
        try:
            return self.retry[name]
        except KeyError as exc:
            raise ValueError(f"unknown retry configuration: {name!r}") from exc

    @classmethod
    def from_env(cls) -> "GatewayConfig":
        """Build configuration once, with environment parsing in one place."""
        retry = default_retry_configs()
        retry["default"] = RetryConfig(
            timeout=float(os.getenv("TIMEOUT", "62")),
            max_retries=int(os.getenv("MAX_RETRIES", "20")),
        )
        retry["python"] = RetryConfig(
            timeout=float(os.getenv("PYTHON_TIMEOUT", "20")),
            connect_timeout=3,
            max_retries=int(os.getenv("PYTHON_MAX_RETRIES", "3")),
            retry_budget_seconds=float(
                os.getenv("PYTHON_RETRY_BUDGET_SECONDS", "20")
            ),
            retry_backoff_seconds=0.1,
        )
        retry["terminal"] = RetryConfig(
            timeout=float(os.getenv("TERMINAL_TIMEOUT", "20")),
            connect_timeout=3,
            max_retries=int(os.getenv("TERMINAL_MAX_RETRIES", "2")),
            retry_budget_seconds=float(
                os.getenv("TERMINAL_RETRY_BUDGET_SECONDS", "20")
            ),
            retry_backoff_seconds=0.1,
        )
        retry["search"] = RetryConfig(
            timeout=float(os.getenv("SEARCH_TIMEOUT", "65")),
            connect_timeout=3,
            max_retries=int(os.getenv("SEARCH_MAX_RETRIES", "2")),
            retry_budget_seconds=float(
                os.getenv("SEARCH_RETRY_BUDGET_SECONDS", "65")
            ),
            retry_backoff_seconds=0.1,
        )
        affinity_timeout = float(os.getenv("AFFINITY_TIMEOUT", "300"))
        retry["affinity"] = RetryConfig(
            timeout=affinity_timeout,
            connect_timeout=3,
            max_retries=int(os.getenv("AFFINITY_MAX_RETRIES", "1")),
            retry_budget_seconds=affinity_timeout,
            retry_backoff_seconds=0.1,
        )
        return cls(
            host=os.getenv("HOST", "0.0.0.0"),
            port=int(os.getenv("PORT", "8080")),
            retry=retry,
            affinity_ttl_seconds=float(
                os.getenv("AFFINITY_TTL_SECONDS", "900")
            ),
            docker_mirror_affinity_ttl_seconds=float(
                os.getenv("DOCKER_MIRROR_AFFINITY_TTL_SECONDS", "604800")
            ),
            docker_mirror_soft_affinity=_env_bool(
                "DOCKER_MIRROR_SOFT_AFFINITY", True
            ),
            advertise_host=os.getenv("GATEWAY_ADVERTISE_HOST"),
            registration_enabled=_env_bool(
                "GATEWAY_REGISTRATION_ENABLED", True
            ),
            registration_instance_id=os.getenv("GATEWAY_INSTANCE_ID"),
            registration_heartbeat_interval=float(
                os.getenv("GATEWAY_HEARTBEAT_INTERVAL", "10")
            ),
        )


class GatewayRegistration:
    """Publish one gateway lifecycle, independently of its worker count.

    Uvicorn creates the application once in every worker process. A local
    advisory lock elects exactly one of those workers to own the shared
    registry record. Waiting workers take over the same stable record if the
    current owner exits.
    """

    def __init__(
        self,
        *,
        store: Any,
        advertise_host: str,
        advertise_port: int,
        instance_id: str,
        heartbeat_interval: float = 10.0,
        worker_count: int = 1,
        registry: Optional[ServerRegistry] = None,
        leader_lock_path: Optional[str | os.PathLike[str]] = None,
    ) -> None:
        if heartbeat_interval <= 0:
            raise ValueError("heartbeat_interval must be greater than zero")
        if worker_count < 1:
            raise ValueError("worker_count must be at least one")
        if not advertise_host.strip():
            raise ValueError("advertise_host must be non-empty")
        if not instance_id.strip():
            raise ValueError("instance_id must be non-empty")
        formatted_host = advertise_host.strip()
        if ":" in formatted_host and not formatted_host.startswith("["):
            formatted_host = f"[{formatted_host}]"
        self.store = store
        self.registry = registry or ServerRegistry(store=store)
        self.url = f"http://{formatted_host}"
        self.port = int(advertise_port)
        self.instance_id = instance_id.strip()
        identity_digest = hashlib.sha256(
            self.instance_id.encode("utf-8")
        ).hexdigest()[:32]
        # All workers address one public lifecycle record. The deterministic,
        # filesystem-safe identifier also makes leader handoff seamless.
        self.registry.server_id = f"gateway-{identity_digest}"
        self.heartbeat_interval = float(heartbeat_interval)
        self.worker_count = int(worker_count)
        self._leader_lock_path = Path(
            leader_lock_path
            or Path(tempfile.gettempdir())
            / f"literegistry-gateway-{identity_digest}.lock"
        )
        self._leader_lock_file: Optional[Any] = None
        self._registered = False
        self._stop_event: Optional[asyncio.Event] = None
        self._heartbeat_task: Optional[asyncio.Task[None]] = None

    def metadata(self) -> dict[str, Any]:
        return {
            "model_path": "gateway",
            "backend": "registry-gateway",
            "instance_id": self.instance_id,
            "worker_count": self.worker_count,
            "health_endpoint": "/health",
            "capabilities": [
                "service-proxy",
                "strict-affinity",
                "docker-mirror-proxy",
            ],
            "authentication": {"type": "none"},
        }

    def heartbeat_data(self) -> dict[str, Any]:
        return {
            "status": "healthy",
            "worker_count": self.worker_count,
        }

    @property
    def is_leader(self) -> bool:
        """Whether this process currently owns the gateway lifecycle."""
        return self._leader_lock_file is not None

    def _try_acquire_leadership(self) -> bool:
        if self.is_leader:
            return True
        self._leader_lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_file = self._leader_lock_path.open("a+b")
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lock_file.close()
            return False
        self._leader_lock_file = lock_file
        return True

    def _release_leadership(self) -> None:
        lock_file = self._leader_lock_file
        self._leader_lock_file = None
        if lock_file is None:
            return
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        finally:
            lock_file.close()

    async def _register_as_leader(self) -> None:
        await self.registry.register_server(self.url, self.port, self.metadata())
        self._registered = True
        logger.info(
            "Registered gateway server_id=%s instance_id=%s uri=%s:%s",
            self.registry.server_id,
            self.instance_id,
            self.url,
            self.port,
        )

    async def start(self) -> None:
        if self._heartbeat_task is not None:
            return
        if self._try_acquire_leadership():
            try:
                await self._register_as_leader()
            except BaseException:
                self._release_leadership()
                raise
        else:
            logger.info(
                "Gateway instance_id=%s is following the heartbeat owner",
                self.instance_id,
            )
        self._stop_event = asyncio.Event()
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

    async def _heartbeat_loop(self) -> None:
        stop_event = self._stop_event
        assert stop_event is not None
        while not stop_event.is_set():
            delay = (
                self.heartbeat_interval
                if self.is_leader
                else min(self.heartbeat_interval, 1.0)
            )
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=delay)
                break
            except asyncio.TimeoutError:
                pass
            if not self.is_leader:
                if not self._try_acquire_leadership():
                    continue
                try:
                    await self._register_as_leader()
                except asyncio.CancelledError:
                    raise
                except Exception:
                    self._release_leadership()
                    logger.exception("Gateway registry leadership takeover failed")
                continue
            try:
                await self.registry.heartbeat(
                    self.url,
                    self.port,
                    data=self.heartbeat_data(),
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Gateway registry heartbeat failed")

    async def stop(self) -> None:
        task = self._heartbeat_task
        if task is None:
            return
        assert self._stop_event is not None
        self._stop_event.set()
        await task
        self._heartbeat_task = None
        self._stop_event = None
        try:
            if self._registered:
                await self.registry.deregister()
                self._registered = False
        finally:
            self._release_leadership()


def json_response(body: Any) -> Response:
    return JSONResponse(body)


@dataclass(frozen=True)
class ProxyRoute:
    """Declarative description of one JSON proxy endpoint.

    ``prepare`` validates the incoming payload and returns the registry service
    plus the payload to forward.  It is the only route-specific behavior needed
    by the common request handler.
    """

    path: str
    upstream_endpoint: str
    prepare: RequestPreparer
    retry: str = "default"
    name: Optional[str] = None
    methods: tuple[str, ...] = ("POST",)
    response_mapper: ResponseMapper = json_response

    def __post_init__(self) -> None:
        if not self.path.startswith("/"):
            raise ValueError("proxy route path must start with '/'")
        if not self.upstream_endpoint.strip("/"):
            raise ValueError("upstream_endpoint must be non-empty")
        if not self.methods:
            raise ValueError("proxy route must define at least one HTTP method")

    @property
    def route_name(self) -> str:
        return self.name or self.path.strip("/").replace("/", "_") or "root"


@dataclass(frozen=True)
class RoutingRequest:
    """Routing input shared by load-balanced and future affinity policies."""

    service: str
    endpoint: str
    payload: Payload
    retry: RetryConfig
    headers: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class RoutingResponse:
    body: Any
    server_index: Optional[int] = None


class RoutingPolicy(Protocol):
    """Extension point for load-balanced, strict, and soft-affinity routing."""

    async def forward(self, request: RoutingRequest) -> RoutingResponse:
        ...


ClientFactory = Callable[..., RegistryHTTPClient]


class LoadBalancedRouting:
    """Adapter around LiteRegistry's current rotation and bandit client."""

    def __init__(
        self,
        registry: RegistryClient,
        client_factory: ClientFactory = RegistryHTTPClient,
    ) -> None:
        self.registry = registry
        self.client_factory = client_factory

    async def forward(self, request: RoutingRequest) -> RoutingResponse:
        async with self.client_factory(
            self.registry,
            request.service,
            **request.retry.client_kwargs(),
        ) as client:
            body, server_index = await client.request_with_rotation(
                request.endpoint,
                request.payload,
            )
        return RoutingResponse(body=body, server_index=server_index)


class GatewayMetrics:
    """Small in-process metrics collector kept separate from request routing."""

    def __init__(self, window_seconds: float = 5.0) -> None:
        self.window_seconds = window_seconds
        self._counts: Counter[str] = Counter()
        self._durations: MutableMapping[str, deque[tuple[float, float]]] = defaultdict(deque)

    def record(self, name: str, duration: float) -> None:
        now = time.monotonic()
        self._counts[name] += 1
        values = self._durations[name]
        values.append((now, duration))
        cutoff = now - self.window_seconds
        while values and values[0][0] < cutoff:
            values.popleft()

    def snapshot(self) -> dict[str, Any]:
        now = time.monotonic()
        cutoff = now - self.window_seconds
        durations: dict[str, Any] = {}
        for name, values in self._durations.items():
            while values and values[0][0] < cutoff:
                values.popleft()
            recent = [duration for _, duration in values]
            if recent:
                durations[name] = {
                    "count": len(recent),
                    "average_seconds": sum(recent) / len(recent),
                    "maximum_seconds": max(recent),
                }
        return {
            "total_requests": dict(self._counts),
            "window_seconds": self.window_seconds,
            "recent": durations,
        }


def _require_fields(*fields: str) -> Callable[[Payload], None]:
    def validate(payload: Payload) -> None:
        missing = [field for field in fields if field not in payload]
        if missing:
            if len(fields) == 1:
                message = f"{fields[0]} parameter required"
            else:
                message = f"{' and '.join(fields)} parameters are required"
            raise GatewayRequestError(message)

    return validate


def fixed_service(
    service: str,
    *required_fields: str,
) -> RequestPreparer:
    """Create a preparer for a fixed registry service."""
    validate = _require_fields(*required_fields)

    def prepare(payload: Payload) -> PreparedRequest:
        validate(payload)
        return service, dict(payload)

    return prepare


def service_from_field(
    field_name: str,
    *required_fields: str,
) -> RequestPreparer:
    """Create a preparer whose registry service comes from a payload field."""
    validate = _require_fields(field_name, *required_fields)

    def prepare(payload: Payload) -> PreparedRequest:
        validate(payload)
        service = payload.get(field_name)
        if not isinstance(service, str) or not service:
            raise GatewayRequestError(f"{field_name} must be a non-empty string")
        return service, dict(payload)

    return prepare


def prepare_search(payload: Payload) -> PreparedRequest:
    """Validate a search request and consume its gateway-only pool selector."""
    forwarded = dict(payload)
    service = forwarded.pop("model_path", "search")
    if not isinstance(service, str) or not service:
        raise GatewayRequestError("model_path must be a non-empty string")
    mode = forwarded.get("mode")
    if mode not in {"query", "url"}:
        raise GatewayRequestError("mode must be either 'query' or 'url'")
    required = "query" if mode == "query" else "url"
    if not forwarded.get(required):
        raise GatewayRequestError(f"{required} parameter required for {mode} mode")
    return service, forwarded


def default_proxy_routes() -> list[ProxyRoute]:
    """Return the current gateway's proxy endpoints as declarative routes."""
    return [
        ProxyRoute(
            "/v1/completions",
            "v1/completions",
            service_from_field("model"),
            name="completions",
        ),
        ProxyRoute(
            "/v1/chat/completions",
            "v1/chat/completions",
            service_from_field("model"),
            name="chat_completions",
        ),
        ProxyRoute(
            "/classify",
            "classify",
            service_from_field("model"),
            name="classify",
        ),
        ProxyRoute(
            "/python",
            "python",
            fixed_service("python", "code"),
            retry="python",
            name="python",
        ),
        ProxyRoute(
            "/terminal",
            "terminal",
            fixed_service("terminal", "contents", "command"),
            retry="terminal",
            name="terminal",
        ),
        ProxyRoute(
            "/search",
            "search",
            prepare_search,
            retry="search",
            name="search",
        ),
    ]


class Gateway:
    """Composable gateway server with injectable routes and routing policy."""

    def __init__(
        self,
        registry: RegistryClient,
        config: Optional[GatewayConfig] = None,
        routes: Optional[Sequence[ProxyRoute]] = None,
        routing: Optional[RoutingPolicy] = None,
        session_manager: Optional[SharedSessionManager] = None,
        metrics: Optional[GatewayMetrics] = None,
        strict_affinity: Optional[Any] = None,
        enable_strict_affinity: bool = True,
        docker_mirror: Optional[Any] = None,
        enable_docker_mirror: bool = True,
        registration: Optional[GatewayRegistration] = None,
    ) -> None:
        self.registry = registry
        self.config = config or GatewayConfig()
        self.proxy_routes = list(routes) if routes is not None else default_proxy_routes()
        self.routing = routing or LoadBalancedRouting(registry)
        self.session_manager = session_manager or get_session_manager()
        self.metrics = metrics or GatewayMetrics(self.config.stats_window_seconds)
        self.registration = registration
        self._validate_routes()
        self.app = self._create_app()
        self.strict_affinity = None
        if enable_strict_affinity:
            self.install_strict_affinity(strict_affinity)
        self.docker_mirror = None
        if enable_docker_mirror:
            self.install_docker_mirror(docker_mirror)

    def install_strict_affinity(self, strict_affinity: Optional[Any] = None) -> Any:
        """Install the package's handshake/podman/close affinity API."""
        if self.strict_affinity is not None:
            if strict_affinity is None or strict_affinity is self.strict_affinity:
                return self.strict_affinity
            raise ValueError("strict affinity is already installed")

        if strict_affinity is None:
            from literegistry.affinity import StrictAffinityBindingStore
            from literegistry.gateway.affinity import StrictAffinityGateway

            strict_affinity = StrictAffinityGateway(
                self.registry,
                StrictAffinityBindingStore(
                    self.registry.store,
                    default_ttl_seconds=self.config.affinity_ttl_seconds,
                ),
                retry=self.config.retry.get(
                    "affinity", self.config.retry_config("default")
                ),
            )
        strict_affinity.install(self.app)
        self.strict_affinity = strict_affinity
        self.app.state.strict_affinity = strict_affinity
        return strict_affinity

    def install_docker_mirror(self, docker_mirror: Optional[Any] = None) -> Any:
        """Install transparent Docker Registry V2 mirror routes."""
        if self.docker_mirror is not None:
            if docker_mirror is None or docker_mirror is self.docker_mirror:
                return self.docker_mirror
            raise ValueError("docker mirror routes are already installed")

        if docker_mirror is None:
            from literegistry.affinity import SoftAffinityBindingStore
            from literegistry.gateway.mirror import DockerMirrorProxy
            if self.config.docker_mirror_soft_affinity:
                logger.warning(
                    "Experimental Docker mirror soft affinity is enabled"
                )
            docker_mirror = DockerMirrorProxy(
                self.registry,
                self.session_manager,
                service=os.getenv("DOCKER_MIRROR_SERVICE", "docker-mirror"),
                connect_timeout=float(
                    os.getenv("DOCKER_MIRROR_CONNECT_TIMEOUT", "3")
                ),
                read_timeout=float(
                    os.getenv("DOCKER_MIRROR_READ_TIMEOUT", "300")
                ),
                max_retries=int(os.getenv("DOCKER_MIRROR_MAX_RETRIES", "3")),
                bindings=(
                    SoftAffinityBindingStore(
                        self.registry.store,
                        default_ttl_seconds=self.config.docker_mirror_affinity_ttl_seconds,
                    )
                    if (
                        self.config.docker_mirror_soft_affinity
                        and getattr(self.registry, "store", None) is not None
                    )
                    else None
                ),
            )
        docker_mirror.install(self.app)
        self.docker_mirror = docker_mirror
        self.app.state.docker_mirror = docker_mirror
        return docker_mirror

    def _validate_routes(self) -> None:
        management_paths = {"/health", "/session-stats", "/gateway-stats", "/v1/models"}
        seen = set(management_paths)
        for route in self.proxy_routes:
            self.config.retry_config(route.retry)
            if route.path in seen:
                raise ValueError(f"duplicate gateway route: {route.path}")
            seen.add(route.path)

    async def health_check(self, request: Request) -> Response:
        try:
            models = await self.registry.models(force=True)
            return JSONResponse(
                {
                    "status": "healthy",
                    "service": "registry-gateway",
                    "models_count": len(models),
                }
            )
        except Exception as exc:
            logger.warning("Gateway health check failed: %s", exc)
            return JSONResponse(
                {"status": "unhealthy", "error": str(exc)},
                status_code=503,
            )

    async def list_models(self, request: Request) -> Response:
        models_data = await self.registry.models(force=True)
        models = list(models_data)
        return JSONResponse(
            {
                "models": models,
                "status": "success",
                "data": [
                    {"id": model, "metadata": models_data[model]}
                    for model in models
                ],
            }
        )

    async def session_stats(self, request: Request) -> Response:
        stats: dict[str, Any] = {
            "shared_session_initialized": self.session_manager.is_initialized,
            "architecture": "single_shared_session",
        }
        if self.session_manager.is_initialized:
            session = self.session_manager.get_session()
            connector = session.connector
            stats.update(
                {
                    "session_closed": session.closed,
                    "connector_limit": getattr(connector, "limit", None),
                    "connector_limit_per_host": getattr(
                        connector, "limit_per_host", None
                    ),
                }
            )
        return JSONResponse({"status": "success", "session_info": stats})

    async def gateway_stats(self, request: Request) -> Response:
        return JSONResponse({"status": "success", **self.metrics.snapshot()})

    async def handle_proxy(self, request: Request, route: ProxyRoute) -> Response:
        """Run the common parse, prepare, route, and response pipeline."""
        started = time.monotonic()
        try:
            try:
                payload = await request.json()
            except Exception as exc:
                raise GatewayRequestError("valid JSON object required") from exc
            if not isinstance(payload, dict):
                raise GatewayRequestError("JSON object required")

            service, forwarded_payload = route.prepare(payload)
            routing_request = RoutingRequest(
                service=service,
                endpoint=route.upstream_endpoint,
                payload=forwarded_payload,
                retry=self.config.retry_config(route.retry),
                headers=dict(request.headers),
            )
            result = await self.routing.forward(routing_request)
            return route.response_mapper(result.body)
        finally:
            self.metrics.record(route.route_name, time.monotonic() - started)

    def _route_endpoint(self, route: ProxyRoute) -> Callable[[Request], Awaitable[Response]]:
        async def endpoint(request: Request) -> Response:
            return await self.handle_proxy(request, route)

        endpoint.__name__ = f"proxy_{route.route_name}"
        return endpoint

    def _create_app(self) -> Starlette:
        @asynccontextmanager
        async def lifespan(app: Starlette):
            app.state.gateway = self
            owns_session = not self.session_manager.is_initialized
            registration_started = False
            if owns_session:
                await self.session_manager.initialize()
            try:
                if self.registration is not None:
                    await self.registration.start()
                    registration_started = True
                yield
            finally:
                if registration_started:
                    await self.registration.stop()
                if owns_session:
                    await self.session_manager.shutdown()
                await self._close_registry()

        routes = [
            Route("/health", self.health_check, methods=["GET"]),
            Route("/session-stats", self.session_stats, methods=["GET"]),
            Route("/gateway-stats", self.gateway_stats, methods=["GET"]),
            Route("/v1/models", self.list_models, methods=["GET"]),
        ]
        routes.extend(
            Route(
                route.path,
                self._route_endpoint(route),
                methods=list(route.methods),
                name=route.route_name,
            )
            for route in self.proxy_routes
        )
        app = Starlette(routes=routes, lifespan=lifespan)
        app.state.gateway = self
        app.state.gateway_registration = self.registration
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(self.config.cors_origins),
            allow_methods=["*"],
            allow_headers=["*"],
        )
        app.add_exception_handler(GatewayRequestError, self._request_error)
        app.add_exception_handler(HTTPResponseError, self._upstream_error)
        app.add_exception_handler(Exception, self._unexpected_error)
        return app

    async def _close_registry(self) -> None:
        store = getattr(self.registry, "store", None)
        close = getattr(store, "close", None)
        if close is not None:
            await close()

    async def _request_error(self, request: Request, exc: Exception) -> Response:
        assert isinstance(exc, GatewayRequestError)
        return JSONResponse(
            exc.response_body or {"error": str(exc), "status": "failed"},
            status_code=exc.status_code,
        )

    async def _upstream_error(self, request: Request, exc: Exception) -> Response:
        assert isinstance(exc, HTTPResponseError)
        body = exc.body
        if not isinstance(body, (dict, list)):
            body = {"error": str(body), "status": "failed"}
        return JSONResponse(body, status_code=exc.status)

    async def _unexpected_error(self, request: Request, exc: Exception) -> Response:
        logger.exception("Unhandled gateway error", exc_info=exc)
        return JSONResponse(
            {"error": str(exc), "status": "failed"},
            status_code=500,
        )


def create_app(
    config: Optional[GatewayConfig] = None,
    *,
    registry: Optional[RegistryClient] = None,
    routes: Optional[Sequence[ProxyRoute]] = None,
    routing: Optional[RoutingPolicy] = None,
    strict_affinity: Optional[Any] = None,
    enable_strict_affinity: bool = True,
    docker_mirror: Optional[Any] = None,
    enable_docker_mirror: bool = True,
    registration: Optional[GatewayRegistration] = None,
    enable_registration: Optional[bool] = None,
) -> Starlette:
    """Create a configured gateway app for uvicorn or embedding."""
    resolved_config = config or GatewayConfig.from_env()
    if registry is None:
        registry_path = os.getenv(
            "REGISTRY_PATH",
            "redis://klone-login01.hyak.local:6379",
        )
        registry = RegistryClient(
            store=get_kvstore(registry_path),
            service_type="model_path",
            cache_ttl=int(os.getenv("REGISTRY_CACHE_TTL_SECONDS", "5")),
        )
    registration_enabled = (
        resolved_config.registration_enabled
        if enable_registration is None
        else enable_registration
    )
    if registration is None and registration_enabled:
        store = getattr(registry, "store", None)
        if store is None:
            raise ValueError("gateway registration requires a registry store")
        resolved_host = (
            resolved_config.advertise_host
            or os.getenv("BEAKER_NODE_HOSTNAME")
            or socket.getfqdn()
        ).strip()
        instance_id = resolved_config.registration_instance_id or (
            f"gateway-{resolved_host}-{resolved_config.port}"
        )
        registration = GatewayRegistration(
            store=store,
            advertise_host=resolved_host,
            advertise_port=resolved_config.port,
            instance_id=instance_id,
            heartbeat_interval=(
                resolved_config.registration_heartbeat_interval
            ),
            worker_count=int(os.getenv("GATEWAY_WORKERS", "1")),
        )
    gateway = Gateway(
        registry,
        config=resolved_config,
        routes=routes,
        routing=routing,
        strict_affinity=strict_affinity,
        enable_strict_affinity=enable_strict_affinity,
        docker_mirror=docker_mirror,
        enable_docker_mirror=enable_docker_mirror,
        registration=registration,
    )
    return gateway.app


def advertised_gateway_url(
    port: int,
    advertise_host: Optional[str] = None,
) -> str:
    """Return the client-facing URL printed by Gateway at startup."""
    host = (
        advertise_host
        or os.getenv("BEAKER_NODE_HOSTNAME")
        or socket.getfqdn()
    ).strip()
    if not host:
        raise ValueError("gateway advertise host must be non-empty")
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    return f"http://{host}:{port}"


def main(
    registry: str = "redis://klone-login01.hyak.local:6379",
    head_registry: Optional[str] = None,
    host: str = "0.0.0.0",
    port: int = 8080,
    advertise_host: Optional[str] = None,
    instance_id: Optional[str] = None,
    workers: int = 1,
    register: bool = True,
    heartbeat_interval: float = 10.0,
    affinity_ttl_seconds: float = 900,
    docker_mirror_affinity_ttl_seconds: float = 604800,
    registry_cache_ttl_seconds: int = 5,
    timeout: float = 300,
    docker_mirror_service: str = "docker-mirror",
    docker_mirror_connect_timeout: float = 3,
    docker_mirror_read_timeout: float = 300,
    docker_mirror_max_retries: int = 3,
    docker_mirror_soft_affinity: bool = True,
    log_level: str = "info",
    access_log: bool = False,
    reload: bool = False,
) -> None:
    """Start Gateway with Uvicorn.

    A single worker receives an application object directly. Multiple workers
    and reload mode use Uvicorn's application-factory import string so every
    process creates its own registry client and connection pool.

    Retry settings remain centralized in :meth:`GatewayConfig.from_env` and
    can be customized with ``TIMEOUT``, ``MAX_RETRIES``, and the existing
    service-specific environment variables.
    """
    if workers < 1:
        raise ValueError("workers must be at least 1")
    if head_registry is not None:
        if not head_registry.strip():
            raise ValueError("head_registry must be non-empty when supplied")
        registry = head_registry_uri(head_registry)
    if register and heartbeat_interval <= 0:
        raise ValueError("heartbeat_interval must be greater than zero")
    if docker_mirror_affinity_ttl_seconds <= 0:
        raise ValueError(
            "docker_mirror_affinity_ttl_seconds must be greater than zero"
        )
    if affinity_ttl_seconds <= 0:
        raise ValueError("affinity_ttl_seconds must be greater than zero")
    if registry_cache_ttl_seconds < 1:
        raise ValueError("registry_cache_ttl_seconds must be at least 1")
    if timeout <= 0:
        raise ValueError("timeout must be greater than zero")
    if not docker_mirror_service:
        raise ValueError("docker_mirror_service must be non-empty")
    if docker_mirror_connect_timeout <= 0 or docker_mirror_read_timeout <= 0:
        raise ValueError("docker mirror timeouts must be positive")
    if docker_mirror_max_retries < 1:
        raise ValueError("docker_mirror_max_retries must be at least one")
    if reload and workers != 1:
        raise ValueError("reload mode requires workers=1")

    # Factory workers need connection settings to cross the process boundary.
    os.environ["REGISTRY_PATH"] = registry
    os.environ["HOST"] = host
    os.environ["PORT"] = str(port)
    resolved_advertise_host = (
        advertise_host
        or os.getenv("BEAKER_NODE_HOSTNAME")
        or socket.getfqdn()
    ).strip()
    os.environ["GATEWAY_ADVERTISE_HOST"] = resolved_advertise_host
    os.environ["GATEWAY_INSTANCE_ID"] = (
        instance_id or f"gateway-{resolved_advertise_host}-{port}"
    )
    os.environ["GATEWAY_WORKERS"] = str(workers)
    os.environ["GATEWAY_REGISTRATION_ENABLED"] = str(register)
    os.environ["GATEWAY_HEARTBEAT_INTERVAL"] = str(heartbeat_interval)
    os.environ["AFFINITY_TTL_SECONDS"] = str(affinity_ttl_seconds)
    os.environ["DOCKER_MIRROR_AFFINITY_TTL_SECONDS"] = str(
        docker_mirror_affinity_ttl_seconds
    )
    os.environ["REGISTRY_CACHE_TTL_SECONDS"] = str(registry_cache_ttl_seconds)
    os.environ["TIMEOUT"] = str(timeout)
    os.environ["AFFINITY_TIMEOUT"] = str(timeout)
    os.environ["DOCKER_MIRROR_SERVICE"] = docker_mirror_service
    os.environ["DOCKER_MIRROR_CONNECT_TIMEOUT"] = str(docker_mirror_connect_timeout)
    os.environ["DOCKER_MIRROR_READ_TIMEOUT"] = str(docker_mirror_read_timeout)
    os.environ["DOCKER_MIRROR_MAX_RETRIES"] = str(docker_mirror_max_retries)
    os.environ["DOCKER_MIRROR_SOFT_AFFINITY"] = str(
        docker_mirror_soft_affinity
    )

    print(
        f"GATEWAY_URL={advertised_gateway_url(port, resolved_advertise_host)}",
        flush=True,
    )

    use_factory = workers > 1 or reload
    app = (
        "literegistry.gateway:create_app"
        if use_factory
        else create_app(GatewayConfig.from_env())
    )
    uvicorn.run(
        app,
        host=host,
        port=port,
        workers=workers,
        factory=use_factory,
        log_level=log_level,
        access_log=access_log,
        reload=reload,
    )


__all__ = [
    "GatewayConfig",
    "GatewayMetrics",
    "GatewayRegistration",
    "Gateway",
    "GatewayRequestError",
    "LoadBalancedRouting",
    "ProxyRoute",
    "RetryConfig",
    "RoutingPolicy",
    "RoutingRequest",
    "RoutingResponse",
    "advertised_gateway_url",
    "create_app",
    "default_proxy_routes",
    "fixed_service",
    "main",
    "prepare_search",
    "service_from_field",
]


if __name__ == "__main__":
    import sys

    sys.modules["literegistry.gateway"] = sys.modules[__name__]
    import fire

    fire.Fire(main)
