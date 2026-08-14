"""OpenAI-compatible reverse proxy backed by one configured API key.

Start it directly with uvicorn::

    OPENAI_BASE_URL=https://api.openai.com \
    OPENAI_API_KEY=sk-... \
    REGISTRY=redis://registry-host:6379 \
    uvicorn literegistry.openai_proxy:create_app --factory --host 0.0.0.0 --port 8080

Or use the LiteRegistry CLI::

    literegistry openai-proxy --base_url https://api.openai.com --api_key sk-...

Clients connect to this server without an OpenAI key. Their Authorization header is
discarded and the key configured when this process was started is used upstream.
"""

from contextlib import asynccontextmanager
import asyncio
import logging
import os
import socket
import uuid
from typing import AsyncIterator, Dict, Iterable, Optional
from urllib.parse import urlsplit, urlunsplit

import aiohttp
import uvicorn
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route

from literegistry import ServerRegistry, get_kvstore


LOG = logging.getLogger(__name__)
_HOP_BY_HOP_HEADERS = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailer", "transfer-encoding", "upgrade",
}


def _normalise_base_url(base_url: str) -> str:
    """Validate and normalise an upstream HTTP(S) base URL."""
    parsed = urlsplit(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("base_url must be an absolute http:// or https:// URL")
    if parsed.query or parsed.fragment:
        raise ValueError("base_url must not contain a query string or fragment")
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path.rstrip("/"), "", ""))


def _upstream_url(base_url: str, request: Request) -> str:
    path = request.path_params.get("path", "")
    parsed = urlsplit(base_url)
    query = request.scope["query_string"].decode("latin-1")
    return urlunsplit((
        parsed.scheme, parsed.netloc,
        f"{parsed.path.rstrip('/')}/{path.lstrip('/')}",
        query, "",
    ))


def _request_headers(headers: Iterable[tuple[bytes, bytes]], api_key: str) -> Dict[str, str]:
    """Copy safe client headers and install the server's upstream credential."""
    copied = {
        key.decode("latin-1"): value.decode("latin-1")
        for key, value in headers
        if key.decode("latin-1").lower()
        not in _HOP_BY_HOP_HEADERS | {"host", "authorization"}
    }
    copied["Authorization"] = f"Bearer {api_key}"
    return copied


def _response_headers(headers) -> Dict[str, str]:
    return {key: value for key, value in headers.items() if key.lower() not in _HOP_BY_HOP_HEADERS}


class OpenAIProxyServer:
    """A transparent OpenAI API proxy that owns the upstream credential."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        timeout: float = 600.0,
        registry: Optional[str] = None,
        port: int = 8080,
        heartbeat_interval: float = 10.0,
    ):
        if not api_key:
            raise ValueError("api_key is required")
        if timeout <= 0:
            raise ValueError("timeout must be greater than zero")
        if port <= 0:
            raise ValueError("port must be greater than zero")
        if heartbeat_interval <= 0:
            raise ValueError("heartbeat_interval must be greater than zero")
        self.base_url = _normalise_base_url(base_url)
        self.api_key = api_key
        self.timeout = timeout
        self.registry_path = registry
        self.registry_url = f"http://{socket.getfqdn()}"
        self.port = port
        self.heartbeat_interval = heartbeat_interval
        self.app = self._create_app()

    def _registry_metadata(self) -> Dict[str, object]:
        return {
            "model_path": self.base_url,
            "backend": "openai-proxy",
            "upstream": self.base_url,
        }

    async def _heartbeat_loop(self, registry: ServerRegistry) -> None:
        while True:
            await asyncio.sleep(self.heartbeat_interval)
            try:
                await registry.heartbeat(self.registry_url, self.port)
            except Exception:
                LOG.exception("Proxy registry heartbeat failed")

    async def health_check(self, request: Request) -> JSONResponse:
        """Local readiness check; it never exposes the configured key."""
        return JSONResponse({"status": "healthy", "service": "openai-proxy", "upstream": self.base_url})

    async def proxy(self, request: Request) -> StreamingResponse:
        session: aiohttp.ClientSession = request.app.state.upstream_session
        url = _upstream_url(self.base_url, request)
        try:
            upstream = await session.request(
                request.method, url, data=await request.body(),
                headers=_request_headers(request.scope["headers"], self.api_key),
                allow_redirects=False,
            )
        except aiohttp.ClientError as exc:
            LOG.warning("Upstream request to %s failed: %s", url, exc)
            return JSONResponse({"error": {"message": "Upstream request failed", "type": "proxy_error"}}, status_code=502)
        except TimeoutError:
            LOG.warning("Upstream request to %s timed out", url)
            return JSONResponse({"error": {"message": "Upstream request timed out", "type": "proxy_error"}}, status_code=504)

        async def body() -> AsyncIterator[bytes]:
            try:
                async for chunk in upstream.content.iter_any():
                    yield chunk
            finally:
                upstream.release()

        return StreamingResponse(body(), status_code=upstream.status, headers=_response_headers(upstream.headers), media_type=None)

    def _create_app(self) -> Starlette:
        @asynccontextmanager
        async def lifespan(app: Starlette):
            app.state.upstream_session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout), auto_decompress=False,
            )
            registry = None
            heartbeat_task = None
            store = None
            try:
                if self.registry_path:
                    store = get_kvstore(self.registry_path)
                    registry = ServerRegistry(store)
                    registry.server_id = f"{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:8]}"
                    await registry.register_server(
                        url=self.registry_url,
                        port=self.port,
                        metadata=self._registry_metadata(),
                    )
                    app.state.registry = registry
                    heartbeat_task = asyncio.create_task(self._heartbeat_loop(registry))
                    LOG.info(
                        "Registered proxy at %s:%s as model %s",
                        self.registry_url,
                        self.port,
                        self.base_url,
                    )
                yield
            finally:
                if heartbeat_task is not None:
                    heartbeat_task.cancel()
                    try:
                        await heartbeat_task
                    except asyncio.CancelledError:
                        pass
                if registry is not None:
                    try:
                        await registry.deregister()
                    except Exception:
                        LOG.exception("Failed to deregister proxy")
                if store is not None and hasattr(store, "close"):
                    await store.close()
                await app.state.upstream_session.close()

        app = Starlette(
            routes=[
                Route("/health", self.health_check, methods=["GET"]),
                Route("/{path:path}", self.proxy, methods=["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"]),
            ],
            lifespan=lifespan,
        )
        app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
        return app


def create_app() -> Starlette:
    """Uvicorn app factory configured with OPENAI_BASE_URL and OPENAI_API_KEY."""
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com")
    api_key = os.environ.get("OPENAI_API_KEY")
    timeout = float(os.environ.get("OPENAI_PROXY_TIMEOUT", "600"))
    registry = os.environ.get("REGISTRY")
    port = int(os.environ.get("OPENAI_PROXY_PORT", "8080"))
    heartbeat_interval = float(os.environ.get("OPENAI_PROXY_HEARTBEAT_INTERVAL", "10"))
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY must be set before starting the proxy")
    return OpenAIProxyServer(
        base_url=base_url,
        api_key=api_key,
        timeout=timeout,
        registry=registry,
        port=port,
        heartbeat_interval=heartbeat_interval,
    ).app


def main(
    base_url="https://api.openai.com",
    api_key=None,
    host="0.0.0.0",
    port=8080,
    timeout=600,
    registry=None,
    heartbeat_interval=10,
    workers=1,
):
    """Run the proxy. REGISTRY registers it under model_path=<base_url>."""
    if not api_key:
        api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Pass --api-key or set OPENAI_API_KEY")
    registry = registry or os.environ.get("REGISTRY")
    workers = int(workers)
    if workers < 1:
        raise ValueError("workers must be at least one")

    if workers > 1:
        # Uvicorn workers import the factory in fresh processes, so pass every
        # boot-time setting through the environment. Each worker registers one
        # replica and maintains its own heartbeat.
        os.environ["OPENAI_BASE_URL"] = base_url
        os.environ["OPENAI_API_KEY"] = api_key
        os.environ["OPENAI_PROXY_TIMEOUT"] = str(timeout)
        os.environ["OPENAI_PROXY_PORT"] = str(port)
        os.environ["OPENAI_PROXY_HEARTBEAT_INTERVAL"] = str(heartbeat_interval)
        if registry:
            os.environ["REGISTRY"] = registry
        uvicorn.run(
            "literegistry.openai_proxy:create_app",
            factory=True,
            host=host,
            port=int(port),
            workers=workers,
            log_level="info",
        )
        return

    app = OpenAIProxyServer(
        base_url=base_url,
        api_key=api_key,
        timeout=float(timeout),
        registry=registry,
        port=int(port),
        heartbeat_interval=float(heartbeat_interval),
    ).app
    uvicorn.run(app, host=host, port=int(port), log_level="info")


if __name__ == "__main__":
    main()
