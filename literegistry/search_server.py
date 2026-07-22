"""Cached query-search and direct-URL worker for LiteRegistry."""

import asyncio
import hashlib
import json
import logging
import os
import socket
import time
from dataclasses import dataclass, field
from typing import Any, Literal
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import aiohttp
import redis.asyncio as redis
from fastapi import FastAPI, HTTPException
from literegistry import ServerRegistry, get_kvstore
from pydantic import BaseModel, Field, root_validator


logger = logging.getLogger(__name__)


class SearchRequest(BaseModel):
    """A query search or a request to fetch one URL."""

    mode: Literal["query", "url"]
    query: str | None = Field(default=None, min_length=1, max_length=4096)
    url: str | None = Field(default=None, min_length=1, max_length=8192)
    num_results: int = Field(default=10, ge=1, le=100)
    parameters: dict[str, Any] = Field(default_factory=dict)

    @root_validator(skip_on_failure=True)
    def validate_mode_fields(cls, values: dict[str, Any]) -> dict[str, Any]:
        mode = values.get("mode")
        query = values.get("query")
        url = values.get("url")
        if mode == "query" and not query:
            raise ValueError("query is required when mode='query'")
        if mode == "url" and not url:
            raise ValueError("url is required when mode='url'")
        if mode == "url" and url:
            parsed = urlsplit(url)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                raise ValueError("url must be an absolute HTTP or HTTPS URL")
        return values


class SearchResponse(BaseModel):
    success: bool
    mode: Literal["query", "url"]
    data: Any = None
    cache_hit: bool = False
    execution_time: float
    upstream_status: int | None = None
    error: str | None = None


@dataclass
class SearchServerConfig:
    host: str = "0.0.0.0"
    port: int = 1214
    registry: str = "redis://klone-login01.hyak.local:6379"
    heartbeat_interval: int = 30
    provider: Literal["serper", "generic"] = "serper"
    search_api_url: str | None = "https://google.serper.dev/search"
    fetch_api_url: str | None = "https://scrape.serper.dev"
    upstream_timeout: float = 60
    cache_db: int = 1
    cache_ttl: int = 3600
    upstream_headers: dict[str, str] = field(default_factory=dict)


def _redis_url_for_db(url: str, db: int) -> str:
    """Return the registry Redis URL with its logical database replaced."""
    parsed = urlsplit(url)
    query = [(key, value) for key, value in parse_qsl(parsed.query) if key != "db"]
    return urlunsplit(
        (parsed.scheme, parsed.netloc, f"/{db}", urlencode(query), parsed.fragment)
    )


class SearchServer:
    """FastAPI worker registered under ``model_path='search'``."""

    def __init__(self, config: SearchServerConfig | None = None):
        self.config = config or SearchServerConfig()
        self.registry = ServerRegistry(store=get_kvstore(self.config.registry))
        self.url = f"http://{socket.getfqdn()}"
        self.should_run = False
        self.heartbeat_task: asyncio.Task | None = None
        self.session: aiohttp.ClientSession | None = None
        self.cache = redis.from_url(
            _redis_url_for_db(self.config.registry, self.config.cache_db),
            decode_responses=False,
        )
        self.app = FastAPI(title="LiteRegistry Search Server")
        self._install_routes()

    def _metadata(self) -> dict[str, Any]:
        return {
            "model_path": "search",
            "host": self.config.host,
            "port": self.config.port,
            "backend": "http-json",
            "extra_kwargs": {
                "modes": ["query", "url"],
                "provider": self.config.provider,
                "cache_db": self.config.cache_db,
                "cache_ttl": self.config.cache_ttl,
            },
        }

    def _payload_and_endpoint(
        self, request: SearchRequest
    ) -> tuple[dict[str, Any], str | None]:
        if request.mode == "query":
            if self.config.provider == "serper":
                payload = {
                    **request.parameters,
                    "q": request.query,
                    "num": request.num_results,
                }
            else:
                payload = {
                    **request.parameters,
                    "query": request.query,
                    "num_results": request.num_results,
                }
            return payload, self.config.search_api_url
        return {**request.parameters, "url": request.url}, self.config.fetch_api_url

    def _cache_key(
        self, mode: str, endpoint: str, payload: dict[str, Any]
    ) -> str:
        canonical = json.dumps(
            {
                "mode": mode,
                "endpoint": endpoint,
                "headers": self.config.upstream_headers,
                "payload": payload,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return f"search_cache:v1:{digest}"

    async def _cache_get(self, key: str) -> dict[str, Any] | None:
        try:
            value = await self.cache.get(key)
            if value is None:
                return None
            return json.loads(value)
        except Exception as exc:
            logger.warning("Search cache read failed: %s", exc)
            return None

    async def _cache_set(self, key: str, value: dict[str, Any]) -> None:
        if self.config.cache_ttl <= 0:
            return
        try:
            await self.cache.set(
                key,
                json.dumps(value, ensure_ascii=False, separators=(",", ":")),
                ex=self.config.cache_ttl,
            )
        except Exception as exc:
            logger.warning("Search cache write failed: %s", exc)

    async def execute(self, request: SearchRequest) -> SearchResponse:
        started = time.perf_counter()
        payload, endpoint = self._payload_and_endpoint(request)
        if not endpoint:
            raise HTTPException(
                status_code=503,
                detail=f"No upstream endpoint configured for mode {request.mode!r}",
            )

        cache_key = self._cache_key(request.mode, endpoint, payload)
        cached = await self._cache_get(cache_key)
        if cached is not None:
            return SearchResponse(
                success=True,
                mode=request.mode,
                data=cached["data"],
                cache_hit=True,
                execution_time=time.perf_counter() - started,
                upstream_status=cached.get("upstream_status"),
            )

        if self.session is None:
            raise RuntimeError("Search server has not started")

        try:
            async with self.session.post(
                endpoint,
                json=payload,
                headers=self.config.upstream_headers,
            ) as response:
                body = await response.text()
                try:
                    data = json.loads(body)
                except json.JSONDecodeError:
                    data = body
                if response.status >= 400:
                    return SearchResponse(
                        success=False,
                        mode=request.mode,
                        data=data,
                        execution_time=time.perf_counter() - started,
                        upstream_status=response.status,
                        error=f"Upstream returned HTTP {response.status}",
                    )
                cached_value = {"data": data, "upstream_status": response.status}
                await self._cache_set(cache_key, cached_value)
                return SearchResponse(
                    success=True,
                    mode=request.mode,
                    data=data,
                    execution_time=time.perf_counter() - started,
                    upstream_status=response.status,
                )
        except asyncio.TimeoutError as exc:
            raise HTTPException(status_code=504, detail="Upstream request timed out") from exc
        except aiohttp.ClientError as exc:
            raise HTTPException(
                status_code=502, detail=f"Upstream request failed: {exc}"
            ) from exc

    async def start(self) -> None:
        timeout = aiohttp.ClientTimeout(total=self.config.upstream_timeout)
        self.session = aiohttp.ClientSession(timeout=timeout)
        try:
            await self.registry.register_server(
                self.url, self.config.port, self._metadata()
            )
        except Exception:
            await self.session.close()
            self.session = None
            await self.cache.aclose()
            raise
        self.should_run = True
        self.heartbeat_task = asyncio.create_task(self._heartbeat_loop())

    async def _heartbeat_loop(self) -> None:
        while self.should_run:
            try:
                await self.registry.heartbeat(self.url, self.config.port)
            except Exception:
                logger.exception("Search server heartbeat failed")
            await asyncio.sleep(self.config.heartbeat_interval)

    async def cleanup_async(self) -> None:
        self.should_run = False
        if self.heartbeat_task:
            self.heartbeat_task.cancel()
            try:
                await self.heartbeat_task
            except asyncio.CancelledError:
                pass
        await self.registry.deregister()
        if self.session:
            await self.session.close()
            self.session = None
        await self.cache.aclose()

    def _install_routes(self) -> None:
        @self.app.post("/search", response_model=SearchResponse)
        async def search(request: SearchRequest) -> SearchResponse:
            return await self.execute(request)

        @self.app.get("/health")
        async def health() -> dict[str, Any]:
            return {
                "status": "healthy",
                "service": "search",
                "modes": ["query", "url"],
            }

        @self.app.get("/")
        async def root() -> dict[str, Any]:
            return {
                "message": "POST /search with mode='query' or mode='url'",
                "search_api_configured": bool(self.config.search_api_url),
                "fetch_api_configured": bool(self.config.fetch_api_url),
                "cache_db": self.config.cache_db,
                "cache_ttl": self.config.cache_ttl,
            }

        @self.app.on_event("startup")
        async def startup() -> None:
            await self.start()

        @self.app.on_event("shutdown")
        async def shutdown() -> None:
            await self.cleanup_async()


def main(
    host: str = "0.0.0.0",
    port: int = 1214,
    registry: str = "redis://klone-login01.hyak.local:6379",
    heartbeat_interval: int = 30,
    provider: str = "serper",
    search_api_url: str | None = None,
    fetch_api_url: str | None = None,
    upstream_timeout: float = 60,
    cache_db: int = 1,
    cache_ttl: int = 3600,
    upstream_headers_json: str | None = None,
) -> None:
    """Run a cached Serper or generic JSON search worker."""
    import uvicorn

    if provider not in {"serper", "generic"}:
        raise ValueError("provider must be either 'serper' or 'generic'")
    if provider == "serper":
        search_api_url = (
            search_api_url
            or os.getenv("SEARCH_API_URL")
            or "https://google.serper.dev/search"
        )
        fetch_api_url = (
            fetch_api_url
            or os.getenv("FETCH_API_URL")
            or "https://scrape.serper.dev"
        )
    else:
        search_api_url = search_api_url or os.getenv("SEARCH_API_URL")
        fetch_api_url = fetch_api_url or os.getenv("FETCH_API_URL")
    if not search_api_url or not fetch_api_url:
        raise ValueError(
            "Both search_api_url and fetch_api_url are required "
            "(or set SEARCH_API_URL and FETCH_API_URL)"
        )
    headers_text = upstream_headers_json or os.getenv("SEARCH_UPSTREAM_HEADERS", "{}")
    upstream_headers = json.loads(headers_text)
    if not isinstance(upstream_headers, dict):
        raise ValueError("upstream_headers_json must decode to a JSON object")
    if provider == "serper":
        serper_api_key = os.getenv("SERPER_API_KEY")
        if not serper_api_key:
            raise ValueError("SERPER_API_KEY is required for the Serper provider")
        upstream_headers.setdefault("X-API-KEY", serper_api_key)
        upstream_headers.setdefault("Content-Type", "application/json")

    config = SearchServerConfig(
        host=host,
        port=port,
        registry=registry,
        heartbeat_interval=heartbeat_interval,
        provider=provider,
        search_api_url=search_api_url,
        fetch_api_url=fetch_api_url,
        upstream_timeout=upstream_timeout,
        cache_db=cache_db,
        cache_ttl=cache_ttl,
        upstream_headers={str(key): str(value) for key, value in upstream_headers.items()},
    )
    uvicorn.run(SearchServer(config).app, host=host, port=port)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
