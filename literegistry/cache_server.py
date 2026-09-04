"""Registered HTTP cache service backed by a private Redis process."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import socket
from dataclasses import dataclass
from typing import Any, Literal

import fire
import redis.asyncio as redis
from fastapi import FastAPI, HTTPException
from literegistry import RegistryClient, ServerRegistry, get_kvstore
from literegistry.http import RegistryHTTPClient
from pydantic import BaseModel, Field, root_validator


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_CACHE_KEY_PREFIX = "literegistry-cache:v1:"
_EVICTION_POLICIES = {
    "allkeys-lfu",
    "allkeys-lru",
    "allkeys-random",
    "volatile-lfu",
    "volatile-lru",
    "volatile-random",
    "volatile-ttl",
}


class CacheRequest(BaseModel):
    operation: Literal["get", "set", "delete"]
    key: str = Field(..., min_length=1, max_length=1024)
    value: Any = None
    ttl_seconds: int | None = Field(default=None, ge=1, le=7 * 24 * 3600)

    @root_validator(skip_on_failure=True)
    def validate_operation(cls, values: dict[str, Any]) -> dict[str, Any]:
        operation = values.get("operation")
        if operation == "set" and values.get("value") is None:
            raise ValueError("value is required for operation='set'")
        if operation != "set" and values.get("ttl_seconds") is not None:
            raise ValueError("ttl_seconds is only valid for operation='set'")
        return values


class CacheResponse(BaseModel):
    success: bool = True
    hit: bool = False
    value: Any = None


@dataclass
class CacheServerConfig:
    host: str = "0.0.0.0"
    port: int = 1215
    registry: str = "redis://klone-login01.hyak.local:6379"
    heartbeat_interval: float = 30
    service_name: str = "cache"
    backend_redis: str | None = None
    backend_port: int = 6379
    redis_server_path: str | None = None
    maxmemory: str = "4gb"
    maxmemory_policy: str = "allkeys-lfu"
    default_ttl: int = 3600
    startup_timeout: float = 15
    max_value_bytes: int = 8 * 1024 * 1024

    def validate(self) -> None:
        if not 1 <= self.port <= 65535:
            raise ValueError("port must be between 1 and 65535")
        if not 1 <= self.backend_port <= 65535:
            raise ValueError("backend_port must be between 1 and 65535")
        if self.heartbeat_interval <= 0 or self.startup_timeout <= 0:
            raise ValueError("heartbeat_interval and startup_timeout must be positive")
        if self.default_ttl <= 0:
            raise ValueError("default_ttl must be positive")
        if self.max_value_bytes <= 0:
            raise ValueError("max_value_bytes must be positive")
        if not self.maxmemory.strip():
            raise ValueError("maxmemory must be non-empty")
        if self.maxmemory_policy not in _EVICTION_POLICIES:
            choices = ", ".join(sorted(_EVICTION_POLICIES))
            raise ValueError(f"maxmemory_policy must be one of: {choices}")


class CacheServiceClient:
    """Discover cache services through LiteRegistry and use their HTTP API."""

    def __init__(
        self,
        registry: RegistryClient,
        service_name: str = "cache",
        timeout: float = 5,
        max_retries: int = 3,
    ) -> None:
        self.http = RegistryHTTPClient(
            registry=registry,
            value=service_name,
            timeout=timeout,
            connect_timeout=min(timeout, 2),
            max_retries=max_retries,
            retry_budget_seconds=timeout,
            use_shared_session=False,
        )
        self._started = False

    async def start(self) -> None:
        if not self._started:
            await self.http.__aenter__()
            self._started = True

    async def close(self) -> None:
        if self._started:
            await self.http.__aexit__(None, None, None)
            self._started = False

    async def _request(self, request: CacheRequest) -> CacheResponse:
        if not self._started:
            raise RuntimeError("cache service client has not started")
        response, _ = await self.http.request_with_rotation(
            "cache", request.dict(exclude_none=True)
        )
        return CacheResponse.parse_obj(response)

    async def get(self, key: str) -> CacheResponse:
        return await self._request(CacheRequest(operation="get", key=key))

    async def set(self, key: str, value: Any, ttl_seconds: int) -> CacheResponse:
        return await self._request(
            CacheRequest(
                operation="set",
                key=key,
                value=value,
                ttl_seconds=ttl_seconds,
            )
        )

    async def delete(self, key: str) -> CacheResponse:
        return await self._request(CacheRequest(operation="delete", key=key))


class CacheServer:
    """FastAPI service that owns all access to its private Redis backend."""

    def __init__(self, config: CacheServerConfig | None = None):
        self.config = config or CacheServerConfig()
        self.config.validate()
        self.registry = ServerRegistry(store=get_kvstore(self.config.registry))
        self.url = f"http://{socket.getfqdn()}"
        self.cache: redis.Redis | None = None
        self.backend_process: asyncio.subprocess.Process | None = None
        self.should_run = False
        self.registered = False
        self.heartbeat_task: asyncio.Task | None = None
        self.app = FastAPI(title="LiteRegistry Cache Server")
        self._install_routes()

    @property
    def backend_url(self) -> str:
        return self.config.backend_redis or (
            f"redis://127.0.0.1:{self.config.backend_port}/0"
        )

    def _metadata(self) -> dict[str, Any]:
        return {
            "model_path": self.config.service_name,
            "host": self.config.host,
            "port": self.config.port,
            "backend": "http-cache",
            "extra_kwargs": {
                "operations": ["get", "set", "delete"],
                "storage": "redis",
                "managed_backend": self.config.backend_redis is None,
                "default_ttl": self.config.default_ttl,
                "max_value_bytes": self.config.max_value_bytes,
            },
        }

    def _backend_command(self) -> list[str]:
        executable = self.config.redis_server_path or os.getenv("REDIS_SERVER_PATH")
        if executable:
            executable = os.path.expanduser(executable)
        else:
            executable = shutil.which("redis-server")
        if not executable:
            raise RuntimeError(
                "redis-server was not found; use the existing Redis image, install "
                "redis-server, set REDIS_SERVER_PATH, or pass --redis-server-path"
            )
        return [
            executable,
            "--bind",
            "127.0.0.1",
            "--protected-mode",
            "yes",
            "--save",
            "",
            "--appendonly",
            "no",
            "--port",
            str(self.config.backend_port),
            "--maxmemory",
            self.config.maxmemory,
            "--maxmemory-policy",
            self.config.maxmemory_policy,
        ]

    async def _start_backend(self) -> None:
        if self.config.backend_redis is None:
            command = self._backend_command()
            logger.info(
                "Starting private Redis cache backend port=%s maxmemory=%s policy=%s",
                self.config.backend_port,
                self.config.maxmemory,
                self.config.maxmemory_policy,
            )
            self.backend_process = await asyncio.create_subprocess_exec(*command)
        self.cache = redis.from_url(self.backend_url, decode_responses=False)
        deadline = asyncio.get_running_loop().time() + self.config.startup_timeout
        last_error: Exception | None = None
        while asyncio.get_running_loop().time() < deadline:
            if self.backend_process and self.backend_process.returncode is not None:
                raise RuntimeError(
                    f"private Redis backend exited with code "
                    f"{self.backend_process.returncode}"
                )
            try:
                await self.cache.ping()
                return
            except Exception as exc:
                last_error = exc
                await asyncio.sleep(0.1)
        raise RuntimeError(
            f"Redis cache backend did not become ready at {self.backend_url}: "
            f"{last_error}"
        )

    async def _stop_backend(self) -> None:
        if self.cache is not None:
            await self.cache.aclose()
            self.cache = None
        process = self.backend_process
        self.backend_process = None
        if process is None or process.returncode is not None:
            return
        process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=5)
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()

    @staticmethod
    def _storage_key(key: str) -> str:
        return f"{_CACHE_KEY_PREFIX}{key}"

    async def execute(self, request: CacheRequest) -> CacheResponse:
        if self.cache is None:
            raise HTTPException(status_code=503, detail="cache backend is not ready")
        key = self._storage_key(request.key)
        try:
            if request.operation == "get":
                encoded = await self.cache.get(key)
                if encoded is None:
                    return CacheResponse(hit=False)
                return CacheResponse(hit=True, value=json.loads(encoded))
            if request.operation == "delete":
                deleted = bool(await self.cache.delete(key))
                return CacheResponse(hit=deleted)

            encoded = json.dumps(
                request.value, ensure_ascii=False, separators=(",", ":")
            ).encode("utf-8")
            if len(encoded) > self.config.max_value_bytes:
                raise HTTPException(
                    status_code=413,
                    detail=(
                        f"cache value exceeds {self.config.max_value_bytes} bytes"
                    ),
                )
            ttl = request.ttl_seconds or self.config.default_ttl
            await self.cache.set(key, encoded, ex=ttl)
            return CacheResponse(hit=True)
        except HTTPException:
            raise
        except (TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=400, detail=f"cache value is not valid JSON: {exc}"
            ) from exc
        except Exception as exc:
            logger.warning("Cache backend operation failed: %s", exc)
            raise HTTPException(
                status_code=503, detail="cache backend is unavailable"
            ) from exc

    async def start(self) -> None:
        try:
            await self._start_backend()
            await self.registry.register_server(
                self.url, self.config.port, self._metadata()
            )
            self.registered = True
            self.should_run = True
            self.heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        except Exception:
            await self._stop_backend()
            raise

    async def _heartbeat_loop(self) -> None:
        while self.should_run:
            try:
                if self.cache is None or not await self.cache.ping():
                    raise RuntimeError("cache backend ping failed")
                await self.registry.heartbeat(self.url, self.config.port)
            except Exception as exc:
                # Do not refresh registration while the backend is unhealthy.
                logger.warning("Cache service heartbeat failed: %s", exc)
            await asyncio.sleep(self.config.heartbeat_interval)

    async def cleanup_async(self) -> None:
        self.should_run = False
        if self.heartbeat_task:
            self.heartbeat_task.cancel()
            try:
                await self.heartbeat_task
            except asyncio.CancelledError:
                pass
            self.heartbeat_task = None
        if self.registered:
            await self.registry.deregister()
            self.registered = False
        await self._stop_backend()

    def _install_routes(self) -> None:
        @self.app.post("/cache", response_model=CacheResponse)
        async def cache(request: CacheRequest) -> CacheResponse:
            return await self.execute(request)

        @self.app.get("/health")
        async def health() -> dict[str, Any]:
            try:
                if self.cache is None or not await self.cache.ping():
                    raise RuntimeError("backend ping failed")
            except Exception as exc:
                raise HTTPException(
                    status_code=503, detail="cache backend is unavailable"
                ) from exc
            return {
                "status": "healthy",
                "service": self.config.service_name,
                "storage": "redis",
            }

        @self.app.get("/")
        async def root() -> dict[str, Any]:
            return {
                "message": "POST /cache with operation=get, set, or delete",
                "service": self.config.service_name,
                "managed_backend": self.config.backend_redis is None,
                "default_ttl": self.config.default_ttl,
                "max_value_bytes": self.config.max_value_bytes,
            }

        @self.app.on_event("startup")
        async def startup() -> None:
            await self.start()

        @self.app.on_event("shutdown")
        async def shutdown() -> None:
            await self.cleanup_async()


def main(
    host: str = "0.0.0.0",
    port: int = 1215,
    registry: str = "redis://klone-login01.hyak.local:6379",
    heartbeat_interval: float = 30,
    service_name: str = "cache",
    backend_redis: str | None = None,
    backend_port: int = 6379,
    redis_server_path: str | None = None,
    maxmemory: str = "4gb",
    maxmemory_policy: str = "allkeys-lfu",
    default_ttl: int = 3600,
    startup_timeout: float = 15,
    max_value_bytes: int = 8 * 1024 * 1024,
) -> None:
    """Run the registered HTTP cache service with Uvicorn."""
    import uvicorn

    config = CacheServerConfig(
        host=host,
        port=port,
        registry=registry,
        heartbeat_interval=heartbeat_interval,
        service_name=service_name,
        backend_redis=backend_redis,
        backend_port=backend_port,
        redis_server_path=redis_server_path,
        maxmemory=maxmemory,
        maxmemory_policy=maxmemory_policy,
        default_ttl=default_ttl,
        startup_timeout=startup_timeout,
        max_value_bytes=max_value_bytes,
    )
    uvicorn.run(CacheServer(config).app, host=host, port=port)


if __name__ == "__main__":
    fire.Fire(main)
