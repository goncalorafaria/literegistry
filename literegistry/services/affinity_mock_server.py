"""Instance-local key/value service for exercising affinity routing.

Every server process owns a separate in-memory session store. A handshake ID
created by one instance is therefore rejected by every other instance, making
this service useful for proving that a gateway pins follow-up requests.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
import secrets
import socket
from typing import Any, Optional

import fire
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn

from literegistry.kvstore import FileSystemKVStore
from literegistry.redis import RedisKVStore
from literegistry.registry import ServerRegistry




def get_kvstore(registry: str):
    """Resolve a store without relying on package-level re-exports.

    Some editable/source environments expose ``literegistry`` as a namespace
    package. Direct submodule imports continue to work in those environments,
    while symbols re-exported from ``literegistry.__init__`` do not.
    """
    if registry.startswith("redis://"):
        return RedisKVStore(registry)
    return FileSystemKVStore(registry)


class AffinitySessionNotFound(KeyError):
    """The affinity ID belongs to no session on this replica."""


class AffinityKeyNotFound(KeyError):
    """The requested key is absent from an existing affinity session."""


class HandshakeRequest(BaseModel):
    client_id: Optional[str] = None


class PutRequest(BaseModel):
    affinity_id: str = Field(..., min_length=1)
    key: str = Field(..., min_length=1)
    value: Any


class GetRequest(BaseModel):
    affinity_id: str = Field(..., min_length=1)
    key: str = Field(..., min_length=1)


class AffinityKVService:
    """In-memory state owned by exactly one mock service replica."""

    def __init__(
        self,
        instance_id: Optional[str] = None,
        service_name: str = "affinity-kv",
    ) -> None:
        self.instance_id = instance_id or f"mock-{secrets.token_hex(8)}"
        self.service_name = service_name
        self._sessions: dict[str, dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    async def handshake(self, client_id: Optional[str] = None) -> dict[str, Any]:
        """Create an opaque ID whose session exists only on this instance."""
        affinity_id = secrets.token_urlsafe(32)
        async with self._lock:
            self._sessions[affinity_id] = {}
        print(f"[affinity-mock] HANDSHAKE instance={self.instance_id} client_id={client_id!r} affinity_id={affinity_id}", flush=True)
        return {
            "affinity_id": affinity_id,
            "instance_id": self.instance_id,
            "service": self.service_name,
            "client_id": client_id,
        }

    async def put(self, affinity_id: str, key: str, value: Any) -> dict[str, Any]:
        print(f"[affinity-mock] PUT input instance={self.instance_id} affinity_id={affinity_id} key={key!r} value={value!r}", flush=True)
        async with self._lock:
            session = self._sessions.get(affinity_id)
            if session is None:
                raise AffinitySessionNotFound(affinity_id)
            session[key] = value
            stored_count = len(session)
        print(f"[affinity-mock] PUT stored instance={self.instance_id} affinity_id={affinity_id} key={key!r} stored_count={stored_count}", flush=True)
        return {
            "affinity_id": affinity_id,
            "instance_id": self.instance_id,
            "key": key,
            "value": value,
            "stored_count": stored_count,
        }

    async def get(self, affinity_id: str, key: str) -> dict[str, Any]:
        print(f"[affinity-mock] GET input instance={self.instance_id} affinity_id={affinity_id} key={key!r}", flush=True)
        async with self._lock:
            session = self._sessions.get(affinity_id)
            if session is None:
                raise AffinitySessionNotFound(affinity_id)
            if key not in session:
                raise AffinityKeyNotFound(key)
            value = session[key]
        print(f"[affinity-mock] GET output instance={self.instance_id} affinity_id={affinity_id} key={key!r} value={value!r}", flush=True)
        return {
            "affinity_id": affinity_id,
            "instance_id": self.instance_id,
            "key": key,
            "value": value,
        }

    async def session_count(self) -> int:
        async with self._lock:
            return len(self._sessions)


def create_mock_app(
    service: Optional[AffinityKVService] = None,
    *,
    lifespan=None,
) -> FastAPI:
    """Create the HTTP API around an affinity mock service instance."""
    service = service or AffinityKVService()
    app = FastAPI(title="LiteRegistry Affinity Mock", lifespan=lifespan)
    app.state.affinity_service = service

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "healthy",
            "service": service.service_name,
            "instance_id": service.instance_id,
            "sessions": await service.session_count(),
        }

    @app.post("/handshake")
    async def handshake(
        request: Optional[HandshakeRequest] = None,
    ) -> dict[str, Any]:
        return await service.handshake(
            client_id=request.client_id if request is not None else None
        )

    @app.post("/kv/put")
    async def put(request: PutRequest) -> dict[str, Any]:
        try:
            return await service.put(
                request.affinity_id,
                request.key,
                request.value,
            )
        except AffinitySessionNotFound:
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "affinity_miss",
                    "message": "affinity ID does not belong to this instance",
                    "instance_id": service.instance_id,
                },
            )

    @app.post("/kv/get")
    async def get(request: GetRequest) -> dict[str, Any]:
        try:
            return await service.get(request.affinity_id, request.key)
        except AffinitySessionNotFound:
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "affinity_miss",
                    "message": "affinity ID does not belong to this instance",
                    "instance_id": service.instance_id,
                },
            )
        except AffinityKeyNotFound:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "key_not_found",
                    "key": request.key,
                    "instance_id": service.instance_id,
                },
            )

    return app


@dataclass(frozen=True)
class AffinityMockConfig:
    registry: str = "redis://klone-login01.hyak.local:6379"
    service_name: str = "affinity-kv"
    host: str = "0.0.0.0"
    advertise_host: Optional[str] = None
    port: int = 8090
    heartbeat_interval: float = 10.0
    instance_id: Optional[str] = None


class RegisteredAffinityMockServer:
    """Affinity mock API with LiteRegistry registration and heartbeats."""

    def __init__(self, config: AffinityMockConfig) -> None:
        self.config = config
        self.store = get_kvstore(config.registry)
        self.registry = ServerRegistry(store=self.store)
        self.service = AffinityKVService(
            instance_id=config.instance_id or self.registry.server_id,
            service_name=config.service_name,
        )
        advertised_host = config.advertise_host or (
            socket.getfqdn()
            if config.host in {"0.0.0.0", "::"}
            else config.host
        )
        self.url = f"http://{advertised_host}"
        self.app = create_mock_app(self.service, lifespan=self._lifespan)

    def metadata(self) -> dict[str, Any]:
        return {
            "model_path": self.config.service_name,
            "backend": "affinity-mock-kv",
            "instance_id": self.service.instance_id,
            "affinity": {
                "enabled": True,
                "handshake_endpoint": "handshake",
                "id_field": "affinity_id",
            },
        }

    async def _heartbeat(self) -> None:
        while True:
            await asyncio.sleep(self.config.heartbeat_interval)
            await self.registry.heartbeat(self.url, self.config.port)

    @asynccontextmanager
    async def _lifespan(self, app: FastAPI):
        print(f"[affinity-mock] REGISTERING instance={self.service.instance_id} service={self.config.service_name} registry={self.config.registry}", flush=True)
        await self.registry.register_server(
            self.url,
            self.config.port,
            self.metadata(),
        )
        print(f"[affinity-mock] REGISTERED instance={self.service.instance_id} uri={self.url}:{self.config.port}", flush=True)
        heartbeat = asyncio.create_task(self._heartbeat())
        try:
            yield
        finally:
            heartbeat.cancel()
            await asyncio.gather(heartbeat, return_exceptions=True)
            await self.registry.deregister()
            await self.store.close()


def main(
    registry: str = "redis://klone-login01.hyak.local:6379",
    service_name: str = "affinity-kv",
    host: str = "0.0.0.0",
    advertise_host: Optional[str] = None,
    port: int = 8090,
    heartbeat_interval: float = 10.0,
    instance_id: Optional[str] = None,
) -> None:
    """Run one mock replica; start another process to test pinning."""
    server = RegisteredAffinityMockServer(
        AffinityMockConfig(
            registry=registry,
            service_name=service_name,
            host=host,
            advertise_host=advertise_host,
            port=port,
            heartbeat_interval=heartbeat_interval,
            instance_id=instance_id,
        )
    )
    uvicorn.run(server.app, host=host, port=port, workers=1)


if __name__ == "__main__":
    fire.Fire(main)

