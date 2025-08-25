from literegistry import ServerRegistry, FileSystemKVStore, RedisKVStore
import asyncio
from fastapi import FastAPI, HTTPException
from typing import List, Optional, Dict, Any
import time
from threading import Thread
import socket


class ServiceAPI(FastAPI):
    """
    FastAPI extension that automatically handles server registration, heartbeat, and deregistration.
    """

    def __init__(
        self,
        *args,
        registry_path: str = "redis://klone-login01.hyak.local:6379",# "/gscratch/ark/graf/registry", # "redis://klone-login01.hyak.local:6379"
        port: int = None,
        hostname: str = None,
        metadata: Dict[str, Any] = None,
        heartbeat_interval: int = 120,
        max_history=3600,
        **kwargs,
    ):
        """
        Initialize RewardModelServer with automatic registration and heartbeat.

        Args:
            *args: Arguments to pass to FastAPI constructor
            registry_path: Path to the registry filesystem
            port: Port number for the server
            metadata: Server metadata for registration
            heartbeat_interval: Interval in seconds for heartbeat
            **kwargs: Keyword arguments to pass to FastAPI constructor
        """
        super().__init__(*args, **kwargs)

        if "redis://" in registry_path:
            store = RedisKVStore(registry_path)
        else:
            store = FileSystemKVStore(registry_path)
            
            
        self.registry_path = registry_path
        self.port = port
        self.hostname = hostname
        self.metadata = metadata or {}
        self.heartbeat_interval = heartbeat_interval
        self.registry = ServerRegistry(
            store=store,#RedisKVStore("redis://klone-login01.hyak.local:6379"),#FileSystemKVStore(self.registry_path),
            max_history=max_history,
        )
        self.heartbeat_thread = None
        self.url = f"http://{hostname}"

        # Register startup and shutdown events
        self._register_startup_events()
        self._register_shutdown_events()

    def _register_startup_events(self):
        """Register startup event handlers."""

        @self.on_event("startup")
        async def startup_event():

            # Register server
            await self.registry.register_server(
                url= self.url,
                port=self.port,
                metadata=self.metadata,
            )

            # Start heartbeat thread
            self._start_heartbeat_thread()

    def _register_shutdown_events(self):
        """Register shutdown event handlers."""

        @self.on_event("shutdown")
        async def shutdown_event():
            if self.registry:
                await self.registry.deregister()

    def _start_heartbeat_thread(self):
        """Start a daemon thread for heartbeat operations."""

        def heartbeat_loop():
            while True:
                asyncio.run(self.registry.heartbeat(self.url, self.port))
                time.sleep(self.heartbeat_interval)

        self.heartbeat_thread = Thread(target=heartbeat_loop, daemon=True)
        self.heartbeat_thread.start()
