import json
import socket
import time
from collections import deque
from typing import Optional, Dict, List, Any
import asyncio
from literegistry.kvstore import KeyValueStore, FileSystemKVStore


class ServerRegistry:
    """Server registry implementation using KeyValueStore abstraction"""

    def __init__(self, store: KeyValueStore, max_history: int = 3600, max_heartbeat_interval: int = 200):
        """
        Initialize ServerRegistry with a key-value store backend

        Args:
            store: KeyValueStore implementation to use for storage
            max_history: Maximum number of requests to keep in history
        """
        self.store = store
        self.server_id = f"{socket.gethostname()}-{time.time()}"[:20]
        self.request_timestamps = deque(maxlen=max_history)
        self._metadata = {}  # Store metadata for re-registration if needed
        self.max_heartbeat_interval = max_heartbeat_interval

    async def roster(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get status of all servers in the cluster"""
        servers = []
        now = time.time()

        # Get all server keys
        server_keys = await self.store.keys()

        for key in server_keys:
            try:
                info_bytes = await self.store.get(key)
                if info_bytes:
                    info = json.loads(info_bytes.decode("utf-8"))
                    # Mark servers that haven't sent heartbeat in 30 seconds as inactive
                    if now - info["last_heartbeat"] > self.max_heartbeat_interval:
                        info["status"] = "inactive"
                        # Optionally delete inactive servers
                        # await self.store.delete(key)
                    else:
                        servers.append(info)
            except Exception as e:
                continue

        return {"servers": servers}

    async def register_server(
        self, url: str, port: int, metadata: Optional[Dict] = None,
    ) -> str:
        """Register a new server with the registry"""
        self._metadata = metadata or {}  # Store for potential re-registration
        #self._url = f"http://{socket.gethostname()}.hyak.local"  # Store URL for re-registration

        info = {
            "server_id": self.server_id,
            "host": socket.gethostname(),
            "port": port,
            "last_heartbeat": time.time(),
            "metadata": self._metadata,
            "status": "active",
            "uri": f"{url}:{port}",
        }

        key = f"server_{self.server_id}"
        await self.store.set(key, json.dumps(info))
        return self.server_id

    async def heartbeat(self, url: str, port: int, data: Optional[Dict] = None):
        """Update server heartbeat and stats"""
        key = f"server_{self.server_id}"

        try:
            info_bytes = await self.store.get(key)
            if info_bytes:
                info = json.loads(info_bytes.decode("utf-8"))
                if data:
                    info["data"] = data
                info["last_heartbeat"] = time.time()
                await self.store.set(key, json.dumps(info))
            else:
                # Re-register if key disappeared
                await self.register_server(url, port, self._metadata)
        except Exception:
            # Re-register on any error
            await self.register_server(url, port, self._metadata)

    async def deregister(self):
        """Remove server from registry"""
        key = f"server_{self.server_id}"
        await self.store.delete(key)


# Example usage
async def main():
    # Using FileSystem backend
    store = FileSystemKVStore("registry_data")
    registry = ServerRegistry(store)

    # Register a server
    server_id = await registry.register_server(8000, {"model": "gpt-3"})
    print(f"Registered server: {server_id}")

    # Update heartbeat
    await registry.heartbeat(8000)

    # Get roster
    roster = await registry.roster()
    print("Active servers:", roster)

    # Cleanup
    await registry.deregister()


if __name__ == "__main__":
    asyncio.run(main())
