from aioconsul import Consul
import abc
import asyncio
from pathlib import Path
from typing import Optional, Union, List
from literegistry.kvstore import KeyValueStore


class ConsulKVStore(KeyValueStore):
    """Consul-based key-value store using aioconsul"""

    def __init__(self, consul: Consul = None, prefix: str = "kv/"):
        self.consul = consul or Consul()
        self.prefix = prefix

    def _prefixed_key(self, key: str) -> str:
        return f"{self.prefix}{key}"

    def _unprefix_key(self, prefixed_key: str) -> str:
        """Remove prefix from a key"""
        if prefixed_key.startswith(self.prefix):
            return prefixed_key[len(self.prefix) :]
        return prefixed_key

    async def get(self, key: str) -> Optional[bytes]:
        prefixed_key = self._prefixed_key(key)
        result = await self.consul.kv.get(prefixed_key)
        if result and result["Value"]:
            return result["Value"]
        return None

    async def set(self, key: str, value: Union[bytes, str]) -> bool:
        if isinstance(value, str):
            value = value.encode("utf-8")
        prefixed_key = self._prefixed_key(key)
        return await self.consul.kv.set(prefixed_key, value)

    async def delete(self, key: str) -> bool:
        prefixed_key = self._prefixed_key(key)
        return await self.consul.kv.delete(prefixed_key)

    async def exists(self, key: str) -> bool:
        prefixed_key = self._prefixed_key(key)
        result = await self.consul.kv.get(prefixed_key)
        return result is not None

    async def keys(self) -> List[str]:
        """Get a list of all keys in the store under the current prefix"""
        keys = await self.consul.kv.keys(prefix=self.prefix)
        # Remove prefix from keys
        return [self._unprefix_key(key) for key in keys]


# Usage Example
async def main():

    # Consul Example (requires running Consul agent)
    async with Consul() as consul:
        consul_store = ConsulKVStore(consul)
        await consul_store.set("app/config1", b"{'timeout': 30}")
        await consul_store.set("app/config2", b"{'retry': 3}")
        print(await consul_store.keys())  # ['app/config1', 'app/config2']


if __name__ == "__main__":
    asyncio.run(main())
