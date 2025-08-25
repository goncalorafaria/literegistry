import abc
import asyncio
from pathlib import Path
from typing import Optional, Union, List
import functools
import redis.asyncio as redis


class KeyValueStore(abc.ABC):
    """Abstract base class for key-value storage"""

    @abc.abstractmethod
    async def get(self, key: str) -> Optional[bytes]:
        """Get value for a key"""
        pass

    @abc.abstractmethod
    async def set(self, key: str, value: Union[bytes, str]) -> bool:
        """Set value for a key"""
        pass

    @abc.abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete a key"""
        pass

    @abc.abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists"""
        pass

    @abc.abstractmethod
    async def keys(self) -> List[str]:
        """Get a list of all keys in the store"""
        pass


class RedisKVStore(KeyValueStore):
    """Redis-based key-value store"""
    #  http://klone-login01.hyak.local:8080/v1/models
    def __init__(self, url: str = "redis://klone-login01.hyak.local:6379", db: int = 0):
        """
        Initialize Redis KV store
        
        Args:
            url: Redis connection URL (e.g., "redis://localhost:6379", "redis://user:pass@host:port")
            db: Redis database number
        """
        self.url = url
        self.db = db
        self._redis = None

    async def _get_redis(self) -> redis.Redis:
        """Get Redis connection, creating it if necessary"""
        if self._redis is None:
            try:
                self._redis = redis.from_url(self.url, db=self.db, decode_responses=False)
                # Test the connection
                await self._redis.ping()
                print(f"Successfully connected to Redis at {self.url}")
            except Exception as e:
                print(f"Failed to connect to Redis at {self.url}: {e}")
                raise
        return self._redis

    async def get(self, key: str) -> Optional[bytes]:
        """Get value for a key from Redis"""
        redis_client = await self._get_redis()
        try:
            value = await redis_client.get(key)
            return value
        except Exception:
            return None

    async def set(self, key: str, value: Union[bytes, str]) -> bool:
        """Set value for a key in Redis"""
        redis_client = await self._get_redis()
        try:
            if isinstance(value, str):
                value = value.encode("utf-8")
            await redis_client.set(key, value)
            return True
        except Exception:
            return False

    async def delete(self, key: str) -> bool:
        """Delete a key from Redis"""
        redis_client = await self._get_redis()
        try:
            result = await redis_client.delete(key)
            return result > 0
        except Exception:
            return False

    async def exists(self, key: str) -> bool:
        """Check if key exists in Redis"""
        redis_client = await self._get_redis()
        try:
            result = await redis_client.exists(key)
            return result > 0
        except Exception:
            return False

    async def keys(self) -> List[str]:
        """Get a list of all keys in Redis"""
        redis_client = await self._get_redis()
        try:
            keys = await redis_client.keys("*")
            return [key.decode('utf-8') if isinstance(key, bytes) else key for key in keys]
        except Exception:
            return []

    async def close(self):
        """Close Redis connection"""
        if self._redis:
            await self._redis.aclose()
            self._redis = None


class FileSystemKVStore(KeyValueStore):
    """Filesystem-based key-value store (keys = files, values = content)"""

    def __init__(self, root: Union[str, Path] = "/gscratch/ark/graf/registry"):
        self.root = Path(root)
        self.root.mkdir(exist_ok=True)

    async def get(self, key: str) -> Optional[bytes]:
        key_path = self.root / key
        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, key_path.read_bytes)
        except FileNotFoundError:
            return None

    async def set(self, key: str, value: Union[bytes, str]) -> bool:
        key_path = self.root / key
        if isinstance(value, str):
            value = value.encode("utf-8")
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, functools.partial(key_path.write_bytes, value))
        return True

    async def delete(self, key: str) -> bool:
        key_path = self.root / key
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, key_path.unlink)
            return True
        except FileNotFoundError:
            return False

    async def exists(self, key: str) -> bool:
        key_path = self.root / key
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, key_path.exists)

    async def keys(self) -> List[str]:
        """Get a list of all keys (filenames) in the store"""

        def _get_keys():
            return [p.name for p in self.root.glob("*") if p.is_file()]

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _get_keys)


# Usage Example
async def main():
    # FileSystem Example
    #fs_store = FileSystemKVStore()
    #await fs_store.set("test1.txt", "Hello FS!")
    #await fs_store.set("test2.txt", "World FS!")
    #print(await fs_store.keys())  # ['test1.txt', 'test2.txt']

    # Redis Example
    try:
        redis_store = RedisKVStore("redis://klone-login01.hyak.local:6379")
        print("Setting test values...")
        await redis_store.set("test1", "Hello Redis!")
        await redis_store.set("test2", "World Redis!")
        print("Getting keys...")
        keys = await redis_store.keys()
        print(f"Keys found: {keys}")
        await redis_store.close()
    except Exception as e:
        print(f"Redis test failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())
