import abc
import asyncio
from pathlib import Path
from typing import Optional, Union, List
import functools


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
    

