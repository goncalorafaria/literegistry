import abc
import asyncio
import hashlib
import math
from pathlib import Path
import time
from typing import Optional, Union, List


class KeyValueStore(abc.ABC):
    """Abstract base class for key-value storage"""

    @abc.abstractmethod
    async def get(self, key: str) -> Optional[bytes]:
        """Get value for a key"""
        pass

    @abc.abstractmethod
    async def set(
        self,
        key: str,
        value: Union[bytes, str],
        ttl_seconds: Optional[float] = None,
    ) -> bool:
        """Set value for a key, optionally expiring it after ``ttl_seconds``."""
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
    async def keys(self, prefix: Optional[str] = None) -> List[str]:
        """Get keys in the store, optionally restricted to a prefix."""
        pass


class FileSystemKVStore(KeyValueStore):
    """Filesystem-based key-value store (keys = files, values = content)"""

    def __init__(self, root: Union[str, Path] = "/gscratch/ark/graf/registry"):
        self.root = Path(root)
        self.root.mkdir(exist_ok=True)
        self._ttl_root = self.root / ".literegistry_ttl"

    @staticmethod
    def _validate_ttl(ttl_seconds: Optional[float]) -> Optional[float]:
        if ttl_seconds is None:
            return None
        ttl = float(ttl_seconds)
        if not math.isfinite(ttl) or ttl <= 0:
            raise ValueError("ttl_seconds must be a finite value greater than zero")
        return ttl

    def _ttl_path(self, key: str) -> Path:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self._ttl_root / digest

    def _delete_sync(self, key: str) -> bool:
        deleted = False
        for path in (self.root / key, self._ttl_path(key)):
            try:
                path.unlink()
                deleted = True
            except FileNotFoundError:
                pass
        return deleted

    def _read_sync(self, key: str) -> Optional[bytes]:
        key_path = self.root / key
        ttl_path = self._ttl_path(key)
        try:
            expires_at = float(ttl_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            expires_at = None
        except (OSError, ValueError):
            # Bad TTL metadata must not hide an otherwise valid registry value.
            try:
                ttl_path.unlink()
            except FileNotFoundError:
                pass
            expires_at = None

        if expires_at is not None and time.time() >= expires_at:
            self._delete_sync(key)
            return None

        try:
            return key_path.read_bytes()
        except FileNotFoundError:
            try:
                ttl_path.unlink()
            except FileNotFoundError:
                pass
            return None

    async def get(self, key: str) -> Optional[bytes]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._read_sync, key)

    async def set(
        self,
        key: str,
        value: Union[bytes, str],
        ttl_seconds: Optional[float] = None,
    ) -> bool:
        ttl = self._validate_ttl(ttl_seconds)
        key_path = self.root / key
        if isinstance(value, str):
            value = value.encode("utf-8")

        def _set_sync() -> None:
            key_path.write_bytes(value)
            ttl_path = self._ttl_path(key)
            if ttl is None:
                try:
                    ttl_path.unlink()
                except FileNotFoundError:
                    pass
                return
            self._ttl_root.mkdir(exist_ok=True)
            ttl_path.write_text(str(time.time() + ttl), encoding="utf-8")

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, _set_sync)
        return True

    async def delete(self, key: str) -> bool:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._delete_sync, key)

    async def exists(self, key: str) -> bool:
        return await self.get(key) is not None

    async def keys(self, prefix: Optional[str] = None) -> List[str]:
        """Get a list of all keys (filenames) in the store"""

        def _get_keys():
            result = []
            for path in self.root.glob("*"):
                if not path.is_file():
                    continue
                key = path.name
                if prefix is not None and not key.startswith(prefix):
                    continue
                if self._read_sync(key) is not None:
                    result.append(key)
            return result

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _get_keys)

    async def close(self):
        """No-op for compatibility; nothing to close for filesystem store."""
        pass

