from .registry import ServerRegistry
from .client import RegistryClient
from .kvstore import FileSystemKVStore
from .redis import RedisKVStore, start_redis_server
from .http import RegistryHTTPClient
from .api import ServiceAPI

__all__ = [
    "RegistryClient",
    "ServerRegistry",
    "FileSystemKVStore",
    "RedisKVStore",
    "RegistryHTTPClient",
    "ServiceAPI",
    "start_redis_server",
]

def get_kvstore(registry):
    if "redis://" in registry:
        return RedisKVStore(registry)
    else:
        return FileSystemKVStore(registry)