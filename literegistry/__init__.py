from .registry import ServerRegistry
from .client import RegistryClient
from .kvstore import FileSystemKVStore
from .redis import RedisKVStore, start_redis_server
from .http import RegistryHTTPClient
from .api import ServiceAPI
from .affinity import (
    AffinityBinding,
    AffinityBindingConflict,
    AffinityBindingError,
    AffinityBindingStore,
    AffinityBindingTypeMismatch,
    InvalidAffinityBinding,
    SoftAffinityBinding,
    SoftAffinityBindingStore,
    StrictAffinityBinding,
    StrictAffinityBindingStore,
)

__all__ = [
    "RegistryClient",
    "ServerRegistry",
    "FileSystemKVStore",
    "RedisKVStore",
    "RegistryHTTPClient",
    "ServiceAPI",
    "start_redis_server",
    "AffinityBinding",
    "AffinityBindingConflict",
    "AffinityBindingError",
    "AffinityBindingStore",
    "AffinityBindingTypeMismatch",
    "InvalidAffinityBinding",
    "SoftAffinityBinding",
    "SoftAffinityBindingStore",
    "StrictAffinityBinding",
    "StrictAffinityBindingStore",
]

def get_kvstore(registry):
    if "redis://" in registry:
        return RedisKVStore(registry)
    else:
        return FileSystemKVStore(registry)