from .registry import ServerRegistry
from .client import RegistryClient
from .kvstore import (
    FILE_REGISTRY_SCHEME,
    FileSystemKVStore,
    filesystem_registry_path,
    filesystem_registry_uri,
)
from .redis import RedisKVStore, start_redis_server
from .sqlite import SQLITE_REGISTRY_SCHEME, SQLiteKVStore, sqlite_registry_path
from .head_registry import (
    HEAD_REGISTRY_SCHEME,
    HeadRegistryClosedError,
    HeadRegistryKVStore,
    head_registry_backend,
    head_registry_path,
    head_registry_uri,
    is_head_registry_uri,
)
from .http import RegistryHTTPClient
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
    "FILE_REGISTRY_SCHEME",
    "filesystem_registry_path",
    "filesystem_registry_uri",
    "RedisKVStore",
    "SQLiteKVStore",
    "SQLITE_REGISTRY_SCHEME",
    "sqlite_registry_path",
    "HeadRegistryKVStore",
    "HeadRegistryClosedError",
    "HEAD_REGISTRY_SCHEME",
    "head_registry_backend",
    "head_registry_path",
    "head_registry_uri",
    "is_head_registry_uri",
    "get_kvstore",
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

def get_kvstore(registry=None, *, head_registry=None, raise_on_error=False):
    """Build a KV store from a Redis URL, SQLite URI, path, or head registry."""
    if head_registry is not None:
        if registry is not None:
            raise ValueError("supply only one of registry or head_registry")
        return HeadRegistryKVStore(head_registry)
    if registry is None:
        raise ValueError("registry or head_registry is required")
    value = str(registry)
    if is_head_registry_uri(value):
        return HeadRegistryKVStore(head_registry_backend(value))
    if value.startswith(("redis://", "rediss://")):
        return RedisKVStore(value, raise_on_error=raise_on_error)
    if value.startswith(SQLITE_REGISTRY_SCHEME):
        return SQLiteKVStore(value)
    if value.startswith(FILE_REGISTRY_SCHEME):
        return FileSystemKVStore(value)
    return FileSystemKVStore(value)


# ServiceAPI imports get_kvstore from this module, so import it only after the
# factory has been defined.
from .api import ServiceAPI
