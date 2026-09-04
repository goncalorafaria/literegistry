"""Persistent affinity bindings for stateful LiteRegistry services."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
import time
from typing import Callable, Generic, Optional, TypeVar, cast
from urllib.parse import quote

from literegistry.kvstore import KeyValueStore


AFFINITY_KEY_PREFIX = "affinity:"
AFFINITY_BINDING_VERSION = 1
STRICT_AFFINITY_TYPE = "strict"
SOFT_AFFINITY_TYPE = "soft"


class AffinityBindingError(RuntimeError):
    """Base error raised by affinity binding persistence."""


class InvalidAffinityBinding(AffinityBindingError):
    """The stored affinity record is malformed or does not match its key."""


class AffinityBindingTypeMismatch(AffinityBindingError):
    """A binding was accessed through the wrong affinity store type."""


class AffinityBindingConflict(AffinityBindingError):
    """An existing binding cannot be reassigned through bind()."""


@dataclass(frozen=True)
class AffinityBinding(ABC):
    """Shared persisted fields for strict and soft affinity bindings."""

    service: str
    affinity_id_hash: str
    server_id: str
    server_uri: str
    created_at: float
    last_used_at: float
    expires_at: float
    version: int = AFFINITY_BINDING_VERSION

    @classmethod
    @abstractmethod
    def type_name(cls) -> str:
        """Return the stable discriminator persisted with this binding."""

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["affinity_type"] = self.type_name()
        return value

    @staticmethod
    def _text_field(value: dict[str, object], name: str) -> str:
        item = value.get(name)
        if not isinstance(item, str) or not item.strip():
            raise InvalidAffinityBinding(
                f"affinity binding field {name!r} must be a non-empty string"
            )
        return item

    @staticmethod
    def _timestamp_field(value: dict[str, object], name: str) -> float:
        item = value.get(name)
        if isinstance(item, bool):
            raise InvalidAffinityBinding(
                f"affinity binding field {name!r} must be a timestamp"
            )
        try:
            timestamp = float(item)
        except (TypeError, ValueError) as exc:
            raise InvalidAffinityBinding(
                f"affinity binding field {name!r} must be a timestamp"
            ) from exc
        if not math.isfinite(timestamp):
            raise InvalidAffinityBinding(
                f"affinity binding field {name!r} must be finite"
            )
        return timestamp

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "AffinityBinding":
        """Deserialize either concrete binding type.

        Records created before binding types were introduced are interpreted as
        strict bindings, preserving the original no-handoff behavior.
        """
        if not isinstance(value, dict):
            raise InvalidAffinityBinding("affinity binding record is not an object")

        affinity_type = value.get("affinity_type", STRICT_AFFINITY_TYPE)
        binding_classes = {
            STRICT_AFFINITY_TYPE: StrictAffinityBinding,
            SOFT_AFFINITY_TYPE: SoftAffinityBinding,
        }
        binding_class = binding_classes.get(affinity_type)
        if binding_class is None:
            raise InvalidAffinityBinding(
                f"unsupported affinity binding type: {affinity_type!r}"
            )

        try:
            version = int(
                value.get("version", AFFINITY_BINDING_VERSION)
            )
        except (TypeError, ValueError) as exc:
            raise InvalidAffinityBinding(
                "affinity binding version must be an integer"
            ) from exc
        if version != AFFINITY_BINDING_VERSION:
            raise InvalidAffinityBinding(
                f"unsupported affinity binding version: {version}"
            )

        common = {
            "service": cls._text_field(value, "service"),
            "affinity_id_hash": cls._text_field(
                value,
                "affinity_id_hash",
            ),
            "server_id": cls._text_field(value, "server_id"),
            "server_uri": cls._text_field(value, "server_uri"),
            "created_at": cls._timestamp_field(value, "created_at"),
            "last_used_at": cls._timestamp_field(
                value,
                "last_used_at",
            ),
            "expires_at": cls._timestamp_field(value, "expires_at"),
            "version": version,
        }

        digest = common["affinity_id_hash"]
        if len(digest) != 64:
            raise InvalidAffinityBinding(
                "affinity_id_hash must be a SHA-256 hexadecimal digest"
            )
        try:
            int(digest, 16)
        except ValueError as exc:
            raise InvalidAffinityBinding(
                "affinity_id_hash must be a SHA-256 hexadecimal digest"
            ) from exc

        if not (
            common["created_at"]
            <= common["last_used_at"]
            < common["expires_at"]
        ):
            raise InvalidAffinityBinding(
                "affinity binding timestamps are out of order"
            )

        if binding_class is StrictAffinityBinding:
            return StrictAffinityBinding(**common)

        handoff_count = value.get("handoff_count", 0)
        if (
            isinstance(handoff_count, bool)
            or not isinstance(handoff_count, int)
            or handoff_count < 0
        ):
            raise InvalidAffinityBinding(
                "soft affinity handoff_count must be a non-negative integer"
            )
        previous_server_id = value.get("previous_server_id")
        last_handoff_at = value.get("last_handoff_at")
        if handoff_count == 0:
            if previous_server_id is not None or last_handoff_at is not None:
                raise InvalidAffinityBinding(
                    "a new soft binding cannot contain handoff metadata"
                )
        else:
            if (
                not isinstance(previous_server_id, str)
                or not previous_server_id.strip()
            ):
                raise InvalidAffinityBinding(
                    "a handed-off soft binding requires previous_server_id"
                )
            last_handoff_at = cls._timestamp_field(
                value,
                "last_handoff_at",
            )
            if not (
                common["created_at"]
                <= last_handoff_at
                <= common["last_used_at"]
            ):
                raise InvalidAffinityBinding(
                    "last_handoff_at is outside the binding lifetime"
                )

        return SoftAffinityBinding(
            **common,
            handoff_count=handoff_count,
            previous_server_id=previous_server_id,
            last_handoff_at=last_handoff_at,
        )


class StrictAffinityBinding(AffinityBinding):
    """A binding whose owning server may never change."""

    @classmethod
    def type_name(cls) -> str:
        return STRICT_AFFINITY_TYPE


@dataclass(frozen=True)
class SoftAffinityBinding(AffinityBinding):
    """A binding that may be explicitly handed to another server."""

    handoff_count: int = 0
    previous_server_id: Optional[str] = None
    last_handoff_at: Optional[float] = None

    @classmethod
    def type_name(cls) -> str:
        return SOFT_AFFINITY_TYPE


BindingT = TypeVar("BindingT", bound=AffinityBinding)


class AffinityBindingStore(ABC, Generic[BindingT]):
    """Shared storage API for concrete strict and soft affinity stores."""

    def __init__(
        self,
        store: KeyValueStore,
        default_ttl_seconds: float = 900,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.store = store
        self.default_ttl_seconds = self._validate_ttl(default_ttl_seconds)
        self._clock = clock

    @classmethod
    @abstractmethod
    def binding_class(cls) -> type[BindingT]:
        """Return the concrete binding class managed by this store."""

    @staticmethod
    def _validate_text(value: str, name: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a non-empty string")
        return value

    @staticmethod
    def _validate_ttl(ttl_seconds: float) -> float:
        ttl = float(ttl_seconds)
        if not math.isfinite(ttl) or ttl <= 0:
            raise ValueError(
                "ttl_seconds must be a finite value greater than zero"
            )
        return ttl

    def _ttl(self, ttl_seconds: Optional[float]) -> float:
        return self._validate_ttl(
            self.default_ttl_seconds
            if ttl_seconds is None
            else ttl_seconds
        )

    def _now(self) -> float:
        now = float(self._clock())
        if not math.isfinite(now):
            raise ValueError("clock must return a finite timestamp")
        return now

    @staticmethod
    def hash_affinity_id(affinity_id: str) -> str:
        AffinityBindingStore._validate_text(affinity_id, "affinity_id")
        return hashlib.sha256(affinity_id.encode("utf-8")).hexdigest()

    @staticmethod
    def service_prefix(service: str) -> str:
        AffinityBindingStore._validate_text(service, "service")
        return f"{AFFINITY_KEY_PREFIX}{quote(service, safe='')}:"

    @classmethod
    def binding_key(cls, service: str, affinity_id: str) -> str:
        return (
            f"{cls.service_prefix(service)}"
            f"{cls.hash_affinity_id(affinity_id)}"
        )

    async def _set_binding(
        self,
        key: str,
        binding: BindingT,
        ttl_seconds: float,
    ) -> None:
        payload = json.dumps(
            binding.to_dict(),
            separators=(",", ":"),
            sort_keys=True,
        )
        try:
            written = await self.store.set(
                key,
                payload,
                ttl_seconds=ttl_seconds,
            )
        except (TypeError, NotImplementedError):
            # Older/custom stores still get application-level expiration
            # from expires_at.
            written = await self.store.set(key, payload)
        if not written:
            raise AffinityBindingError(
                f"failed to persist affinity binding {key}"
            )

    def _ensure_type(self, binding: AffinityBinding) -> BindingT:
        expected_class = self.binding_class()
        if not isinstance(binding, expected_class):
            raise AffinityBindingTypeMismatch(
                f"{type(self).__name__} cannot manage "
                f"{binding.type_name()!r} affinity bindings"
            )
        return cast(BindingT, binding)

    async def _load_key(self, key: str) -> Optional[AffinityBinding]:
        payload = await self.store.get(key)
        if payload is None:
            return None
        return await self._load_payload(key, payload)

    async def _load_payload(
        self,
        key: str,
        payload: bytes | str,
    ) -> Optional[AffinityBinding]:
        try:
            decoded = (
                payload.decode("utf-8")
                if isinstance(payload, bytes)
                else payload
            )
            value = json.loads(decoded)
        except (UnicodeDecodeError, json.JSONDecodeError, TypeError) as exc:
            raise InvalidAffinityBinding(
                f"invalid affinity binding stored at {key}"
            ) from exc

        binding = AffinityBinding.from_dict(value)
        expected_key = (
            f"{self.service_prefix(binding.service)}"
            f"{binding.affinity_id_hash}"
        )
        if key != expected_key:
            raise InvalidAffinityBinding(
                f"affinity binding does not match key {key}"
            )
        if binding.expires_at <= self._now():
            await self.store.delete(key)
            return None
        return binding

    def _new_binding(
        self,
        service: str,
        affinity_id: str,
        server_id: str,
        server_uri: str,
        now: float,
        ttl: float,
    ) -> BindingT:
        return self.binding_class()(
            service=service,
            affinity_id_hash=self.hash_affinity_id(affinity_id),
            server_id=server_id,
            server_uri=server_uri,
            created_at=now,
            last_used_at=now,
            expires_at=now + ttl,
        )

    async def bind(
        self,
        service: str,
        affinity_id: str,
        server_id: str,
        server_uri: str,
        ttl_seconds: Optional[float] = None,
    ) -> BindingT:
        """Create a binding without changing an existing owner."""
        self._validate_text(service, "service")
        self._validate_text(affinity_id, "affinity_id")
        self._validate_text(server_id, "server_id")
        self._validate_text(server_uri, "server_uri")
        ttl = self._ttl(ttl_seconds)
        key = self.binding_key(service, affinity_id)
        existing = await self._load_key(key)
        if existing is not None:
            existing = self._ensure_type(existing)
            if (
                existing.server_id != server_id
                or existing.server_uri != server_uri
            ):
                raise AffinityBindingConflict(
                    "affinity ID is already bound to another server"
                )
            return await self._refresh(key, existing, ttl)

        now = self._now()
        binding = self._new_binding(
            service,
            affinity_id,
            server_id,
            server_uri,
            now,
            ttl,
        )
        await self._set_binding(key, binding, ttl)
        return binding

    async def resolve(
        self,
        service: str,
        affinity_id: str,
    ) -> Optional[BindingT]:
        """Resolve a binding, returning None when absent or expired."""
        binding = await self._load_key(
            self.binding_key(service, affinity_id)
        )
        if binding is None:
            return None
        return self._ensure_type(binding)

    async def _refresh(
        self,
        key: str,
        binding: BindingT,
        ttl: float,
    ) -> BindingT:
        now = self._now()
        refreshed = replace(
            binding,
            last_used_at=now,
            expires_at=now + ttl,
        )
        await self._set_binding(key, refreshed, ttl)
        return cast(BindingT, refreshed)

    async def refresh_binding(
        self,
        binding: BindingT,
        ttl_seconds: Optional[float] = None,
    ) -> BindingT:
        """Refresh an already-resolved binding without reading it again."""
        binding = self._ensure_type(binding)
        key = f"{self.service_prefix(binding.service)}{binding.affinity_id_hash}"
        return await self._refresh(key, binding, self._ttl(ttl_seconds))

    async def touch(
        self,
        service: str,
        affinity_id: str,
        ttl_seconds: Optional[float] = None,
    ) -> Optional[BindingT]:
        """Refresh a binding's sliding expiration window."""
        binding = await self.resolve(service, affinity_id)
        if binding is None:
            return None
        return await self._refresh(
            self.binding_key(service, affinity_id),
            binding,
            self._ttl(ttl_seconds),
        )

    async def release(self, service: str, affinity_id: str) -> bool:
        """Remove one binding managed by this concrete store type."""
        binding = await self.resolve(service, affinity_id)
        if binding is None:
            return False
        return await self.store.delete(
            self.binding_key(service, affinity_id)
        )

    async def _keys(self, prefix: str) -> list[str]:
        try:
            return await self.store.keys(prefix=prefix)
        except TypeError:
            return [
                key
                for key in await self.store.keys()
                if key.startswith(prefix)
            ]

    async def _items(
        self,
        prefix: str,
        *,
        service: Optional[str] = None,
        server_id: Optional[str] = None,
    ) -> list[tuple[str, bytes | str]]:
        affinity_items = getattr(self.store, "affinity_items", None)
        if callable(affinity_items):
            return await affinity_items(
                service=service,
                affinity_type=self.binding_class().type_name(),
                server_id=server_id,
            )
        items = getattr(self.store, "items", None)
        if callable(items):
            try:
                return await items(prefix=prefix)
            except TypeError:
                pass
        keys = await self._keys(prefix)
        values = await asyncio.gather(*(self.store.get(key) for key in keys))
        return [
            (key, value)
            for key, value in zip(keys, values)
            if value is not None
        ]

    async def list_bindings(
        self,
        service: Optional[str] = None,
    ) -> list[BindingT]:
        """List live bindings managed by this concrete store type."""
        prefix = (
            self.service_prefix(service)
            if service is not None
            else AFFINITY_KEY_PREFIX
        )
        bindings = []
        expected_class = self.binding_class()
        for key, payload in await self._items(prefix, service=service):
            binding = await self._load_payload(key, payload)
            if binding is not None and isinstance(binding, expected_class):
                bindings.append(cast(BindingT, binding))
        return bindings

    async def release_server(
        self,
        server_id: str,
        service: Optional[str] = None,
    ) -> int:
        """Release this store type's bindings owned by one server."""
        self._validate_text(server_id, "server_id")
        if service is not None:
            self._validate_text(service, "service")
        delete_affinity_bindings = getattr(
            self.store,
            "delete_affinity_bindings",
            None,
        )
        if callable(delete_affinity_bindings):
            return await delete_affinity_bindings(
                affinity_type=self.binding_class().type_name(),
                server_id=server_id,
                service=service,
            )
        prefix = (
            self.service_prefix(service)
            if service is not None
            else AFFINITY_KEY_PREFIX
        )
        released = 0
        expected_class = self.binding_class()
        for key, payload in await self._items(
            prefix,
            service=service,
            server_id=server_id,
        ):
            binding = await self._load_payload(key, payload)
            if (
                binding is not None
                and isinstance(binding, expected_class)
                and binding.server_id == server_id
                and await self.store.delete(key)
            ):
                released += 1
        return released


class StrictAffinityBindingStore(
    AffinityBindingStore[StrictAffinityBinding]
):
    """Store for bindings that must fail when their server is unavailable."""

    @classmethod
    def binding_class(cls) -> type[StrictAffinityBinding]:
        return StrictAffinityBinding


class SoftAffinityBindingStore(
    AffinityBindingStore[SoftAffinityBinding]
):
    """Store for bindings that may move to a replacement server."""

    @classmethod
    def binding_class(cls) -> type[SoftAffinityBinding]:
        return SoftAffinityBinding

    async def handoff(
        self,
        service: str,
        affinity_id: str,
        server_id: str,
        server_uri: str,
        ttl_seconds: Optional[float] = None,
    ) -> Optional[SoftAffinityBinding]:
        """Move a live soft binding to another server and record the transition."""
        self._validate_text(server_id, "server_id")
        self._validate_text(server_uri, "server_uri")
        binding = await self.resolve(service, affinity_id)
        if binding is None:
            return None
        ttl = self._ttl(ttl_seconds)
        key = self.binding_key(service, affinity_id)
        if (
            binding.server_id == server_id
            and binding.server_uri == server_uri
        ):
            return await self._refresh(key, binding, ttl)

        now = self._now()
        handed_off = replace(
            binding,
            server_id=server_id,
            server_uri=server_uri,
            previous_server_id=binding.server_id,
            handoff_count=binding.handoff_count + 1,
            last_handoff_at=now,
            last_used_at=now,
            expires_at=now + ttl,
        )
        await self._set_binding(key, handed_off, ttl)
        return handed_off
