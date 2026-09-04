"""SQLite-backed key-value storage for LiteRegistry.

SQLite is useful when a deployment wants one durable database file without
running a Redis service. Operations run in worker threads so synchronous
``sqlite3`` calls do not block the asyncio event loop.
"""

from __future__ import annotations

import asyncio
import json
import math
from pathlib import Path
import sqlite3
import threading
import time
from typing import Optional, Union
from urllib.parse import quote, unquote, urlsplit

from literegistry.kvstore import KeyValueStore


SQLITE_REGISTRY_SCHEME = "sqlite:"
_TABLE = "literegistry_kv"
_AFFINITY_TABLE = "literegistry_affinity"
_AFFINITY_PREFIX = "affinity:"
_MAX_UNICODE = 0x10FFFF


def _prefix_bounds(prefix: str) -> tuple[str, str | None]:
    """Return the indexed half-open key range containing one literal prefix."""

    if not prefix:
        return "", None
    codepoints = [ord(character) for character in prefix]
    for index in range(len(codepoints) - 1, -1, -1):
        if codepoints[index] < _MAX_UNICODE:
            upper = (
                prefix[:index]
                + chr(codepoints[index] + 1)
            )
            return prefix, upper
    return prefix, None


def sqlite_registry_path(value: str | Path) -> Path:
    """Return the database path encoded by a ``sqlite:`` registry URI."""

    text = str(value)
    if not text.startswith(SQLITE_REGISTRY_SCHEME):
        return Path(text).expanduser()

    parsed = urlsplit(text)
    if parsed.scheme != "sqlite":
        raise ValueError("SQLite registry URI must use the sqlite: scheme")
    if parsed.query or parsed.fragment:
        raise ValueError("SQLite registry URI cannot contain a query or fragment")
    if parsed.netloc not in {"", "localhost"}:
        raise ValueError(
            "SQLite is file-backed and cannot use a remote hostname; "
            "use sqlite:///absolute/path/to/registry.sqlite3"
        )

    raw_path = unquote(parsed.path)
    if not raw_path:
        raise ValueError("SQLite registry URI must include a database path")
    if raw_path in {":memory:", "/:memory:"}:
        raise ValueError("in-memory SQLite registries are not supported")
    return Path(raw_path).expanduser()


class SQLiteKVStore(KeyValueStore):
    """Persistent SQLite implementation of LiteRegistry's KV contract.

    A short-lived SQLite connection is opened for each operation. This keeps
    the store safe when one instance is used by multiple asyncio tasks and
    avoids retaining a connection across a process fork. SQLite itself
    coordinates independent processes through database file locks.
    """

    def __init__(
        self,
        database: str | Path,
        *,
        timeout: float = 30.0,
        cleanup_interval: float = 60.0,
    ) -> None:
        self.path = sqlite_registry_path(database).absolute()
        self.timeout = float(timeout)
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("timeout must be a finite value greater than zero")
        self.cleanup_interval = float(cleanup_interval)
        if (
            not math.isfinite(self.cleanup_interval)
            or self.cleanup_interval <= 0
        ):
            raise ValueError(
                "cleanup_interval must be a finite value greater than zero"
            )
        self._closed = False
        self._cleanup_lock = threading.Lock()
        self._next_cleanup = time.monotonic() + self.cleanup_interval
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_sync()

    @staticmethod
    def _validate_ttl(ttl_seconds: Optional[float]) -> Optional[float]:
        if ttl_seconds is None:
            return None
        ttl = float(ttl_seconds)
        if not math.isfinite(ttl) or ttl <= 0:
            raise ValueError("ttl_seconds must be a finite value greater than zero")
        return ttl

    def _connect(self) -> sqlite3.Connection:
        if self._closed:
            raise RuntimeError("SQLiteKVStore is closed")
        connection = sqlite3.connect(str(self.path), timeout=self.timeout)
        connection.execute(f"PRAGMA busy_timeout = {math.ceil(self.timeout * 1000)}")
        return connection

    def _initialize_sync(self) -> None:
        with self._connect() as connection:
            # DELETE journaling avoids WAL's shared-memory requirement and is
            # the more portable choice when the database lives on shared disk.
            connection.execute("PRAGMA journal_mode = DELETE")
            connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {_TABLE} (
                    key TEXT PRIMARY KEY,
                    value BLOB NOT NULL,
                    expires_at REAL
                ) WITHOUT ROWID
                """
            )
            connection.execute(
                f"CREATE INDEX IF NOT EXISTS {_TABLE}_expires_at "
                f"ON {_TABLE}(expires_at)"
            )
            connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {_AFFINITY_TABLE} (
                    key TEXT PRIMARY KEY,
                    service TEXT NOT NULL,
                    affinity_type TEXT NOT NULL,
                    server_id TEXT NOT NULL,
                    value BLOB NOT NULL,
                    expires_at REAL NOT NULL
                ) WITHOUT ROWID
                """
            )
            connection.execute(
                f"CREATE INDEX IF NOT EXISTS {_AFFINITY_TABLE}_lookup "
                f"ON {_AFFINITY_TABLE}(affinity_type, service, server_id)"
            )
            connection.execute(
                f"CREATE INDEX IF NOT EXISTS {_AFFINITY_TABLE}_server "
                f"ON {_AFFINITY_TABLE}(affinity_type, server_id, service)"
            )
            connection.execute(
                f"CREATE INDEX IF NOT EXISTS {_AFFINITY_TABLE}_expires_at "
                f"ON {_AFFINITY_TABLE}(expires_at)"
            )
            self._migrate_affinity_rows(connection)

    @staticmethod
    def _affinity_columns(
        key: str,
        payload: bytes,
        expires_at: Optional[float],
    ) -> Optional[tuple[str, str, str, float]]:
        """Extract indexed fields from a valid internal affinity record."""

        if not key.startswith(_AFFINITY_PREFIX):
            return None
        try:
            value = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None
        if not isinstance(value, dict):
            return None
        service = value.get("service")
        affinity_hash = value.get("affinity_id_hash")
        affinity_type = value.get("affinity_type", "strict")
        server_id = value.get("server_id")
        if not all(
            isinstance(item, str) and item
            for item in (service, affinity_hash, affinity_type, server_id)
        ):
            return None
        expected_key = (
            f"{_AFFINITY_PREFIX}{quote(service, safe='')}:{affinity_hash}"
        )
        if key != expected_key or affinity_type not in {"strict", "soft"}:
            return None
        logical_expiration = value.get("expires_at")
        try:
            effective_expiration = float(
                expires_at if expires_at is not None else logical_expiration
            )
        except (TypeError, ValueError):
            return None
        if not math.isfinite(effective_expiration):
            return None
        return service, affinity_type, server_id, effective_expiration

    def _migrate_affinity_rows(self, connection: sqlite3.Connection) -> None:
        """Move affinity records from the legacy generic table in-place."""

        lower, upper = _prefix_bounds(_AFFINITY_PREFIX)
        rows = connection.execute(
            f"SELECT key, value, expires_at FROM {_TABLE} "
            "WHERE key >= ? AND key < ?",
            (lower, upper),
        ).fetchall()
        for key, value, expires_at in rows:
            payload = bytes(value)
            columns = self._affinity_columns(str(key), payload, expires_at)
            if columns is None:
                continue
            service, affinity_type, server_id, effective_expiration = columns
            connection.execute(
                f"""
                INSERT INTO {_AFFINITY_TABLE}
                    (key, service, affinity_type, server_id, value, expires_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    service = excluded.service,
                    affinity_type = excluded.affinity_type,
                    server_id = excluded.server_id,
                    value = excluded.value,
                    expires_at = excluded.expires_at
                """,
                (
                    key,
                    service,
                    affinity_type,
                    server_id,
                    sqlite3.Binary(payload),
                    effective_expiration,
                ),
            )
            connection.execute(f"DELETE FROM {_TABLE} WHERE key = ?", (key,))

    def _cleanup_expired_if_due(
        self,
        connection: sqlite3.Connection,
        now: float,
    ) -> None:
        """Amortize TTL cleanup so ordinary reads do not become write locks."""

        monotonic_now = time.monotonic()
        if monotonic_now < self._next_cleanup:
            return
        with self._cleanup_lock:
            if monotonic_now < self._next_cleanup:
                return
            connection.execute(
                f"DELETE FROM {_TABLE} "
                "WHERE expires_at IS NOT NULL AND expires_at <= ?",
                (now,),
            )
            connection.execute(
                f"DELETE FROM {_AFFINITY_TABLE} WHERE expires_at <= ?",
                (now,),
            )
            self._next_cleanup = monotonic_now + self.cleanup_interval

    @staticmethod
    def _live_key_query(
        prefix: Optional[str],
        *,
        include_value: bool,
        table: str = _TABLE,
    ) -> tuple[str, tuple[object, ...]]:
        columns = "key, value" if include_value else "key"
        where = "(expires_at IS NULL OR expires_at > ?)"
        parameters: list[object] = [time.time()]
        if prefix is not None:
            lower, upper = _prefix_bounds(prefix)
            where += " AND key >= ?"
            parameters.append(lower)
            if upper is not None:
                where += " AND key < ?"
                parameters.append(upper)
        return (
            f"SELECT {columns} FROM {table} WHERE {where} ORDER BY key",
            tuple(parameters),
        )

    async def get(self, key: str) -> Optional[bytes]:
        def operation() -> Optional[bytes]:
            now = time.time()
            with self._connect() as connection:
                tables = (
                    (_AFFINITY_TABLE, _TABLE)
                    if key.startswith(_AFFINITY_PREFIX)
                    else (_TABLE,)
                )
                row = None
                row_table = _TABLE
                for table in tables:
                    row = connection.execute(
                        f"SELECT value, expires_at FROM {table} WHERE key = ?",
                        (key,),
                    ).fetchone()
                    if row is not None:
                        row_table = table
                        break
                if row is None:
                    return None
                if row[1] is None or float(row[1]) > now:
                    return bytes(row[0])
                connection.execute(
                    f"DELETE FROM {row_table} "
                    "WHERE key = ? AND expires_at IS NOT NULL AND expires_at <= ?",
                    (key, now),
                )
                return None

        return await asyncio.to_thread(operation)

    async def set(
        self,
        key: str,
        value: Union[bytes, str],
        ttl_seconds: Optional[float] = None,
    ) -> bool:
        ttl = self._validate_ttl(ttl_seconds)
        payload = value.encode("utf-8") if isinstance(value, str) else bytes(value)
        expires_at = None if ttl is None else time.time() + ttl
        affinity_columns = self._affinity_columns(key, payload, expires_at)

        def operation() -> None:
            with self._connect() as connection:
                if affinity_columns is not None:
                    (
                        service,
                        affinity_type,
                        server_id,
                        affinity_expiration,
                    ) = affinity_columns
                    connection.execute(
                        f"""
                        INSERT INTO {_AFFINITY_TABLE}
                            (key, service, affinity_type, server_id, value, expires_at)
                        VALUES (?, ?, ?, ?, ?, ?)
                        ON CONFLICT(key) DO UPDATE SET
                            service = excluded.service,
                            affinity_type = excluded.affinity_type,
                            server_id = excluded.server_id,
                            value = excluded.value,
                            expires_at = excluded.expires_at
                        """,
                        (
                            key,
                            service,
                            affinity_type,
                            server_id,
                            sqlite3.Binary(payload),
                            affinity_expiration,
                        ),
                    )
                    connection.execute(f"DELETE FROM {_TABLE} WHERE key = ?", (key,))
                else:
                    connection.execute(
                        f"""
                        INSERT INTO {_TABLE} (key, value, expires_at)
                        VALUES (?, ?, ?)
                        ON CONFLICT(key) DO UPDATE SET
                            value = excluded.value,
                            expires_at = excluded.expires_at
                        """,
                        (key, sqlite3.Binary(payload), expires_at),
                    )
                    if key.startswith(_AFFINITY_PREFIX):
                        connection.execute(
                            f"DELETE FROM {_AFFINITY_TABLE} WHERE key = ?",
                            (key,),
                        )

        await asyncio.to_thread(operation)
        return True

    async def delete(self, key: str) -> bool:
        def operation() -> bool:
            with self._connect() as connection:
                cursor = connection.execute(
                    f"DELETE FROM {_TABLE} WHERE key = ?",
                    (key,),
                )
                deleted = cursor.rowcount
                if key.startswith(_AFFINITY_PREFIX):
                    cursor = connection.execute(
                        f"DELETE FROM {_AFFINITY_TABLE} WHERE key = ?",
                        (key,),
                    )
                    deleted += cursor.rowcount
                return deleted > 0

        return await asyncio.to_thread(operation)

    async def exists(self, key: str) -> bool:
        return await self.get(key) is not None

    async def keys(self, prefix: Optional[str] = None) -> list[str]:
        def operation() -> list[str]:
            now = time.time()
            with self._connect() as connection:
                self._cleanup_expired_if_due(connection, now)
                keys: set[str] = set()
                for table in (_TABLE, _AFFINITY_TABLE):
                    query, parameters = self._live_key_query(
                        prefix,
                        include_value=False,
                        table=table,
                    )
                    keys.update(
                        str(row[0])
                        for row in connection.execute(query, parameters).fetchall()
                    )
                return sorted(keys)

        return await asyncio.to_thread(operation)

    async def items(
        self,
        prefix: Optional[str] = None,
    ) -> list[tuple[str, bytes]]:
        """Return matching live rows with one indexed SQLite query."""

        def operation() -> list[tuple[str, bytes]]:
            now = time.time()
            with self._connect() as connection:
                self._cleanup_expired_if_due(connection, now)
                items: dict[str, bytes] = {}
                for table in (_TABLE, _AFFINITY_TABLE):
                    query, parameters = self._live_key_query(
                        prefix,
                        include_value=True,
                        table=table,
                    )
                    items.update(
                        (str(key), bytes(value))
                        for key, value in connection.execute(
                            query,
                            parameters,
                        ).fetchall()
                    )
                return sorted(items.items())

        return await asyncio.to_thread(operation)

    async def affinity_items(
        self,
        *,
        service: Optional[str] = None,
        affinity_type: Optional[str] = None,
        server_id: Optional[str] = None,
    ) -> list[tuple[str, bytes]]:
        """Query affinity bindings using their dedicated indexed columns."""

        def operation() -> list[tuple[str, bytes]]:
            now = time.time()
            clauses = ["expires_at > ?"]
            parameters: list[object] = [now]
            for column, value in (
                ("affinity_type", affinity_type),
                ("service", service),
                ("server_id", server_id),
            ):
                if value is not None:
                    clauses.append(f"{column} = ?")
                    parameters.append(value)
            query = (
                f"SELECT key, value FROM {_AFFINITY_TABLE} WHERE "
                + " AND ".join(clauses)
                + " ORDER BY key"
            )
            with self._connect() as connection:
                self._cleanup_expired_if_due(connection, now)
                rows = connection.execute(query, parameters).fetchall()
                return [(str(key), bytes(value)) for key, value in rows]

        return await asyncio.to_thread(operation)

    async def delete_affinity_bindings(
        self,
        *,
        affinity_type: str,
        server_id: str,
        service: Optional[str] = None,
    ) -> int:
        """Delete one server's bindings with one indexed transaction."""

        def operation() -> int:
            clauses = [
                "affinity_type = ?",
                "server_id = ?",
                "expires_at > ?",
            ]
            parameters: list[object] = [affinity_type, server_id, time.time()]
            if service is not None:
                clauses.append("service = ?")
                parameters.append(service)
            with self._connect() as connection:
                cursor = connection.execute(
                    f"DELETE FROM {_AFFINITY_TABLE} WHERE "
                    + " AND ".join(clauses),
                    parameters,
                )
                return cursor.rowcount

        return await asyncio.to_thread(operation)

    async def close(self) -> None:
        """Prevent future operations; per-operation connections close eagerly."""

        self._closed = True
