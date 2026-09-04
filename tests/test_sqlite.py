from __future__ import annotations

import asyncio
import json
from pathlib import Path
import sqlite3

import pytest

from literegistry import SQLiteKVStore, get_kvstore, sqlite_registry_path
from literegistry.affinity import (
    SoftAffinityBindingStore,
    StrictAffinityBindingStore,
)
from literegistry.client import RegistryClient
from literegistry.registry import ServerRegistry


class _ObservedSQLiteKVStore(SQLiteKVStore):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.generic_item_queries = 0
        self.affinity_queries = 0
        self.owner_deletes = 0

    async def items(self, prefix=None):
        self.generic_item_queries += 1
        return await super().items(prefix=prefix)

    async def affinity_items(self, **filters):
        self.affinity_queries += 1
        return await super().affinity_items(**filters)

    async def delete_affinity_bindings(self, **filters):
        self.owner_deletes += 1
        return await super().delete_affinity_bindings(**filters)


def test_sqlite_store_crud_prefix_and_persistence(tmp_path: Path) -> None:
    async def scenario() -> None:
        database = tmp_path / "nested" / "registry.sqlite3"
        store = SQLiteKVStore(database)
        assert await store.set("server_one", "first")
        assert await store.set("server_two", b"second")
        assert await store.set("affinity_one", "other")
        assert await store.get("server_one") == b"first"
        assert await store.exists("server_two")
        assert await store.keys(prefix="server_") == ["server_one", "server_two"]
        assert await store.delete("server_one")
        assert not await store.delete("server_one")
        await store.close()

        reopened = SQLiteKVStore(database)
        assert await reopened.get("server_two") == b"second"
        assert await reopened.keys() == ["affinity_one", "server_two"]
        await reopened.close()

    asyncio.run(scenario())


def test_sqlite_store_expires_ttl_records(tmp_path: Path) -> None:
    async def scenario() -> None:
        store = SQLiteKVStore(tmp_path / "registry.sqlite3")
        await store.set("short", "value", ttl_seconds=0.05)
        await store.set("durable", "value")
        assert await store.get("short") == b"value"
        await asyncio.sleep(0.08)
        assert await store.get("short") is None
        assert not await store.exists("short")
        assert await store.keys() == ["durable"]
        await store.close()

    asyncio.run(scenario())


def test_sqlite_prefix_is_literal(tmp_path: Path) -> None:
    async def scenario() -> None:
        store = SQLiteKVStore(tmp_path / "registry.sqlite3")
        await store.set("soft_%_one", "one")
        await store.set("soft_value", "two")
        assert await store.keys(prefix="soft_%") == ["soft_%_one"]
        await store.close()

    asyncio.run(scenario())


def test_sqlite_store_serializes_concurrent_independent_writers(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        database = tmp_path / "registry.sqlite3"
        first = SQLiteKVStore(database)
        second = SQLiteKVStore(database)
        await asyncio.gather(
            *(first.set(f"first_{index}", str(index)) for index in range(20)),
            *(second.set(f"second_{index}", str(index)) for index in range(20)),
        )
        assert len(await first.keys(prefix="first_")) == 20
        assert len(await second.keys(prefix="second_")) == 20
        assert len(await first.keys()) == 40
        await first.close()
        await second.close()

    asyncio.run(scenario())


def test_get_kvstore_selects_sqlite_uri(tmp_path: Path) -> None:
    database = tmp_path / "registry.sqlite3"
    uri = f"sqlite://{database}"
    store = get_kvstore(uri)
    assert isinstance(store, SQLiteKVStore)
    assert store.path == database
    assert sqlite_registry_path(uri) == database
    asyncio.run(store.close())


def test_sqlite_store_supports_registry_lifecycle(tmp_path: Path) -> None:
    async def scenario() -> None:
        store = SQLiteKVStore(tmp_path / "registry.sqlite3")
        server = ServerRegistry(store)
        await server.register_server(
            "http://worker.example",
            8080,
            {"model_path": "python"},
        )
        client = RegistryClient(store)
        assert await client.get_all("python", force=True) == [
            "http://worker.example:8080"
        ]
        await store.close()

    asyncio.run(scenario())


def test_sqlite_affinity_uses_dedicated_indexed_table(tmp_path: Path) -> None:
    async def scenario() -> None:
        database = tmp_path / "registry.sqlite3"
        kv = _ObservedSQLiteKVStore(database)
        strict = StrictAffinityBindingStore(kv)
        soft = SoftAffinityBindingStore(kv)
        await strict.bind("podman", "strict-one", "server-a", "http://a")
        await strict.bind("podman", "strict-two", "server-b", "http://b")
        await soft.bind("mirror", "image-one", "server-a", "http://a")
        await kv.set("server:health", "healthy")

        assert len(await strict.list_bindings("podman")) == 2
        assert len(await soft.list_bindings("mirror")) == 1
        assert await strict.release_server("server-a", "podman") == 1
        assert kv.affinity_queries == 2
        assert kv.owner_deletes == 1
        assert kv.generic_item_queries == 0
        await kv.close()

        with sqlite3.connect(database) as connection:
            affinity_count = connection.execute(
                "SELECT count(*) FROM literegistry_affinity"
            ).fetchone()[0]
            generic_count = connection.execute(
                "SELECT count(*) FROM literegistry_kv"
            ).fetchone()[0]
            service_plan = " ".join(
                str(row[-1])
                for row in connection.execute(
                    "EXPLAIN QUERY PLAN SELECT key, value "
                    "FROM literegistry_affinity "
                    "WHERE expires_at > ? AND affinity_type = ? AND service = ?",
                    (0, "strict", "podman"),
                )
            )
            server_plan = " ".join(
                str(row[-1])
                for row in connection.execute(
                    "EXPLAIN QUERY PLAN SELECT key, value "
                    "FROM literegistry_affinity "
                    "WHERE expires_at > ? AND affinity_type = ? AND server_id = ?",
                    (0, "strict", "server-b"),
                )
            )

        assert affinity_count == 2
        assert generic_count == 1
        assert "literegistry_affinity_lookup" in service_plan
        assert "literegistry_affinity_server" in server_plan

    asyncio.run(scenario())


def test_sqlite_migrates_legacy_affinity_rows_transparently(tmp_path: Path) -> None:
    async def scenario() -> None:
        database = tmp_path / "registry.sqlite3"
        kv = SQLiteKVStore(database)
        bindings = StrictAffinityBindingStore(kv)
        binding = await bindings.bind(
            "podman",
            "legacy-container",
            "server-a",
            "http://a",
        )
        key = bindings.binding_key("podman", "legacy-container")
        payload = json.dumps(binding.to_dict()).encode("utf-8")
        await kv.close()

        with sqlite3.connect(database) as connection:
            connection.execute(
                "DELETE FROM literegistry_affinity WHERE key = ?",
                (key,),
            )
            connection.execute(
                "INSERT INTO literegistry_kv (key, value, expires_at) "
                "VALUES (?, ?, ?)",
                (key, payload, binding.expires_at),
            )

        reopened = SQLiteKVStore(database)
        resolved = await StrictAffinityBindingStore(reopened).resolve(
            "podman",
            "legacy-container",
        )
        assert resolved is not None
        assert resolved.server_id == "server-a"
        await reopened.close()

        with sqlite3.connect(database) as connection:
            assert connection.execute(
                "SELECT count(*) FROM literegistry_affinity WHERE key = ?",
                (key,),
            ).fetchone()[0] == 1
            assert connection.execute(
                "SELECT count(*) FROM literegistry_kv WHERE key = ?",
                (key,),
            ).fetchone()[0] == 0

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "uri",
    [
        "sqlite://remote.example/registry.sqlite3",
        "sqlite:///tmp/registry.sqlite3?mode=ro",
        "sqlite://",
        "sqlite:///:memory:",
    ],
)
def test_sqlite_uri_rejects_unsupported_forms(uri: str) -> None:
    with pytest.raises(ValueError):
        sqlite_registry_path(uri)
