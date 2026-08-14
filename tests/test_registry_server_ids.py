"""Regression coverage for co-located registry workers."""

from __future__ import annotations

import pytest
import asyncio

from literegistry.registry import ServerRegistry


class MemoryStore:
    def __init__(self) -> None:
        self.values: dict[str, bytes] = {}

    async def get(self, key: str) -> bytes | None:
        return self.values.get(key)

    async def set(self, key: str, value: bytes | str) -> bool:
        self.values[key] = value.encode() if isinstance(value, str) else value
        return True

    async def delete(self, key: str) -> bool:
        return self.values.pop(key, None) is not None

    async def exists(self, key: str) -> bool:
        return key in self.values

    async def keys(self) -> list[str]:
        return list(self.values)


def test_colocated_registries_have_independent_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    async def check() -> None:
        store = MemoryStore()
        first = ServerRegistry(store)
        second = ServerRegistry(store)

        assert first.server_id != second.server_id
        await first.register_server("http://host", 8001, {"model_path": "model"})
        await second.register_server("http://host", 8002, {"model_path": "model"})
        await first.deregister()

        roster = await second.roster()
        assert [server["port"] for server in roster["servers"]] == [8002]

    monkeypatch.setattr("literegistry.registry.socket.gethostname", lambda: "holmes-cs-aus-553.reviz.ai2.in")
    asyncio.run(check())
