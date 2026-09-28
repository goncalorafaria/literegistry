import asyncio
import threading
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from literegistry.api import ServiceAPI
from literegistry.services.executable_wrapper import ExecutableWrapper


class Registry:
    def __init__(self, *, fail_deregister=False):
        self.loops = []
        self.calls = []
        self.recovered = threading.Event()
        self.fail_deregister = fail_deregister
        self.store = SimpleNamespace(close=self.close)

    def record(self, name):
        self.loops.append(asyncio.get_running_loop())
        self.calls.append(name)

    async def register_server(self, **kwargs):
        self.record("register")

    async def heartbeat(self, *args):
        self.record("heartbeat")
        if self.calls.count("heartbeat") == 1:
            raise ConnectionError("temporary registry outage")
        self.recovered.set()

    async def deregister(self):
        self.record("deregister")
        if self.fail_deregister:
            raise ConnectionError("registry unavailable at shutdown")

    async def close(self):
        self.record("close")


@pytest.mark.parametrize("fail_deregister", [False, True])
def test_service_heartbeat_recovers_and_closes_on_owning_loop(monkeypatch, fail_deregister):
    registry = Registry(fail_deregister=fail_deregister)
    monkeypatch.setattr("literegistry.api.get_kvstore", lambda _: registry.store)
    monkeypatch.setattr("literegistry.api.ServerRegistry", lambda **_: registry)
    app = ServiceAPI(hostname="localhost", port=1234, heartbeat_interval=0.001)

    async def scenario():
        expected = pytest.raises(ConnectionError) if fail_deregister else nullcontext()
        with expected:
            async with app.router.lifespan_context(app):
                async def wait_for_recovery():
                    while not registry.recovered.is_set():
                        await asyncio.sleep(0.001)
                await asyncio.wait_for(wait_for_recovery(), timeout=2)
        assert app.heartbeat_task.done()
        assert registry.calls[-2:] == ["deregister", "close"]
        assert set(registry.loops) == {asyncio.get_running_loop()}

    asyncio.run(scenario())


class Wrapper(ExecutableWrapper):
    def get_server_command(self):
        return []

    def get_model_flag(self):
        return "--model"

    def get_server_name(self):
        return "test"


@pytest.mark.parametrize("fail_deregister", [False, True])
def test_model_heartbeat_retries_and_cleanup_joins_thread(monkeypatch, fail_deregister):
    registry = Registry(fail_deregister=fail_deregister)
    monkeypatch.setattr("literegistry.services.executable_wrapper.get_kvstore", lambda _: registry.store)
    monkeypatch.setattr("literegistry.services.executable_wrapper.ServerRegistry", lambda **_: registry)
    wrapper = Wrapper("unused", heartbeat_interval=0.001)
    wrapper.check_health = lambda: True
    wrapper.process = Mock()
    wrapper.heartbeat_thread = threading.Thread(target=wrapper.heartbeat_loop)
    wrapper.heartbeat_thread.start()
    try:
        assert registry.recovered.wait(timeout=2)
    finally:
        wrapper.cleanup()
    assert not wrapper.heartbeat_thread.is_alive()
    assert registry.calls[0] == "register"
    assert registry.calls[-2:] == ["deregister", "close"]
    assert len(set(registry.loops)) == 1
    wrapper.process.terminate.assert_called_once()
    wrapper.process.wait.assert_called_once()
