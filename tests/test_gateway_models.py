import asyncio
import json
from unittest.mock import patch

from starlette.requests import Request
from literegistry.gateway import StarletteGatewayServer


class FakeRegistry:
    def __init__(self):
        self.calls = []

    async def models(self, force=False):
        self.calls.append(force)
        return {"cached-model": [{"uri": "http://localhost:8000"}]}


def test_gateway_model_list_forces_registry_refresh():
    registry = FakeRegistry()
    server = StarletteGatewayServer(registry)

    response = asyncio.run(server.list_models(None))
    payload = json.loads(response.body.decode("utf-8"))

    assert registry.calls == [True]
    assert "cached-model" in payload["models"]


def test_gateway_uses_dedicated_python_retry_policy():
    registry = FakeRegistry()
    server = StarletteGatewayServer(
        registry,
        timeout=99,
        python_timeout=7,
        max_retries=20,
        python_max_retries=2,
        python_retry_budget_seconds=8,
    )
    captured = {}

    class FakeClient:
        def __init__(self, *args, **kwargs):
            captured.update(kwargs)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def request_with_rotation(self, endpoint, payload):
            return {"success": True}, 0

    async def receive():
        return {
            "type": "http.request",
            "body": b'{"code": "print(1)"}',
            "more_body": False,
        }

    request = Request({"type": "http", "method": "POST", "headers": []}, receive)
    with patch("literegistry.gateway.RegistryHTTPClient", FakeClient):
        response = asyncio.run(server.handle_python(request))

    assert response.status_code == 200
    assert captured["timeout"] == 7
    assert captured["connect_timeout"] == 3
    assert captured["max_retries"] == 2
    assert captured["retry_budget_seconds"] == 8
    assert captured["retry_backoff_seconds"] == 0.1


def test_gateway_routes_terminal_pipeline_to_terminal_workers():
    registry = FakeRegistry()
    server = StarletteGatewayServer(
        registry,
        terminal_timeout=9,
        terminal_max_retries=2,
        terminal_retry_budget_seconds=10,
    )
    captured = {}

    class FakeClient:
        def __init__(self, registry, model, **kwargs):
            captured["model"] = model
            captured.update(kwargs)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def request_with_rotation(self, endpoint, payload):
            captured["endpoint"] = endpoint
            captured["payload"] = payload
            return {"success": True, "stdout": "ERROR"}, 0

    async def receive():
        return {
            "type": "http.request",
            "body": b'{"contents": "ERROR disk full\\n", "command": "rg ERROR"}',
            "more_body": False,
        }

    request = Request({"type": "http", "method": "POST", "headers": []}, receive)
    with patch("literegistry.gateway.RegistryHTTPClient", FakeClient):
        response = asyncio.run(server.handle_terminal(request))

    assert response.status_code == 200
    assert captured["model"] == "terminal"
    assert captured["endpoint"] == "terminal"
    assert captured["timeout"] == 9
    assert captured["max_retries"] == 2
    assert captured["retry_budget_seconds"] == 10
