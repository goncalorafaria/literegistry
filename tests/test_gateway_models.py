import asyncio
import json

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
