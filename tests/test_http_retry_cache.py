import asyncio

from literegistry.http import RegistryHTTPClient


class LegacySampleServersRegistry:
    def __init__(self):
        self.sample_calls = 0
        self.force_values = []
        self.latencies = []

    async def sample_servers(self, value, n, force=False):
        self.sample_calls += 1
        self.force_values.append(force)
        if self.sample_calls == 1:
            return [("http://bad-server", 1.0)]
        return [("http://good-server", 1.0)]

    def report_latency(self, uri, response_time, prob=1.0, success=True):
        self.latencies.append((uri, success))


def test_retry_refresh_does_not_require_force_sample_servers_keyword():
    registry = LegacySampleServersRegistry()
    client = RegistryHTTPClient(registry, "python", max_retries=2)

    async def fail_then_succeed(server, endpoint, payload):
        if server == "http://bad-server":
            raise RuntimeError("failed")
        return {"status": "ok", "server": server}

    client._make_http_request = fail_then_succeed

    result, _ = asyncio.run(client.request_with_rotation("python", {"code": "1"}))

    assert result == {"status": "ok", "server": "http://good-server"}
    assert registry.sample_calls == 2
    assert registry.force_values == [False, True]
