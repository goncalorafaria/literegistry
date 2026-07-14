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


def test_retryable_python_failure_rotates_to_an_untried_server():
    registry = LegacySampleServersRegistry()
    client = RegistryHTTPClient(
        registry,
        "python",
        max_retries=2,
        retry_budget_seconds=1,
        retry_backoff_seconds=0,
    )
    requests = []

    async def fail_then_succeed(server, endpoint, payload):
        requests.append(server)
        if server == "http://bad-server":
            return {
                "success": False,
                "retryable": True,
                "stderr": "worker pool crashed",
            }
        return {"success": True, "server": server}

    client._make_http_request = fail_then_succeed

    result, _ = asyncio.run(client.request_with_rotation("python", {"code": "1"}))

    assert result == {"success": True, "server": "http://good-server"}
    assert requests == ["http://bad-server", "http://good-server"]
    assert registry.latencies == [
        ("http://bad-server", False),
        ("http://good-server", True),
    ]


def test_python_retry_budget_stops_a_slow_retry_loop():
    registry = LegacySampleServersRegistry()
    client = RegistryHTTPClient(
        registry,
        "python",
        max_retries=3,
        retry_budget_seconds=0.01,
        retry_backoff_seconds=0,
    )

    async def slow_failure(server, endpoint, payload):
        await asyncio.sleep(0.02)
        raise RuntimeError("timed out")

    client._make_http_request = slow_failure

    try:
        asyncio.run(client.request_with_rotation("python", {"code": "1"}))
    except RuntimeError as exc:
        assert "Retry budget" in str(exc)
    else:
        raise AssertionError("expected retry budget failure")


def test_retryable_code_executor_failure_rotates_to_untried_server():
    class Registry:
        def __init__(self):
            self.latencies = []

        async def sample_servers(self, value, n, force=False):
            return [("http://bad-server", 0.5), ("http://good-server", 0.5)]

        def report_latency(self, uri, response_time, prob=1.0, success=True):
            self.latencies.append((uri, success))

    registry = Registry()
    client = RegistryHTTPClient(
        registry, "python", max_retries=3, retry_backoff_seconds=0
    )
    calls = []

    async def fail_then_succeed(server, endpoint, payload):
        calls.append(server)
        if server == "http://bad-server":
            return {"success": False, "retryable": True, "stderr": "pool restarted"}
        return {"success": True, "server": server}

    client._make_http_request = fail_then_succeed

    result, _ = asyncio.run(client.request_with_rotation("python", {"code": "1"}))

    assert result == {"success": True, "server": "http://good-server"}
    assert calls == ["http://bad-server", "http://good-server"]
    assert registry.latencies == [
        ("http://bad-server", False),
        ("http://good-server", True),
    ]


def test_non_retryable_user_code_failure_is_returned_without_rotation():
    class Registry:
        def __init__(self):
            self.latencies = []

        async def sample_servers(self, value, n, force=False):
            return [("http://server", 1.0)]

        def report_latency(self, uri, response_time, prob=1.0, success=True):
            self.latencies.append((uri, success))

    registry = Registry()
    client = RegistryHTTPClient(registry, "python", max_retries=3)

    async def user_code_error(server, endpoint, payload):
        return {"success": False, "retryable": False, "stderr": "NameError: x"}

    client._make_http_request = user_code_error

    result, _ = asyncio.run(client.request_with_rotation("python", {"code": "x"}))

    assert result["success"] is False
    assert registry.latencies == [("http://server", True)]
