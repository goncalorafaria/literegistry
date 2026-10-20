import asyncio, time, json
from aiohttp import web
from literegistry_podman_client import PodmanClient, PodmanGatewayError
from literegistry_podman_client.discovery import GatewayDiscovery


async def main():
    checks = []
    attempts = []

    async def unavailable():
        attempts.append(1)
        raise OSError("discovery unavailable")

    c = PodmanClient(
        gateway_discovery=GatewayDiscovery(unavailable), recovery_timeout=0.7
    )
    try:
        await c._request("POST", "affinity/podman")
        assert False
    except PodmanGatewayError as e:
        assert e.request_not_sent
    assert len(attempts) > 1
    await c.aclose()
    checks.append("open_discovery_retries")
    for name, safe in [("unsent_retried", True), ("unknown_not_retried", False)]:
        c = PodmanClient("http://localhost:1", recovery_timeout=2)
        calls = []

        async def once(*args):
            calls.append(1)
            if len(calls) == 1:
                e = PodmanGatewayError("injected")
                e.request_not_sent = safe
                raise e
            return {"ok": True}

        c._request_once = once
        try:
            r = await c._request("POST", "affinity/podman")
            assert safe and r["ok"]
        except PodmanGatewayError:
            assert not safe
        assert len(calls) == (2 if safe else 1)
        checks.append(name)
    c = PodmanClient("http://localhost:1", recovery_timeout=0.05)

    async def fail(*args):
        e = PodmanGatewayError("unsent")
        e.request_not_sent = True
        raise e

    c._request_once = fail
    t = time.monotonic()
    try:
        await c._request("POST", "affinity/handshake")
        assert False
    except PodmanGatewayError:
        pass
    assert time.monotonic() - t < 0.5
    checks.append("bounded_wait")
    lookups = []

    async def lookup():
        lookups.append(1)
        await asyncio.sleep(0.02)
        return "http://localhost:1"

    d = GatewayDiscovery(lookup)
    await asyncio.gather(*(d.resolve() for _ in range(32)))
    assert len(lookups) == 1
    checks.append("shared_discovery")
    # Actual lost response: handler sees command once, then drops connection.
    seen = []

    async def drop(req):
        seen.append(await req.json())
        req.transport.close()
        return web.json_response({})

    app = web.Application()
    app.router.add_post("/affinity/podman", drop)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    async with PodmanClient(f"http://127.0.0.1:{port}", recovery_timeout=1) as c:
        try:
            await c.execute("abc", "touch /tmp/test")
            assert False
        except PodmanGatewayError:
            pass
    assert len(seen) == 1
    checks.append("real_disconnect_no_duplicate")
    await runner.cleanup()
    print(json.dumps({"passed": checks}))


def test_recovery_contract():
    asyncio.run(main())


if __name__ == "__main__":
    test_recovery_contract()
