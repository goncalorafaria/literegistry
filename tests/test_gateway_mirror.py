from __future__ import annotations

import asyncio
import tempfile

import aiohttp
from starlette.requests import Request

from literegistry.affinity import SoftAffinityBindingStore
from literegistry.gateway import Gateway, GatewayConfig
from literegistry.gateway.mirror import DockerMirrorProxy
from literegistry.kvstore import FileSystemKVStore


class _GatewayRegistry:
    store = object()

    async def models(self, force: bool = False):
        return {}


def test_canonical_gateway_installs_podman_and_mirror_routes() -> None:
    gateway = Gateway(_GatewayRegistry())
    paths = {route.path for route in gateway.app.routes}

    assert {
        "/affinity/handshake",
        "/affinity/podman",
        "/affinity/close",
        "/v2",
        "/v2/",
        "/v2/{path:path}",
    } <= paths
    assert gateway.app.state.strict_affinity is gateway.strict_affinity
    assert gateway.app.state.docker_mirror is gateway.docker_mirror



def test_mirror_soft_affinity_is_experimental_optional_and_enabled_by_default() -> None:
    default_gateway = Gateway(_GatewayRegistry())
    assert isinstance(
        default_gateway.docker_mirror.bindings,
        SoftAffinityBindingStore,
    )

    disabled_gateway = Gateway(
        _GatewayRegistry(),
        config=GatewayConfig(docker_mirror_soft_affinity=False),
    )
    assert disabled_gateway.docker_mirror.bindings is None


class _Content:
    def __init__(self, chunks: list[bytes]) -> None:
        self.chunks = chunks

    async def iter_chunked(self, size: int):
        assert size == 1024 * 1024
        for chunk in self.chunks:
            yield chunk


class _Response:
    def __init__(self, status: int, chunks: list[bytes], headers=None) -> None:
        self.status = status
        self.content = _Content(chunks)
        self.headers = headers or {}
        self.released = False

    def release(self) -> None:
        self.released = True


class _Session:
    def __init__(self, outcomes) -> None:
        self.outcomes = list(outcomes)
        self.calls = []

    async def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class _Sessions:
    def __init__(self, session) -> None:
        self.session = session

    def get_session(self):
        return self.session


class _Registry:
    def __init__(self, servers) -> None:
        self.servers = servers
        self.reports = []

    async def sample_servers(self, service, n):
        assert service == "docker-mirror"
        assert n == 3
        return self.servers

    def report_latency(self, server, latency, prob, success):
        self.reports.append((server, latency, prob, success))


def _request(
    method: str = "GET",
    path: str = "/v2/library/alpine/manifests/3.20",
    *,
    query: bytes = b"",
    raw_path: bytes | None = None,
    headers=None,
) -> Request:
    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    return Request(
        {
            "type": "http",
            "http_version": "1.1",
            "method": method,
            "scheme": "http",
            "server": ("gateway.example", 8080),
            "path": path,
            "raw_path": raw_path or path.encode(),
            "query_string": query,
            "headers": headers
            or [
                (b"host", b"gateway.example:8080"),
                (b"accept", b"application/vnd.oci.image.manifest.v1+json"),
                (b"range", b"bytes=0-1023"),
            ],
        },
        receive,
    )


def test_proxy_decodes_digest_and_preserves_duplicate_accept_headers() -> None:
    request = _request(
        path="/v2/library/ubuntu/manifests/sha256:abc",
        raw_path=b"/v2/library/ubuntu/manifests/sha256%3Aabc",
        headers=[
            (b"host", b"gateway.example:8080"),
            (b"accept", b"application/vnd.docker.distribution.manifest.v2+json"),
            (b"accept", b"application/vnd.oci.image.manifest.v1+json"),
            (b"connection", b"keep-alive"),
        ],
    )

    assert DockerMirrorProxy._upstream_url("http://mirror-a:5000", request) == (
        "http://mirror-a:5000/v2/library/ubuntu/manifests/sha256:abc"
    )
    assert DockerMirrorProxy._forward_request_headers(request) == [
        ("accept", "application/vnd.docker.distribution.manifest.v2+json"),
        ("accept", "application/vnd.oci.image.manifest.v1+json"),
    ]


def test_proxy_streams_registry_response_without_affinity() -> None:
    async def check() -> None:
        response = _Response(
            200,
            [b"manifest-", b"bytes"],
            {
                "Content-Type": "application/vnd.oci.image.manifest.v1+json",
                "Docker-Content-Digest": "sha256:abc",
                "Transfer-Encoding": "chunked",
            },
        )
        session = _Session([response])
        registry = _Registry([("http://mirror-a:5000", 1.0)])
        proxy = DockerMirrorProxy(registry, _Sessions(session))

        result = await proxy.forward(_request(query=b"ns=test"))
        body = b"".join([chunk async for chunk in result.body_iterator])

        assert body == b"manifest-bytes"
        assert result.status_code == 200
        assert result.headers["docker-content-digest"] == "sha256:abc"
        assert "transfer-encoding" not in result.headers
        assert session.calls[0][0:2] == (
            "GET",
            "http://mirror-a:5000/v2/library/alpine/manifests/3.20?ns=test",
        )
        assert session.calls[0][2]["auto_decompress"] is False
        assert response.released is True
        assert registry.reports[-1][3] is True

    asyncio.run(check())


def test_proxy_retries_before_streaming_and_rewrites_location() -> None:
    async def check() -> None:
        response = _Response(
            307,
            [],
            {"Location": "http://mirror-b:5000/v2/blobs/sha256:abc"},
        )
        session = _Session([aiohttp.ClientConnectionError("down"), response])
        registry = _Registry(
            [("http://mirror-a:5000", 0.5), ("http://mirror-b:5000", 0.5)]
        )
        proxy = DockerMirrorProxy(registry, _Sessions(session))

        result = await proxy.forward(_request(method="HEAD"))

        assert result.status_code == 307
        assert result.headers["location"] == (
            "http://gateway.example:8080/v2/blobs/sha256:abc"
        )
        assert response.released is True
        assert [report[3] for report in registry.reports] == [False, True]

    asyncio.run(check())


class _AffinityRegistry:
    def __init__(self, records) -> None:
        self.records = list(records)
        self.model_forces = []
        self.reports = []

    async def models(self, force=False):
        self.model_forces.append(force)
        return {"docker-mirror": list(self.records)}

    async def sample_servers(self, service, n, force=False):
        assert service == "docker-mirror"
        return [
            (record["uri"], 1.0 / len(self.records))
            for record in self.records[:n]
        ]

    def report_latency(self, server, latency, prob, success):
        self.reports.append((server, latency, prob, success))


def _mirror_record(server_id: str, uri: str, status: str = "active"):
    return {
        "server_id": server_id,
        "uri": uri,
        "status": status,
        "metadata": {"model_path": "docker-mirror"},
    }


def test_image_affinity_id_is_inferred_from_exact_v2_object() -> None:
    assert DockerMirrorProxy._image_affinity_id(
        _request(
            path="/v2/org/team/image/manifests/latest",
            query=b"ns=registry-1.docker.io",
        )
    ) == "org/team/image/manifests/latest"
    assert DockerMirrorProxy._image_affinity_id(
        _request(path="/v2/org/team/image/blobs/sha256:abc")
    ) == "org/team/image/blobs/sha256:abc"
    assert DockerMirrorProxy._image_affinity_id(_request(path="/v2/")) is None


def test_object_affinity_reuses_live_mirror_without_handshake() -> None:
    async def check(root: str) -> None:
        records = [
            _mirror_record("server-a", "http://mirror-a:5000"),
            _mirror_record("server-b", "http://mirror-b:5000"),
        ]
        registry = _AffinityRegistry(records)
        bindings = SoftAffinityBindingStore(
            FileSystemKVStore(root),
            default_ttl_seconds=900,
        )
        session = _Session([_Response(200, [b"manifest"]), _Response(200, [b"blob"])])
        proxy = DockerMirrorProxy(
            registry,
            _Sessions(session),
            bindings=bindings,
        )

        first = await proxy.forward(
            _request(
                path="/v2/library/alpine/manifests/3.20",
                query=b"ns=registry-1.docker.io",
            )
        )
        assert b"".join([chunk async for chunk in first.body_iterator]) == b"manifest"

        registry.records.reverse()
        second = await proxy.forward(
            _request(
                path="/v2/library/alpine/manifests/3.20",
                query=b"ns=registry-1.docker.io",
            )
        )
        assert b"".join([chunk async for chunk in second.body_iterator]) == b"blob"

        assert [call[1].split("/v2/")[0] for call in session.calls] == [
            "http://mirror-a:5000",
            "http://mirror-a:5000",
        ]
        binding = await bindings.resolve(
            "docker-mirror:image",
            "library/alpine/manifests/3.20",
        )
        assert binding is not None
        assert binding.server_id == "server-a"
        assert binding.handoff_count == 0
        assert registry.model_forces == [False, False]

    with tempfile.TemporaryDirectory() as root:
        asyncio.run(check(root))


def test_distinct_registry_objects_can_spread_across_mirrors() -> None:
    async def check(root: str) -> None:
        records = [
            _mirror_record("server-a", "http://mirror-a:5000"),
            _mirror_record("server-b", "http://mirror-b:5000"),
        ]
        registry = _AffinityRegistry(records)
        bindings = SoftAffinityBindingStore(FileSystemKVStore(root))
        session = _Session([_Response(200, [b"one"]), _Response(200, [b"two"])])
        proxy = DockerMirrorProxy(registry, _Sessions(session), bindings=bindings)

        first = await proxy.forward(
            _request(path="/v2/org/image/manifests/tag-one")
        )
        assert b"".join([chunk async for chunk in first.body_iterator]) == b"one"
        registry.records.reverse()
        second = await proxy.forward(
            _request(path="/v2/org/image/manifests/tag-two")
        )
        assert b"".join([chunk async for chunk in second.body_iterator]) == b"two"

        assert [call[1].split("/v2/")[0] for call in session.calls] == [
            "http://mirror-a:5000",
            "http://mirror-b:5000",
        ]

    with tempfile.TemporaryDirectory() as root:
        asyncio.run(check(root))



def test_soft_affinity_rebinds_before_routing_when_owner_is_not_live() -> None:
    async def check(root: str) -> None:
        registry = _AffinityRegistry(
            [_mirror_record("server-b", "http://mirror-b:5000")]
        )
        bindings = SoftAffinityBindingStore(
            FileSystemKVStore(root),
            default_ttl_seconds=900,
        )
        await bindings.bind(
            "docker-mirror:image",
            "library/alpine/manifests/3.20",
            "server-a",
            "http://mirror-a:5000",
        )
        session = _Session([_Response(200, [b"manifest"])])
        proxy = DockerMirrorProxy(
            registry,
            _Sessions(session),
            bindings=bindings,
        )

        result = await proxy.forward(
            _request(path="/v2/library/alpine/manifests/3.20")
        )
        assert b"".join([chunk async for chunk in result.body_iterator]) == b"manifest"

        assert len(session.calls) == 1
        assert session.calls[0][1].startswith("http://mirror-b:5000/")
        binding = await bindings.resolve(
            "docker-mirror:image",
            "library/alpine/manifests/3.20",
        )
        assert binding is not None
        assert binding.server_id == "server-b"
        assert binding.previous_server_id == "server-a"
        assert binding.handoff_count == 1
        assert registry.model_forces == [False]

    with tempfile.TemporaryDirectory() as root:
        asyncio.run(check(root))


def test_soft_affinity_hands_off_when_live_owner_request_fails() -> None:
    async def check(root: str) -> None:
        records = [
            _mirror_record("server-a", "http://mirror-a:5000"),
            _mirror_record("server-b", "http://mirror-b:5000"),
        ]
        registry = _AffinityRegistry(records)
        bindings = SoftAffinityBindingStore(FileSystemKVStore(root))
        await bindings.bind(
            "docker-mirror:image",
            "library/alpine/manifests/3.20",
            "server-a",
            "http://mirror-a:5000",
        )
        session = _Session(
            [aiohttp.ClientConnectionError("owner stopped"), _Response(200, [b"ok"])]
        )
        proxy = DockerMirrorProxy(
            registry,
            _Sessions(session),
            bindings=bindings,
        )

        result = await proxy.forward(
            _request(path="/v2/library/alpine/manifests/3.20")
        )
        assert b"".join([chunk async for chunk in result.body_iterator]) == b"ok"
        assert [call[1].split("/v2/")[0] for call in session.calls] == [
            "http://mirror-a:5000",
            "http://mirror-b:5000",
        ]
        binding = await bindings.resolve(
            "docker-mirror:image",
            "library/alpine/manifests/3.20",
        )
        assert binding is not None
        assert binding.server_id == "server-b"
        assert binding.previous_server_id == "server-a"
        assert binding.handoff_count == 1

    with tempfile.TemporaryDirectory() as root:
        asyncio.run(check(root))


def test_proxy_returns_final_upstream_5xx_when_no_replacement_exists() -> None:
    async def check() -> None:
        response = _Response(503, [b"upstream unavailable"])
        session = _Session([response])
        registry = _Registry([("http://mirror-a:5000", 1.0)])
        proxy = DockerMirrorProxy(registry, _Sessions(session))

        result = await proxy.forward(_request())
        body = b"".join([chunk async for chunk in result.body_iterator])

        assert result.status_code == 503
        assert body == b"upstream unavailable"
        assert len(session.calls) == 1

    asyncio.run(check())
