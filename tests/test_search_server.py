import asyncio
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from literegistry.cache_server import CacheResponse
from literegistry.services.search_server import (
    SearchRequest,
    SearchServer,
    SearchServerConfig,
)


class FakeRegistry:
    async def register_server(self, *args, **kwargs):
        return None

    async def heartbeat(self, *args, **kwargs):
        return None

    async def deregister(self):
        return None


class FakeCacheService:
    def __init__(self):
        self.values = {}
        self.started = False
        self.closed = False
        self.set_calls = []

    async def start(self):
        self.started = True

    async def get(self, key):
        if key not in self.values:
            return CacheResponse(hit=False)
        return CacheResponse(hit=True, value=self.values[key])

    async def set(self, key, value, ttl_seconds):
        self.values[key] = value
        self.set_calls.append((key, value, ttl_seconds))
        return CacheResponse(hit=True)

    async def close(self):
        self.closed = True


class FakeResponse:
    def __init__(self, body='{"results": [{"title": "result"}]}', status=200):
        self.body = body
        self.status = status

    async def text(self):
        return self.body

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None


class FakeSession:
    def __init__(self, response=None):
        self.calls = []
        self.response = response or FakeResponse()

    def post(self, endpoint, **kwargs):
        self.calls.append((endpoint, kwargs))
        return self.response


def make_server():
    config = SearchServerConfig(
        registry="redis://controller.example:6379/0",
        provider="generic",
        search_api_url="http://search.example/api",
        fetch_api_url="http://fetch.example/api",
        jina_api_key="jina-test-key",
        cache_service="cache",
    )
    fake_cache = FakeCacheService()
    with (
        patch("literegistry.services.search_server.get_kvstore"),
        patch(
            "literegistry.services.search_server.RegistryClient",
            return_value=FakeRegistry(),
        ),
        patch(
            "literegistry.services.search_server.CacheServiceClient",
            return_value=fake_cache,
        ) as cache_client,
    ):
        server = SearchServer(config)
    return server, fake_cache, cache_client


def test_search_discovers_cache_service_through_registry():
    server, _, cache_client = make_server()

    cache_client.assert_called_once_with(
        registry=server.registry,
        service_name="cache",
        timeout=5,
        max_retries=3,
    )


def test_request_requires_field_for_selected_mode():
    with pytest.raises(ValidationError):
        SearchRequest(mode="query")
    with pytest.raises(ValidationError):
        SearchRequest(mode="url", url="file:///tmp/private")


def test_query_requests_are_cached_through_service():
    server, cache, _ = make_server()
    server.session = FakeSession()
    request = SearchRequest(mode="query", query="distributed inference", num_results=3)

    first = asyncio.run(server.execute(request))
    second = asyncio.run(server.execute(request))

    assert first.success is True
    assert first.cache_hit is False
    assert second.success is True
    assert second.cache_hit is True
    assert len(server.session.calls) == 1
    assert len(cache.set_calls) == 1
    assert cache.set_calls[0][2] == 3600
    endpoint, kwargs = server.session.calls[0]
    assert endpoint == "http://search.example/api"
    assert kwargs["json"] == {
        "query": "distributed inference",
        "num_results": 3,
    }


def test_cache_service_failure_is_fail_open():
    server, cache, _ = make_server()
    server.session = FakeSession()

    async def unavailable(_key):
        raise RuntimeError("no cache service")

    cache.get = unavailable
    response = asyncio.run(
        server.execute(SearchRequest(mode="query", query="fresh result"))
    )

    assert response.success is True
    assert response.cache_hit is False
    assert len(server.session.calls) == 1


def test_url_mode_posts_url_to_jina_and_normalizes_response():
    server, _, _ = make_server()
    server.session = FakeSession(
        FakeResponse('{"data": {"title": "Example", "content": "Page text"}}')
    )

    response = asyncio.run(
        server.execute(SearchRequest(mode="url", url="https://example.com/page"))
    )

    assert response.success is True
    endpoint, kwargs = server.session.calls[0]
    assert endpoint == "http://fetch.example/api"
    assert kwargs["json"] == {"url": "https://example.com/page"}
    assert kwargs["headers"] == {
        "Accept": "application/json",
        "Authorization": "Bearer jina-test-key",
        "X-Return-Format": "markdown",
    }
    assert response.data == {
        "title": "Example",
        "content": "Page text",
        "url": "https://example.com/page",
    }


def test_serper_mode_uses_serper_request_fields():
    server, _, _ = make_server()
    server.config.provider = "serper"

    payload, endpoint = server._payload_and_endpoint(
        SearchRequest(
            mode="query",
            query="distributed inference",
            num_results=3,
            parameters={"gl": "us", "hl": "en"},
        )
    )

    assert endpoint == "http://search.example/api"
    assert payload == {
        "q": "distributed inference",
        "num": 3,
        "gl": "us",
        "hl": "en",
    }


def test_registry_metadata_exposes_cache_service_dependency():
    server, _, _ = make_server()
    metadata = server._metadata()
    assert metadata["model_path"] == "search"
    assert metadata["extra_kwargs"]["modes"] == ["query", "url"]
    assert metadata["extra_kwargs"]["cache_service"] == "cache"
    assert metadata["extra_kwargs"]["cache_ttl"] == 3600
