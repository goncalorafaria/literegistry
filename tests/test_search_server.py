import asyncio
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from literegistry.search_server import (
    SearchRequest,
    SearchServer,
    SearchServerConfig,
    _redis_url_for_db,
)


class FakeRegistry:
    async def register_server(self, *args, **kwargs):
        return None

    async def heartbeat(self, *args, **kwargs):
        return None

    async def deregister(self):
        return None


class FakeCache:
    def __init__(self):
        self.values = {}

    async def get(self, key):
        return self.values.get(key)

    async def set(self, key, value, ex=None):
        self.values[key] = value.encode() if isinstance(value, str) else value
        return True

    async def aclose(self):
        return None


class FakeResponse:
    status = 200

    async def text(self):
        return '{"results": [{"title": "result"}]}'

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None


class FakeSession:
    def __init__(self):
        self.calls = []

    def post(self, endpoint, **kwargs):
        self.calls.append((endpoint, kwargs))
        return FakeResponse()


def make_server():
    config = SearchServerConfig(
        registry="redis://localhost:6379/0",
        provider="generic",
        search_api_url="http://search.example/api",
        fetch_api_url="http://fetch.example/api",
    )
    with (
        patch("literegistry.search_server.get_kvstore"),
        patch("literegistry.search_server.ServerRegistry", return_value=FakeRegistry()),
        patch("literegistry.search_server.redis.from_url", return_value=FakeCache()),
    ):
        return SearchServer(config)


def test_cache_uses_separate_redis_database():
    assert _redis_url_for_db("redis://localhost:6379/0", 2) == "redis://localhost:6379/2"
    assert (
        _redis_url_for_db("redis://localhost:6379?db=0&ssl=true", 1)
        == "redis://localhost:6379/1?ssl=true"
    )


def test_request_requires_field_for_selected_mode():
    with pytest.raises(ValidationError):
        SearchRequest(mode="query")
    with pytest.raises(ValidationError):
        SearchRequest(mode="url", url="file:///tmp/private")


def test_query_requests_are_cached():
    server = make_server()
    server.session = FakeSession()
    request = SearchRequest(mode="query", query="distributed inference", num_results=3)

    first = asyncio.run(server.execute(request))
    second = asyncio.run(server.execute(request))

    assert first.success is True
    assert first.cache_hit is False
    assert second.success is True
    assert second.cache_hit is True
    assert len(server.session.calls) == 1
    endpoint, kwargs = server.session.calls[0]
    assert endpoint == "http://search.example/api"
    assert kwargs["json"] == {
        "query": "distributed inference",
        "num_results": 3,
    }


def test_url_mode_posts_url_to_fetch_endpoint():
    server = make_server()
    server.session = FakeSession()

    response = asyncio.run(
        server.execute(SearchRequest(mode="url", url="https://example.com/page"))
    )

    assert response.success is True
    endpoint, kwargs = server.session.calls[0]
    assert endpoint == "http://fetch.example/api"
    assert kwargs["json"] == {"url": "https://example.com/page"}


def test_serper_mode_uses_serper_request_fields():
    server = make_server()
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


def test_registry_metadata_exposes_search_modes():
    metadata = make_server()._metadata()
    assert metadata["model_path"] == "search"
    assert metadata["extra_kwargs"]["modes"] == ["query", "url"]
    assert metadata["extra_kwargs"]["cache_db"] == 1
