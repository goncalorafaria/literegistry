"""Client for extracting readable content from web pages."""

from __future__ import annotations

import asyncio
import json
import os
import urllib.parse
from collections.abc import Mapping
from typing import Any

import aiohttp

from ._transport import (
    HTTP_RETRY_BASE_DELAY_SECONDS,
    HTTP_RETRY_MAX_DELAY_SECONDS,
    RETRYABLE_HTTP_STATUSES,
    ToolClient,
)


DEFAULT_FETCH_SERVER_URL = "https://r.jina.ai"


class FetchClient(ToolClient):
    """Fetch a URL through Jina Reader or a LiteRegistry URL-fetch service."""

    def __init__(
        self,
        server_url: str = DEFAULT_FETCH_SERVER_URL,
        *,
        api_key: str | None = None,
        timeout: float = 65,
        max_retries: int = 3,
        max_content_length: int | None = None,
        local_search_server_url: str | None = None,
        local_search_model_path: str | None = None,
    ) -> None:
        super().__init__(server_url, timeout=timeout, max_retries=max_retries)
        if local_search_model_path and not local_search_server_url:
            raise ValueError("local_search_model_path requires local_search_server_url")
        self.api_key = api_key
        self.max_content_length = max_content_length
        self.local_search_server_url = local_search_server_url
        self.local_search_model_path = local_search_model_path

    @property
    def request_name(self) -> str:
        if self.local_search_server_url:
            service = self.local_search_model_path or "gateway default"
            return f"Gateway URL fetch request ({service})"
        return "Jina Reader request"

    async def execute(
        self,
        *,
        url: str,
        parameters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        url = str(url or "").strip()
        if not url:
            raise ValueError("url is required")
        if self.local_search_server_url:
            body = {"mode": "url", "url": url}
            if self.local_search_model_path:
                body["model_path"] = self.local_search_model_path
            return await self._post_with_retries_async(
                body,
                server_url=self.local_search_server_url,
            )

        api_key = self.api_key or os.getenv("JINA_API_KEY")
        if not api_key:
            raise RuntimeError("JINA_API_KEY is required for Jina Reader requests")
        encoded_url = urllib.parse.quote(url, safe=":/")
        request_url = f"{self.server_url}/{encoded_url}"
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {api_key}",
            "X-Return-Format": "markdown",
        }
        for key, value in (parameters or {}).items():
            if value is not None:
                headers[str(key)] = str(value)

        delay = HTTP_RETRY_BASE_DELAY_SECONDS
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        transient_network_errors = (
            aiohttp.ClientConnectionError,
            aiohttp.ServerDisconnectedError,
            asyncio.TimeoutError,
        )
        for attempt in range(self.max_retries):
            try:
                async with aiohttp.ClientSession(
                    timeout=timeout, trust_env=False
                ) as session:
                    async with session.get(request_url, headers=headers) as response:
                        response_text = await response.text()
                        if (
                            response.status in RETRYABLE_HTTP_STATUSES
                            and attempt < self.max_retries - 1
                        ):
                            await asyncio.sleep(delay)
                            delay = min(delay * 2.0, HTTP_RETRY_MAX_DELAY_SECONDS)
                            continue
                        if response.status >= 400:
                            raise RuntimeError(
                                f"{self.request_name} failed with HTTP "
                                f"{response.status} at {request_url}: "
                                f"{response_text[:1000]}"
                            )
                        try:
                            payload = json.loads(response_text)
                        except json.JSONDecodeError as exc:
                            raise RuntimeError(
                                f"{self.request_name} response was not valid JSON"
                            ) from exc
                        if not isinstance(payload, dict):
                            raise RuntimeError(
                                f"{self.request_name} response was not a JSON object"
                            )
                        code = payload.get("code", 200)
                        if code != 200:
                            return {
                                "success": False,
                                "mode": "url",
                                "error": str(
                                    payload.get("message") or f"Jina API code {code}"
                                ),
                            }
                        data = payload.get("data") or {}
                        if not isinstance(data, Mapping):
                            raise RuntimeError(
                                f"{self.request_name} data was not a JSON object"
                            )
                        title = str(data.get("title") or "").strip()
                        content = str(data.get("content") or data.get("text") or "")
                        if title and content:
                            content = f"# {title}\n\n{content}"
                        if (
                            self.max_content_length is not None
                            and len(content) > self.max_content_length
                        ):
                            content = content[: self.max_content_length]
                        return {
                            "success": True,
                            "mode": "url",
                            "data": {"title": title, "content": content, "url": url},
                        }
            except transient_network_errors as exc:
                if attempt == self.max_retries - 1:
                    raise RuntimeError(
                        f"{self.request_name} failed after {self.max_retries} "
                        f"attempts at {request_url}: {exc}"
                    ) from exc
                await asyncio.sleep(delay)
                delay = min(delay * 2.0, HTTP_RETRY_MAX_DELAY_SECONDS)
        raise RuntimeError("unreachable")

    def __str__(self) -> str:
        return f"FetchClient(server_url={self.server_url})"


__all__ = ["DEFAULT_FETCH_SERVER_URL", "FetchClient"]
