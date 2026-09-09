"""Client for web search through LiteRegistry."""

from __future__ import annotations

from typing import Any

from ._transport import ToolClient


DEFAULT_SEARCH_SERVER_URL = "http://127.0.0.1:1212/search"


class SearchClient(ToolClient):
    """Search for links matching a query."""

    def __init__(
        self,
        server_url: str = DEFAULT_SEARCH_SERVER_URL,
        *,
        model_path: str | None = None,
        timeout: float = 65,
        max_retries: int = 3,
    ) -> None:
        super().__init__(server_url, timeout=timeout, max_retries=max_retries)
        self.model_path = model_path

    @property
    def request_name(self) -> str:
        return "Search request"

    async def execute(
        self,
        *,
        query: str,
        num_results: int = 10,
        parameters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not query:
            raise ValueError("query is required")
        payload: dict[str, Any] = {
            "mode": "query",
            "query": query,
            "num_results": num_results,
            "parameters": parameters or {},
        }
        if self.model_path:
            payload["model_path"] = self.model_path
        return await self._post_with_retries_async(payload)

    def __str__(self) -> str:
        return f"SearchClient(server_url={self.server_url})"


__all__ = ["DEFAULT_SEARCH_SERVER_URL", "SearchClient"]
