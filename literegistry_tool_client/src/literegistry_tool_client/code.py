"""Client for remote Python code execution."""

from __future__ import annotations

from typing import Any

from ._transport import ToolClient


DEFAULT_CODE_SERVER_URL = "http://127.0.0.1:1212/python"


class RemoteCodeExecutionClient(ToolClient):
    """Execute Python through a LiteRegistry tool endpoint."""

    def __init__(
        self,
        server_url: str = DEFAULT_CODE_SERVER_URL,
        *,
        timeout: float = 20,
        max_retries: int = 3,
        max_runtime: int = 2,
    ) -> None:
        super().__init__(server_url, timeout=timeout, max_retries=max_retries)
        self.max_runtime = max_runtime

    @property
    def request_name(self) -> str:
        return "Code execution"

    def _prepare_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {**payload, "model": "python"}

    async def execute(
        self,
        *,
        code: str,
        context_payload: Any = None,
        max_runtime: int | None = None,
    ) -> dict[str, Any]:
        return await self._post_with_retries_async(
            {
                "code": code,
                "context_payload": context_payload,
                "max_runtime": self.max_runtime if max_runtime is None else max_runtime,
            }
        )

    def __str__(self) -> str:
        return f"RemoteCodeExecutionClient(server_url={self.server_url})"


__all__ = ["DEFAULT_CODE_SERVER_URL", "RemoteCodeExecutionClient"]
