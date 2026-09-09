"""Client for restricted terminal execution."""

from __future__ import annotations

from typing import Any

from ._transport import ToolClient


DEFAULT_TERMINAL_SERVER_URL = "http://127.0.0.1:1212/terminal"


class TerminalExecutionClient(ToolClient):
    """Execute a terminal pipeline through a LiteRegistry tool endpoint."""

    def __init__(
        self,
        server_url: str = DEFAULT_TERMINAL_SERVER_URL,
        *,
        timeout: float = 20,
        max_retries: int = 3,
        truncation: int | None = None,
    ) -> None:
        super().__init__(server_url, timeout=timeout, max_retries=max_retries)
        self.truncation = truncation

    @property
    def request_name(self) -> str:
        return "Terminal execution"

    def _prepare_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        body = dict(payload)
        body.setdefault("truncation", self.truncation)
        return body

    async def execute(
        self,
        *,
        command: str,
        contents: Any,
        truncation: int | None = None,
    ) -> dict[str, Any]:
        return await self._post_with_retries_async(
            {
                "contents": "" if contents is None else str(contents),
                "command": command,
                "truncation": self.truncation if truncation is None else truncation,
            }
        )

__all__ = ["DEFAULT_TERMINAL_SERVER_URL", "TerminalExecutionClient"]
