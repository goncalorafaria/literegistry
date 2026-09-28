"""Participant client for strict-affinity Botbook messaging sessions."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Literal

import aiohttp

from ._transport import ToolClient


DEFAULT_BOTBOOK_GATEWAY_URL = "http://127.0.0.1:1212"


def _identifier(value: object, name: str) -> str:
    if not isinstance(value, str) or not 1 <= len(value) <= 256:
        raise ValueError(f"{name} must be a string of 1–256 characters")
    return value


class BotbookClient(ToolClient):
    """Create or join one session as a participant identified by ``user_id``.

    An attached client never closes the shared session on context-manager exit.
    Explicit ``close()`` always deletes the session for all participants.
    Requests use one attempt: posts and mark-as-read operations cannot safely be
    replayed when the response is lost. Injected HTTP sessions remain caller-owned.
    """

    def __init__(
        self,
        gateway_url: str = DEFAULT_BOTBOOK_GATEWAY_URL,
        *,
        user_id: str,
        affinity_id: str | None = None,
        service: str = "botbook",
        timeout: float = 30,
        handshake_timeout: float = 300,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        if not isinstance(gateway_url, str) or not gateway_url.strip().rstrip("/"):
            raise ValueError("gateway_url must be non-empty")
        if not isinstance(service, str) or not service.strip():
            raise ValueError("service must be non-empty")
        if handshake_timeout <= 0:
            raise ValueError("handshake_timeout must be positive")
        self.user_id = _identifier(user_id, "user_id")
        self._affinity_id = (
            _identifier(affinity_id, "affinity_id") if affinity_id is not None else None
        )
        self.gateway_url = gateway_url.rstrip("/")
        self.service = service
        self.handshake_timeout = handshake_timeout
        self._owns_session = False
        self._session_lock = asyncio.Lock()
        super().__init__(
            f"{self.gateway_url}/affinity/botbook",
            timeout=timeout,
            max_retries=1,
            http_session=http_session,
        )

    @property
    def request_name(self) -> str:
        return "Botbook messaging"

    @property
    def affinity_id(self) -> str | None:
        return self._affinity_id

    @property
    def started(self) -> bool:
        return self._affinity_id is not None

    def _prepare_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {**payload, "service": self.service}

    async def start(self) -> dict[str, Any]:
        """Create a session; share its affinity_id with other participants."""
        async with self._session_lock:
            if self.started:
                raise RuntimeError("Botbook session is already started or attached")
            result = await self._post_with_retries_async(
                {}, server_url=f"{self.gateway_url}/affinity/handshake",
                timeout=self.handshake_timeout,
            )
            affinity_id = result.get("affinity_id")
            if not isinstance(affinity_id, str) or not 1 <= len(affinity_id) <= 256:
                raise RuntimeError("Botbook handshake returned no valid affinity_id")
            self._affinity_id = affinity_id
            self._owns_session = True
            return result

    async def execute(
        self,
        *,
        action: Literal["post_general", "post_direct", "get_unread"],
        text: str | None = None,
        recipient_id: str | None = None,
        limit: int = 100,
        board: str | None = None,
    ) -> dict[str, Any]:
        """Send one operation and return its full JSON response envelope."""
        if action not in {"post_general", "post_direct", "get_unread"}:
            raise ValueError("unknown Botbook action")
        payload: dict[str, Any] = {"action": action, "user_id": self.user_id}
        if action == "get_unread":
            if text is not None or recipient_id is not None:
                raise ValueError("get_unread does not accept text or recipient_id")
            if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 1000:
                raise ValueError("limit must be an integer from 1 to 1000")
            payload["limit"] = limit
            if board is not None:
                payload["board"] = _identifier(board, "board")
        else:
            if not isinstance(text, str) or not 1 <= len(text) <= 65536:
                raise ValueError("text must be a string of 1–65536 characters")
            payload["text"] = text
            if action == "post_direct":
                if board is not None:
                    raise ValueError("direct messages do not have a board")
                payload["recipient_id"] = _identifier(recipient_id, "recipient_id")
            elif recipient_id is not None:
                raise ValueError("post_general does not accept recipient_id")
            else:
                payload["board"] = _identifier("general" if board is None else board, "board")
        async with self._session_lock:
            if self._affinity_id is None:
                raise RuntimeError("Botbook session is not started; call start() or attach an affinity_id")
            return await self._post_with_retries_async(
                {**payload, "affinity_id": self._affinity_id}
            )

    async def post(self, text: str, *, board: str = "general") -> dict[str, Any]:
        """Post to a named board, visible to every participant."""
        return await self.execute(action="post_general", text=text, board=_identifier(board, "board"))

    async def dm(self, recipient_id: str, text: str) -> dict[str, Any]:
        """Post a message delivered only to recipient_id."""
        return await self.execute(action="post_direct", recipient_id=recipient_id, text=text)

    async def get_unread(self, *, limit: int = 100, board: str | None = None) -> dict[str, Any]:
        """Read general-board posts plus DMs, or only the explicitly named board."""
        return await self.execute(action="get_unread", limit=limit, board=board)

    async def close(self) -> dict[str, Any] | None:
        """Delete every message and read cursor, for all session participants."""
        async with self._session_lock:
            if self._affinity_id is None:
                return None
            result = await self._post_with_retries_async(
                {"affinity_id": self._affinity_id},
                server_url=f"{self.gateway_url}/affinity/close",
            )
            if result.get("closed") is not True:
                raise RuntimeError("Botbook close did not confirm session deletion")
            self._affinity_id = None
            self._owns_session = False
            return result

    async def __aenter__(self) -> "BotbookClient":
        if not self.started:
            await self.start()
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> bool:
        if self._owns_session:
            try:
                await self.close()
            except Exception:
                if exc is None:
                    raise
                logging.exception("Failed to close Botbook session after an exception")
        return False

    def __str__(self) -> str:
        return f"BotbookClient(gateway_url={self.gateway_url}, user_id={self.user_id!r})"


__all__ = ["DEFAULT_BOTBOOK_GATEWAY_URL", "BotbookClient"]
