"""Client for sequence-classification reward models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ._transport import ToolClient


DEFAULT_CLASSIFY_SERVER_URL = "http://127.0.0.1:1212/classify"


class RewardModelClient(ToolClient):
    """Classify complete OpenAI-style conversations through LiteRegistry."""

    def __init__(
        self,
        model_path: str,
        server_url: str = DEFAULT_CLASSIFY_SERVER_URL,
        *,
        timeout: float = 240,
        max_retries: int = 3,
    ) -> None:
        super().__init__(server_url, timeout=timeout, max_retries=max_retries)
        if not isinstance(model_path, str) or not model_path.strip():
            raise ValueError("model_path must be a non-empty string")
        self.model_path = model_path.strip()

    @property
    def request_name(self) -> str:
        return "Reward model classification"

    async def execute(
        self,
        *,
        messages: Sequence[Mapping[str, Any]],
        model_path: str | None = None,
        chat_template_kwargs: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if isinstance(messages, (str, bytes)) or not isinstance(messages, Sequence):
            raise TypeError("messages must be a sequence of OpenAI message objects")
        normalized_messages: list[dict[str, Any]] = []
        for index, message in enumerate(messages):
            if not isinstance(message, Mapping):
                raise TypeError(f"messages[{index}] must be an object")
            role = message.get("role")
            if not isinstance(role, str) or not role.strip():
                raise ValueError(f"messages[{index}].role must be non-empty")
            if "content" not in message:
                raise ValueError(f"messages[{index}].content is required")
            normalized_messages.append({**dict(message), "role": role.strip()})
        if not normalized_messages:
            raise ValueError("messages must not be empty")

        selected_model = self.model_path if model_path is None else model_path
        if not isinstance(selected_model, str) or not selected_model.strip():
            raise ValueError("model_path must be a non-empty string")
        payload: dict[str, Any] = {
            "model": selected_model.strip(),
            "messages": normalized_messages,
            "add_generation_prompt": False,
        }
        if chat_template_kwargs is not None:
            if not isinstance(chat_template_kwargs, Mapping):
                raise TypeError("chat_template_kwargs must be an object")
            payload["chat_template_kwargs"] = dict(chat_template_kwargs)
        return await self._post_with_retries_async(payload)

    async def classify(
        self,
        *,
        messages: Sequence[Mapping[str, Any]],
        model_path: str | None = None,
        chat_template_kwargs: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        return await self.execute(
            messages=messages,
            model_path=model_path,
            chat_template_kwargs=chat_template_kwargs,
        )

    def __str__(self) -> str:
        return (
            "RewardModelClient("
            f"server_url={self.server_url}, model_path={self.model_path})"
        )


__all__ = ["DEFAULT_CLASSIFY_SERVER_URL", "RewardModelClient"]
