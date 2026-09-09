"""Client for the rubric-judge service."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import aiohttp

from ._transport import ToolClient


DEFAULT_JUDGE_SERVER_URL = "http://127.0.0.1:1212/judge"


class JudgeClient(ToolClient):
    """Verify candidate outputs through a LiteRegistry `/judge` endpoint.

    ``model`` selects the registered LLM that performs the judgment.
    ``service_model_path`` independently selects the judge service pool,
    normally ``"judge"``.
    """

    def __init__(
        self,
        server_url: str = DEFAULT_JUDGE_SERVER_URL,
        *,
        service_model_path: str = "judge",
        timeout: float = 240,
        max_retries: int = 3,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        super().__init__(
            server_url,
            timeout=timeout,
            max_retries=max_retries,
            http_session=http_session,
        )
        if not isinstance(service_model_path, str) or not service_model_path.strip():
            raise ValueError("service_model_path must be non-empty")
        self.service_model_path = service_model_path.strip()

    @property
    def request_name(self) -> str:
        return "Judge request"

    async def execute(
        self,
        *,
        input: str,
        output: str,
        rubrics: Sequence[str],
        model: str,
        service_model_path: str | None = None,
    ) -> dict[str, Any]:
        """Judge one candidate output against one or more textual rubrics."""
        if not isinstance(input, str):
            raise TypeError("input must be a string")
        if not isinstance(output, str):
            raise TypeError("output must be a string")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a non-empty string")
        if isinstance(rubrics, (str, bytes)) or not isinstance(rubrics, Sequence):
            raise TypeError("rubrics must be a sequence of non-empty strings")

        normalized_rubrics = list(rubrics)
        if not normalized_rubrics or any(
            not isinstance(rubric, str) or not rubric.strip()
            for rubric in normalized_rubrics
        ):
            raise ValueError("rubrics must contain at least one non-empty string")

        selected_service = service_model_path or self.service_model_path
        if not isinstance(selected_service, str) or not selected_service.strip():
            raise ValueError("service_model_path must be non-empty")

        return await self._post_with_retries_async(
            {
                "input": input,
                "output": output,
                "rubrics": normalized_rubrics,
                "model": model.strip(),
                "model_path": selected_service.strip(),
            }
        )

    async def verify_output(
        self,
        *,
        input: str,
        output: str,
        rubrics: Sequence[str],
        model: str,
        service_model_path: str | None = None,
    ) -> dict[str, Any]:
        """Descriptive alias for :meth:`execute`."""
        return await self.execute(
            input=input,
            output=output,
            rubrics=rubrics,
            model=model,
            service_model_path=service_model_path,
        )

    def __str__(self) -> str:
        return (
            "JudgeClient("
            f"server_url={self.server_url}, "
            f"service_model_path={self.service_model_path})"
        )
