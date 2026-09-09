from __future__ import annotations

import asyncio
from typing import Any

import pytest

from literegistry_tool_client import JudgeClient


class RecordingJudgeClient(JudgeClient):
    async def _post_with_retries_async(
        self,
        payload: dict[str, Any],
        *,
        server_url: str | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        self.payload = payload
        return {"judgments": [{"label": "pass"}]}


def test_lightweight_judge_client_keeps_model_and_service_distinct() -> None:
    client = RecordingJudgeClient(
        "http://gateway:1212/judge",
        service_model_path="judge-service",
    )
    result = asyncio.run(
        client.verify_output(
            input="question",
            output="answer",
            rubrics=["contains answer"],
            model="policy-checkpoint",
        )
    )

    assert result["judgments"][0]["label"] == "pass"
    assert client.payload == {
        "input": "question",
        "output": "answer",
        "rubrics": ["contains answer"],
        "model": "policy-checkpoint",
        "model_path": "judge-service",
    }


@pytest.mark.parametrize("rubrics", [[], [""], "criterion"])
def test_lightweight_judge_client_rejects_invalid_rubrics(rubrics: object) -> None:
    client = RecordingJudgeClient()
    with pytest.raises((TypeError, ValueError)):
        asyncio.run(
            client.execute(
                input="question",
                output="answer",
                rubrics=rubrics,  # type: ignore[arg-type]
                model="model",
            )
        )
