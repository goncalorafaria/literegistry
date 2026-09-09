"""Rollout-local client for articulated rubric submissions."""

from __future__ import annotations

import copy
import json
from collections.abc import Iterable
from typing import Any

from ._transport import ToolClient


SUBMIT_TOOL_SPEC: dict[str, Any] = {
    "name": "submit",
    "description": (
        "Record the judgment for one rubric. The tool response reports which "
        "rubric IDs are complete and which are still waiting."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "rubric_id": {
                "type": ["string", "integer"],
                "description": "The stable ID of the rubric being judged.",
            },
            "feedback": {
                "type": "string",
                "description": "A concise justification for the judgment.",
            },
            "label": {
                "type": "string",
                "description": "The label assigned under this rubric.",
            },
        },
        "required": ["rubric_id", "feedback", "label"],
        "additionalProperties": False,
    },
}


def _rubric_key(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise TypeError("rubric_id must be a string or integer")
    key = str(value).strip()
    if not key:
        raise ValueError("rubric_id must be non-empty")
    return key


def normalize_expected_rubric_ids(values: Iterable[Any]) -> tuple[str, ...]:
    normalized = tuple(_rubric_key(value) for value in values)
    if len(set(normalized)) != len(normalized):
        raise ValueError("expected_rubric_ids must not contain duplicates")
    return normalized


class SubmissionStore:
    """Own expected IDs and accumulated judgments for one rollout."""

    def __init__(self, expected_rubric_ids: Iterable[str | int]) -> None:
        self._expected_rubric_ids = normalize_expected_rubric_ids(
            expected_rubric_ids
        )
        self._submissions: dict[str, dict[str, str]] = {}

    @property
    def expected_rubric_ids(self) -> tuple[str, ...]:
        return self._expected_rubric_ids

    @property
    def completed_rubric_ids(self) -> tuple[str, ...]:
        return tuple(
            key for key in self._expected_rubric_ids if key in self._submissions
        )

    @property
    def waiting_rubric_ids(self) -> tuple[str, ...]:
        return tuple(
            key for key in self._expected_rubric_ids if key not in self._submissions
        )

    @property
    def all_completed(self) -> bool:
        return bool(self._expected_rubric_ids) and not self.waiting_rubric_ids

    def submit(
        self, *, rubric_id: str | int, feedback: str, label: str
    ) -> dict[str, Any]:
        key = _rubric_key(rubric_id)
        if key not in self._expected_rubric_ids:
            raise ValueError(
                f"unknown rubric_id {key!r}; expected one of "
                f"{list(self._expected_rubric_ids)!r}"
            )
        normalized_feedback = str(feedback or "").strip()
        normalized_label = str(label or "").strip()
        if not normalized_feedback:
            raise ValueError("feedback must be non-empty")
        if not normalized_label:
            raise ValueError("label must be non-empty")
        replaced = key in self._submissions
        self._submissions[key] = {
            "rubric_id": key,
            "feedback": normalized_feedback,
            "label": normalized_label,
        }
        return {
            "accepted_rubric_id": key,
            "replaced": replaced,
            "completed_rubric_ids": list(self.completed_rubric_ids),
            "waiting_rubric_ids": list(self.waiting_rubric_ids),
            "all_completed": self.all_completed,
        }

    def get(self, rubric_id: str | int) -> dict[str, str]:
        return copy.deepcopy(self._submissions[_rubric_key(rubric_id)])

    def as_dict(self) -> dict[str, dict[str, str]]:
        return copy.deepcopy(self._submissions)

    def __contains__(self, rubric_id: object) -> bool:
        try:
            key = _rubric_key(rubric_id)
        except (TypeError, ValueError):
            return False
        return key in self._submissions

    def __len__(self) -> int:
        return len(self._submissions)


class SubmitToolClient(ToolClient):
    """Local submit tool backed by a rollout-owned submission store."""

    def __init__(self) -> None:
        super().__init__("memory://submit", timeout=1, max_retries=1)

    @property
    def request_name(self) -> str:
        return "rubric submission"

    @staticmethod
    def new_session(
        expected_rubric_ids: Iterable[str | int],
    ) -> SubmissionStore:
        return SubmissionStore(expected_rubric_ids)

    async def execute(
        self,
        *,
        rubric_id: str | int,
        feedback: str,
        label: str,
        submission_store: SubmissionStore,
    ) -> dict[str, Any]:
        if not isinstance(submission_store, SubmissionStore):
            raise TypeError("submission_store must be a SubmissionStore")
        status = submission_store.submit(
            rubric_id=rubric_id,
            feedback=feedback,
            label=label,
        )
        return {
            "success": True,
            "stdout": json.dumps(status, ensure_ascii=False),
            "stderr": "",
            "exit_code": 0,
            "data": status,
        }


__all__ = [
    "SUBMIT_TOOL_SPEC",
    "SubmissionStore",
    "SubmitToolClient",
    "normalize_expected_rubric_ids",
]
