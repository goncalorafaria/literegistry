"""Stable model-facing rendering for tool outputs."""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any
from uuid import uuid4


_ABSOLUTE_PATH_RE = re.compile(r"(?<![\w:/])(?:/[\w.+\-]+){2,}/?")


def truncate_output_text(value: Any, limit: int | None = None) -> str | None:
    """Truncate model-facing stdout/stderr text to ``limit`` characters."""
    if value is None:
        return None
    text = str(value)
    if limit is None or len(text) <= limit:
        return text
    if limit <= 0:
        return ""
    omitted_characters = len(text) - limit
    marker = (
        "\n[The rest of the output was truncated; "
        f"{omitted_characters} characters omitted]"
    )
    return text[: max(limit - len(marker), 0)] + marker


def truncate_tool_output(
    tool_output: dict[str, Any],
    limit: int | None = None,
) -> dict[str, Any]:
    """Return a copy with stdout and stderr truncated for model display."""
    if limit is None:
        return tool_output
    return {
        **tool_output,
        "stdout": truncate_output_text(tool_output.get("stdout"), limit),
        "stderr": truncate_output_text(tool_output.get("stderr"), limit),
    }


class ToolOutputDisplay(ABC):
    def __init__(self, truncation: int | None = None) -> None:
        self.truncation = truncation

    @abstractmethod
    def render(self, tool_output: Mapping[str, Any]) -> str:
        """Return the model-facing representation of ``tool_output``."""
        raise NotImplementedError


class FencedToolOutputDisplay(ToolOutputDisplay):
    def render(self, tool_output: Mapping[str, Any]) -> str:
        display = truncate_tool_output(dict(tool_output), self.truncation)
        stdout = display.get("stdout") or ""
        stderr = display.get("stderr") or ""
        if not stdout and not stderr and "output" in display:
            stdout = str(display.get("output") or "")
        if not stdout and not stderr and "error" in display:
            stderr = str(display.get("error") or "")
        return "```stdout\n{}\n```\n\n```stderr\n{}\n```".format(stdout, stderr)


class PlainJsonToolOutputDisplay(ToolOutputDisplay):
    def render(self, tool_output: Mapping[str, Any]) -> str:
        display = truncate_tool_output(dict(tool_output), self.truncation)
        return json.dumps(display, ensure_ascii=False, sort_keys=True)


def terminal_output_payload(tool_output: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "execution_time": tool_output.get("execution_time"),
        "exit_code": tool_output.get("exit_code"),
        "stderr": tool_output.get("stderr") or "",
        "stdout": tool_output.get("stdout") or "",
        "success": bool(tool_output.get("success", False)),
        "truncated": bool(tool_output.get("truncated", False)),
        "truncated_characters": int(tool_output.get("truncated_characters", 0) or 0),
    }


def terminal_output_to_tool_content(
    tool_output: Mapping[str, Any], *, truncation: int | None = None
) -> str:
    return PlainJsonToolOutputDisplay(truncation).render(
        terminal_output_payload(tool_output)
    )


def code_output_to_tool_content(
    code_output: dict[str, Any],
    *,
    code_output_truncation: int | None = None,
) -> str:
    return FencedToolOutputDisplay(code_output_truncation).render(code_output)


def sanitize_tool_output(output: dict[str, Any]) -> dict[str, Any]:
    """Redact absolute paths from tool error channels."""
    sanitized = dict(output)
    for key in ("stderr", "error"):
        value = sanitized.get(key)
        if not isinstance(value, str):
            continue
        aliases: dict[str, str] = {}

        def replace_path(match: re.Match[str]) -> str:
            path = match.group(0)
            if path not in aliases:
                filename = path.rstrip("/").rsplit("/", 1)[-1]
                suffix_index = filename.rfind(".")
                suffix = (
                    filename[suffix_index:]
                    if 0 < suffix_index and len(filename) - suffix_index <= 8
                    else ""
                )
                aliases[path] = f"{uuid4().hex[:8]}{suffix}"
            return aliases[path]

        sanitized[key] = _ABSOLUTE_PATH_RE.sub(replace_path, value)
    return sanitized


__all__ = [
    "FencedToolOutputDisplay",
    "PlainJsonToolOutputDisplay",
    "ToolOutputDisplay",
    "code_output_to_tool_content",
    "sanitize_tool_output",
    "terminal_output_payload",
    "terminal_output_to_tool_content",
    "truncate_output_text",
    "truncate_tool_output",
]
