"""Browser-aware terminal client backed by request-scoped assets."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Mapping
from typing import Any

from .asset_store import AssetStore, WebAssetStore

from ._transport import ToolClient
from .fetch import FetchClient
from .terminal import TerminalExecutionClient


class WebTerminalExecutionClient(ToolClient):
    """Run terminal pipelines over asset IDs or fetched web pages."""

    def __init__(
        self,
        terminal_client: TerminalExecutionClient,
        fetch_client: FetchClient,
        *,
        capture_web_assets: bool = False,
    ) -> None:
        super().__init__(
            terminal_client.server_url,
            timeout=terminal_client.timeout,
            max_retries=terminal_client.max_retries,
        )
        self.terminal_client = terminal_client
        self.fetch_client = fetch_client
        self.capture_web_assets = capture_web_assets

    @property
    def request_name(self) -> str:
        return "Asset terminal execution"

    async def execute(
        self,
        *,
        asset_store: AssetStore,
        command: str,
        contents: Any = None,
        truncation: int | None = None,
    ) -> dict[str, Any]:
        asset_ids, pipeline = _parse_asset_cat_command(command)
        if not asset_ids:
            url, pipeline = _parse_browse_command(command)
            if url is None:
                return await self.terminal_client.execute(
                    command=command,
                    contents=contents,
                    truncation=truncation,
                )
            if not isinstance(asset_store, WebAssetStore):
                return _terminal_error(
                    f"browse: {url}: WebAssetStore is required for URL caching"
                )
            was_cached = asset_store.has_url(url)
            asset_id = asset_store.add_url(url)
            try:
                fetched_content = await self._content_for_asset(asset_store, asset_id)
            except _AssetTerminalError as error:
                return _terminal_error(error.message)
            result = await self.terminal_client.execute(
                command=pipeline,
                contents=fetched_content,
                truncation=truncation,
            )
            if self.capture_web_assets and not was_cached:
                result = {**result, "web_asset": asset_store.get(asset_id)}
            return result

        try:
            resolved_contents = await asyncio.gather(
                *(
                    self._content_for_asset(asset_store, asset_id)
                    for asset_id in asset_ids
                )
            )
        except _AssetTerminalError as error:
            return _terminal_error(error.message)
        return await self.terminal_client.execute(
            command=pipeline,
            contents="\n\n".join(resolved_contents),
            truncation=truncation,
        )

    async def _content_for_asset(self, asset_store: AssetStore, asset_id: str) -> str:
        if asset_id not in asset_store:
            raise _AssetTerminalError(
                f"cat: {asset_id}: No such file or directory"
            )
        asset = asset_store.get(asset_id)
        content = asset.get("content")
        if content is not None:
            return str(content)
        url = asset.get("url")
        if not isinstance(url, str) or not url:
            raise _AssetTerminalError(f"cat: {asset_id}: asset has no URL")
        try:
            fetch_output = await self.fetch_client.execute(url=url)
        except Exception as error:  # noqa: BLE001 - return terminal-style errors
            raise _AssetTerminalError(
                f"cat: {asset_id}: failed to fetch {url}: {error}"
            ) from error
        if fetch_output.get("success") is not True:
            raise _AssetTerminalError(
                f"cat: {asset_id}: failed to fetch {url}: "
                f"{fetch_output.get('error', 'request failed')}"
            )
        fetched_content = _fetch_content(fetch_output)
        if fetched_content is None:
            raise _AssetTerminalError(
                f"cat: {asset_id}: fetched response did not contain content"
            )
        asset_store.set(asset_id, "content", fetched_content)
        return fetched_content


class _AssetTerminalError(Exception):
    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


def _parse_asset_cat_command(command: str) -> tuple[list[str], str]:
    match = re.match(r"^\s*cat\s+(.+?)\s*$", command, flags=re.DOTALL)
    if match is None:
        return [], command
    asset_id_text, separator, pipeline = match.group(1).partition("|")
    asset_ids = asset_id_text.split()
    if not asset_ids:
        return [], command
    return asset_ids, pipeline.strip() if separator and pipeline.strip() else "cat"


def _parse_browse_command(command: str) -> tuple[str | None, str]:
    match = re.match(
        r"^\s*browse\s+(\S+)(?:\s*\|\s*(.*?))?\s*$",
        command,
        flags=re.DOTALL,
    )
    if match is None:
        return None, command
    return match.group(1), match.group(2).strip() if match.group(2) else "cat"


def _fetch_content(fetch_output: Mapping[str, Any]) -> str | None:
    data = fetch_output.get("data")
    if not isinstance(data, Mapping):
        return None
    content = data.get("content", data.get("text"))
    return None if content is None else str(content)


def _terminal_error(stderr: str) -> dict[str, Any]:
    return {
        "success": False,
        "stdout": "",
        "stderr": stderr,
        "exit_code": 1,
    }


__all__ = ["WebTerminalExecutionClient"]
