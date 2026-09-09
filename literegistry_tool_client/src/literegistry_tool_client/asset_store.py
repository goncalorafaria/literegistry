"""In-memory JSON asset storage with stable opaque identifiers."""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping, Sequence
from typing import Any, Callable
from uuid import uuid4


def normalize_web_page_content(content: str) -> str | None:
    """Return a page body, unwrapping a serialized URL-fetch response.

    Historical web-page bundles sometimes stored the entire LiteRegistry
    ``mode=url`` response as the mapping value instead of ``data.content``.
    Passing that transport object to the terminal makes grep-like commands
    inspect response metadata rather than the fetched document.  An envelope
    with no body is treated as a cache miss so callers may fetch the URL again.
    Ordinary text, including arbitrary JSON documents, is returned unchanged.
    """
    try:
        response = json.loads(content)
    except json.JSONDecodeError:
        return content
    if not isinstance(response, Mapping) or response.get("mode") != "url":
        return content
    data = response.get("data")
    if not isinstance(data, Mapping):
        return content
    body = data.get("content", data.get("text"))
    if not isinstance(body, str):
        return None
    return body or None


class AssetStore:
    """Store JSON objects so they can be referenced and enriched across turns.

    The store owns its records: inputs, returned records, and update values are
    copied to prevent callers from changing an asset without using this API.
    """

    def __init__(self) -> None:
        self._assets: dict[str, dict[str, Any]] = {}

    def add(self, asset: Mapping[str, Any]) -> str:
        """Add one JSON object and return its globally unique asset ID."""
        stored_asset = _json_object_copy(asset)
        asset_id = f"asset_{uuid4().hex}"
        self._assets[asset_id] = stored_asset
        return asset_id

    def add_all(self, assets: Sequence[Mapping[str, Any]]) -> list[str]:
        """Add a list of JSON objects and return their asset IDs in order."""
        return [self.add(asset) for asset in assets]

    def get(self, asset_id: str) -> dict[str, Any]:
        """Return a copy of the object identified by ``asset_id``."""
        return copy.deepcopy(self._assets[asset_id])

    def show(
        self, stringify: Callable[[dict[str, Any]], str], keys: Sequence[str]
    ) -> str:
        """Render selected assets as a Markdown search-results list."""
        lines = ["# Search Results :", ""]
        lines.extend(f"* [ {asset_id} ] {stringify(self.get(asset_id))}" for asset_id in keys)
        return "\n".join(lines)

    def update(
        self,
        asset_id: str,
        key: str | Mapping[str, Any] | None = None,
        value: Any = None,
        /,
        **fields: Any,
    ) -> dict[str, Any]:
        """Add or override fields on an existing asset and return its new value.

        ``update(asset_id, "content", text)`` updates one field.
        ``update(asset_id, {"content": text})`` and
        ``update(asset_id, content=text)`` update one or more fields.
        """
        updates = _update_fields(key, value, fields)
        asset = self._assets[asset_id]
        candidate = {**asset, **updates}
        self._assets[asset_id] = _json_object_copy(candidate)
        return copy.deepcopy(self._assets[asset_id])

    def set(self, asset_id: str, key: str, value: Any) -> dict[str, Any]:
        """Add or override one field on an existing asset."""
        return self.update(asset_id, key, value)

    def __contains__(self, asset_id: object) -> bool:
        return isinstance(asset_id, str) and asset_id in self._assets

    def __len__(self) -> int:
        return len(self._assets)


class WebAssetStore(AssetStore):
    """An ``AssetStore`` that indexes exactly one asset per URL."""

    def __init__(self, web_pages: Mapping[str, str] | None = None) -> None:
        """Create a URL-indexed store, optionally seeded with page bodies.

        ``web_pages`` is a request-scoped ``{url: markdown}`` mapping, such as
        the one retained beside a search-agent rollout. A later ``browse
        <url>`` uses seeded content directly and therefore makes no network
        request.
        """
        super().__init__()
        self._asset_ids_by_url: dict[str, str] = {}
        if web_pages is None:
            return
        if not isinstance(web_pages, Mapping):
            raise TypeError("web_pages must be a mapping of URL strings to content strings")
        for url, content in web_pages.items():
            if not isinstance(url, str) or not url:
                raise ValueError("web_pages keys must be non-empty URL strings")
            if not isinstance(content, str):
                raise TypeError("web_pages values must be content strings")
            normalized_content = normalize_web_page_content(content)
            if normalized_content is not None:
                self.add_url(url, content=normalized_content)

    def add(self, asset: Mapping[str, Any]) -> str:
        """Add a URL asset, or return the ID already registered for its URL."""
        stored_asset = _json_object_copy(asset)
        url = _asset_url(stored_asset)
        existing_id = self._asset_ids_by_url.get(url)
        if existing_id is not None:
            return existing_id

        asset_id = super().add(stored_asset)
        self._asset_ids_by_url[url] = asset_id
        return asset_id

    def add_url(self, url: str, **fields: Any) -> str:
        """Add a URL and optional fields, returning its stable asset ID."""
        if "url" in fields:
            raise TypeError("url must be passed as the first argument to add_url")
        return self.add({"url": url, **fields})

    def has_url(self, url: str) -> bool:
        """Return whether ``url`` has an asset in this store."""
        return url in self._asset_ids_by_url

    def get_id_by_url(self, url: str) -> str:
        """Return the ID for ``url`` or raise ``KeyError`` if it is absent."""
        return self._asset_ids_by_url[url]

    def get_by_url(self, url: str) -> dict[str, Any]:
        """Return a copy of the asset registered for ``url``."""
        return self.get(self.get_id_by_url(url))

    def update(
        self,
        asset_id: str,
        key: str | Mapping[str, Any] | None = None,
        value: Any = None,
        /,
        **fields: Any,
    ) -> dict[str, Any]:
        """Update an asset while preserving the one-to-one URL index."""
        updates = _update_fields(key, value, fields)
        asset = self._assets[asset_id]
        candidate = _json_object_copy({**asset, **updates})
        old_url = _asset_url(asset)
        new_url = _asset_url(candidate)
        existing_id = self._asset_ids_by_url.get(new_url)
        if existing_id is not None and existing_id != asset_id:
            raise ValueError(f"URL is already registered to asset {existing_id!r}")

        self._assets[asset_id] = candidate
        if new_url != old_url:
            del self._asset_ids_by_url[old_url]
            self._asset_ids_by_url[new_url] = asset_id
        return copy.deepcopy(candidate)


def _json_object_copy(asset: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(asset, Mapping):
        raise TypeError("asset must be a JSON object (a mapping)")

    copied_asset = copy.deepcopy(dict(asset))
    try:
        json.dumps(copied_asset, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise TypeError("asset must contain only JSON-serializable values") from error
    return copied_asset


def _asset_url(asset: Mapping[str, Any]) -> str:
    url = asset.get("url")
    if not isinstance(url, str) or not url:
        raise ValueError("web assets must contain a non-empty 'url' string")
    return url


def _update_fields(
    key: str | Mapping[str, Any] | None, value: Any, keyword_fields: Mapping[str, Any]
) -> dict[str, Any]:
    if isinstance(key, Mapping):
        if value is not None or keyword_fields:
            raise TypeError("mapping updates cannot be combined with other update fields")
        return dict(key)
    if isinstance(key, str):
        if keyword_fields:
            raise TypeError("a field update cannot be combined with keyword fields")
        return {key: value}
    if key is None and keyword_fields:
        return dict(keyword_fields)
    raise TypeError("provide a field and value, a mapping, or keyword fields")
