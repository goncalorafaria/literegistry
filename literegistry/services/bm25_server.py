"""Pyserini/Lucene BM25 search service with LiteRegistry registration.

Required environment for the Uvicorn factory::

    LOCAL_SEARCH_CORPUS_JSONL=/path/to/corpus.jsonl
    LUCENE_INDEX_DIR=/path/to/lucene-index

Run directly with ``literegistry bm25 --corpus_jsonl=... --lucene_index_dir=...``.
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


@dataclass(frozen=True)
class Document:
    docid: str
    url: str
    title: str
    text: str


def normalize_url(url: str) -> str:
    return unquote(url).replace(" ", "_").rstrip("/")


def parse_document(record: Mapping[str, object], line_number: int) -> Document:
    docid = str(record.get("docid") or record.get("id") or line_number)
    url = str(record.get("url") or record.get("link") or f"local://{docid}")
    text = str(
        record.get("contents")
        or record.get("text")
        or record.get("content")
        or ""
    )
    if not text:
        raise ValueError(f"missing contents/text at corpus line {line_number}")
    title = str(record.get("title") or text.splitlines()[0][:160] or docid)
    return Document(docid, url, title, text)


class Corpus:
    """JSONL corpus mapping used by the JTC/Open Instruct Lucene server."""

    def __init__(self, corpus_path: str | Path) -> None:
        self.by_id: dict[str, Document] = {}
        self.id_by_url: dict[str, str] = {}
        with Path(corpus_path).open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                document = parse_document(json.loads(line), line_number)
                if document.docid in self.by_id:
                    raise ValueError(f"duplicate docid {document.docid!r}")
                self.by_id[document.docid] = document
                self.id_by_url[normalize_url(document.url)] = document.docid

    def get_by_url(self, url: str) -> Document | None:
        docid = self.id_by_url.get(normalize_url(url))
        return self.by_id.get(docid) if docid else None


class LuceneBM25:
    """Thin Pyserini wrapper; scoring is performed by the Lucene index."""

    def __init__(self, index_path: str | Path) -> None:
        try:
            from pyserini.search.lucene import LuceneSearcher
        except ImportError as error:
            raise RuntimeError(
                "Install the BM25 extra: pip install 'literegistry[bm25]'"
            ) from error
        self.searcher = LuceneSearcher(str(index_path))

    def search(self, query: str, topn: int) -> list[tuple[str, float]]:
        return [
            (str(hit.docid), float(hit.score))
            for hit in self.searcher.search(query, k=topn)
        ]


class SearchRequest(BaseModel):
    query: str | None = Field(default=None, min_length=1)
    url: str | None = Field(default=None, min_length=1)
    topn: int | None = Field(default=None, ge=1, le=100)
    mode: str | None = None
    num_results: int | None = Field(default=None, ge=1, le=100)
    parameters: dict[str, object] = Field(default_factory=dict)


class FetchRequest(BaseModel):
    url: str


def snippet(text: str, max_length: int = 180) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= max_length:
        return normalized
    return f"{normalized[:max_length].rstrip()}..."


def _create_bm25_app(
    corpus_path: str | Path | None = None,
    index_path: str | Path | None = None,
    *,
    searcher: LuceneBM25 | None = None,
) -> FastAPI:
    """Create the service; ``searcher`` allows lightweight tests."""
    corpus_path = corpus_path or os.getenv("LOCAL_SEARCH_CORPUS_JSONL")
    index_path = index_path or os.getenv("LUCENE_INDEX_DIR")
    if not corpus_path:
        raise RuntimeError("LOCAL_SEARCH_CORPUS_JSONL is required")
    if searcher is None and not index_path:
        raise RuntimeError("LUCENE_INDEX_DIR is required")

    corpus = Corpus(corpus_path)
    active_searcher = searcher or LuceneBM25(index_path)
    app = FastAPI(title="Local Lucene BM25 Search", version="0.2.0")

    def results(query: str, topn: int) -> list[dict[str, object]]:
        output: list[dict[str, object]] = []
        for position, (docid, score) in enumerate(
            active_searcher.search(query, topn), start=1
        ):
            document = corpus.by_id.get(docid)
            if document is None:
                continue
            summary = snippet(document.text)
            output.append(
                {
                    "url": document.url,
                    "link": document.url,
                    "title": document.title or document.docid,
                    "snippet": summary,
                    "description": summary,
                    "docid": document.docid,
                    "id": document.docid,
                    "score": score,
                    "position": position,
                }
            )
        return output

    @app.get("/")
    def root() -> dict[str, object]:
        return {
            "service": "Local Lucene BM25 Search",
            "documents": len(corpus.by_id),
            "endpoints": ["/search", "/get_content"],
        }

    @app.post("/search")
    def search(request: SearchRequest) -> dict[str, object]:
        if request.mode == "url":
            if not request.url:
                raise HTTPException(
                    status_code=400,
                    detail="url is required when mode='url'",
                )
            document = corpus.get_by_url(request.url)
            if document is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"URL not found in corpus: {request.url}",
                )
            return {
                "success": True,
                "mode": "url",
                "data": {
                    "title": document.title or document.docid,
                    "content": document.text,
                    "url": document.url,
                },
                "cache_hit": False,
            }
        if request.mode is not None and request.mode != "query":
            raise HTTPException(
                status_code=400,
                detail="mode must be 'query' or 'url'",
            )
        if not request.query:
            raise HTTPException(status_code=400, detail="query is required")

        found = results(request.query, request.topn or request.num_results or 10)
        if request.mode == "query":
            return {
                "success": True,
                "mode": "query",
                "data": {"organic": found},
                "cache_hit": False,
            }
        return {"results": found}

    @app.post("/get_content")
    def get_content(request: FetchRequest) -> dict[str, str]:
        document = corpus.get_by_url(request.url)
        if document is None:
            raise HTTPException(
                status_code=404,
                detail=f"URL not found in corpus: {request.url}",
            )
        return {
            "title": document.title or document.docid,
            "content": document.text,
        }

    return app


@dataclass(frozen=True)
class LiteRegistrySearchConfig:
    corpus_jsonl: str
    lucene_index_dir: str
    registry: str
    host: str = "0.0.0.0"
    port: int = 1214
    advertised_host: str | None = None
    heartbeat_interval: int = 30
    service_name: str | None = None


class LiteRegistryLocalSearchServer:
    """Lucene worker using LiteRegistry's native registration protocol."""

    def __init__(
        self,
        config: LiteRegistrySearchConfig,
        *,
        app: FastAPI | None = None,
    ) -> None:
        from literegistry import ServerRegistry, get_kvstore

        self.config = config
        self.service_name = (
            config.service_name or f"localsearch:{Path(config.corpus_jsonl).stem}"
        )
        self.app = app or _create_bm25_app(
            config.corpus_jsonl,
            config.lucene_index_dir,
        )
        self.registry = ServerRegistry(store=get_kvstore(config.registry))
        advertised_host = config.advertised_host or socket.getfqdn()
        self.url = f"http://{advertised_host}"
        self.running = False
        self.heartbeat_task: asyncio.Task[None] | None = None
        self._install_lifecycle()

    def metadata(self) -> dict[str, Any]:
        return {
            "model_path": self.service_name,
            "host": self.config.host,
            "port": self.config.port,
            "backend": "lucene-bm25",
            "extra_kwargs": {
                "modes": ["query", "url"],
                "corpus_jsonl": self.config.corpus_jsonl,
                "lucene_index_dir": self.config.lucene_index_dir,
                "heartbeat_interval": self.config.heartbeat_interval,
                "service_name": self.service_name,
            },
        }

    async def start(self) -> None:
        await self.registry.register_server(
            self.url,
            self.config.port,
            self.metadata(),
        )
        self.running = True
        self.heartbeat_task = asyncio.create_task(self._heartbeat_loop())

    async def _heartbeat_loop(self) -> None:
        while self.running:
            try:
                await self.registry.heartbeat(self.url, self.config.port)
            except Exception:
                pass
            await asyncio.sleep(self.config.heartbeat_interval)

    async def stop(self) -> None:
        self.running = False
        if self.heartbeat_task:
            self.heartbeat_task.cancel()
            try:
                await self.heartbeat_task
            except asyncio.CancelledError:
                pass
            self.heartbeat_task = None
        await self.registry.deregister()

    def _install_lifecycle(self) -> None:
        @self.app.on_event("startup")
        async def startup() -> None:
            await self.start()

        @self.app.on_event("shutdown")
        async def shutdown() -> None:
            await self.stop()


def _create_literegistry_app() -> FastAPI:
    config = LiteRegistrySearchConfig(
        corpus_jsonl=os.environ["LOCAL_SEARCH_CORPUS_JSONL"],
        lucene_index_dir=os.environ["LUCENE_INDEX_DIR"],
        registry=os.environ["LITEREGISTRY_REGISTRY"],
        host=os.getenv("LITEREGISTRY_HOST", "0.0.0.0"),
        port=int(os.getenv("LITEREGISTRY_PORT", "1214")),
        advertised_host=os.getenv("LITEREGISTRY_ADVERTISED_HOST"),
        heartbeat_interval=int(
            os.getenv("LITEREGISTRY_HEARTBEAT_INTERVAL", "30")
        ),
        service_name=os.getenv("LITEREGISTRY_SERVICE_NAME"),
    )
    return LiteRegistryLocalSearchServer(config).app


def create_app(
    corpus_path: str | Path | None = None,
    index_path: str | Path | None = None,
    *,
    searcher: LuceneBM25 | None = None,
) -> FastAPI:
    """Create an unregistered app or an environment-configured worker."""
    if (
        corpus_path is None
        and index_path is None
        and searcher is None
        and os.getenv("LITEREGISTRY_REGISTRY")
    ):
        return _create_literegistry_app()
    return _create_bm25_app(corpus_path, index_path, searcher=searcher)


def main(
    corpus_jsonl: str | None = None,
    lucene_index_dir: str | None = None,
    registry: str | None = None,
    service_name: str | None = None,
    advertised_host: str | None = None,
    host: str = "0.0.0.0",
    port: int = 8000,
    workers: int = 1,
) -> None:
    """Serve local BM25, optionally registering it with LiteRegistry."""
    corpus_jsonl = corpus_jsonl or os.getenv("LOCAL_SEARCH_CORPUS_JSONL")
    lucene_index_dir = lucene_index_dir or os.getenv("LUCENE_INDEX_DIR")
    registry = registry or os.getenv("LITEREGISTRY_REGISTRY")
    if not corpus_jsonl or not lucene_index_dir:
        raise ValueError("corpus_jsonl and lucene_index_dir are required")
    if workers < 1:
        raise ValueError("workers must be positive")
    if registry and workers != 1:
        raise ValueError(
            "registered local search must use workers=1; scale with replicas"
        )

    os.environ["LOCAL_SEARCH_CORPUS_JSONL"] = corpus_jsonl
    os.environ["LUCENE_INDEX_DIR"] = lucene_index_dir
    if registry:
        os.environ["LITEREGISTRY_REGISTRY"] = registry
        os.environ["LITEREGISTRY_PORT"] = str(port)
        if service_name:
            os.environ["LITEREGISTRY_SERVICE_NAME"] = service_name
        if advertised_host:
            os.environ["LITEREGISTRY_ADVERTISED_HOST"] = advertised_host

    import uvicorn

    uvicorn.run(
        "literegistry.services.bm25_server:create_app",
        factory=True,
        host=host,
        port=port,
        workers=workers,
    )


try:
    app = create_app()
except (KeyError, RuntimeError) as configuration_error:
    configuration_message = str(configuration_error)
    app = FastAPI(title="Local Lucene BM25 Search", version="0.2.0")

    @app.get("/")
    def unconfigured() -> dict[str, str]:
        return {"error": configuration_message}


if __name__ == "__main__":
    import fire

    fire.Fire(main)
