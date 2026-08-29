from __future__ import annotations

import json
from unittest.mock import patch

from fastapi.testclient import TestClient

from literegistry.services.bm25_server import Corpus, create_app, main


class FakeLuceneSearcher:
    def search(self, query: str, topn: int) -> list[tuple[str, float]]:
        assert query == "annual revenue"
        assert topn == 2
        return [("filing", 4.2)]


def test_upstream_and_gateway_search_shapes(tmp_path) -> None:
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        json.dumps(
            {
                "id": "filing",
                "url": "https://example.test/filing",
                "title": "Annual filing",
                "contents": "Revenue grew in the annual report.",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    client = TestClient(create_app(corpus, searcher=FakeLuceneSearcher()))

    direct = client.post(
        "/search",
        json={"query": "annual revenue", "topn": 2},
    ).json()
    gateway = client.post(
        "/search",
        json={"mode": "query", "query": "annual revenue", "num_results": 2},
    ).json()

    assert direct["results"][0]["url"] == "https://example.test/filing"
    assert gateway["data"]["organic"][0]["id"] == "filing"
    assert (
        client.post(
            "/get_content",
            json={"url": "https://example.test/filing"},
        ).json()["content"]
        == "Revenue grew in the annual report."
    )


def test_id_contents_only_corpus_uses_local_url(tmp_path) -> None:
    corpus_path = tmp_path / "corpus.jsonl"
    corpus_path.write_text(
        '{"id": 7, "contents": "BC-v2 source text"}\n',
        encoding="utf-8",
    )

    corpus = Corpus(corpus_path)

    assert corpus.by_id["7"].url == "local://7"
    assert corpus.get_by_url("local://7").text == "BC-v2 source text"


def test_fire_entrypoint_uses_literegistry_factory(tmp_path) -> None:
    corpus_path = tmp_path / "corpus.jsonl"
    corpus_path.write_text(
        '{"id": 1, "contents": "ai2 hello"}\n',
        encoding="utf-8",
    )

    with patch("uvicorn.run") as run:
        main(
            corpus_jsonl=str(corpus_path),
            lucene_index_dir=str(tmp_path / "index"),
            port=1214,
        )

    run.assert_called_once_with(
        "literegistry.services.bm25_server:create_app",
        factory=True,
        host="0.0.0.0",
        port=1214,
        workers=1,
    )
