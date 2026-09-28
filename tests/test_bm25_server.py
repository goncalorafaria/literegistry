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


def test_original_corpus_url_title_and_fetch_are_preserved(tmp_path):
    from literegistry.services.bm25_server import parse_document
    text = '---\ntitle: "Pokémon World Championships - Wikipedia"\ndate: 2015-11-20\n---\nFull original page text.'
    record = {'docid':'54072','url':'https://en.wikipedia.org/wiki/Pok%C3%A9mon_World_Championships','text':text}
    document = parse_document(record, 1)
    assert document.title == 'Pokémon World Championships - Wikipedia'
    assert document.url == record['url']
    assert document.text == text
    assert parse_document({**record,'title':'Explicit title'},1).title == 'Explicit title'
    corpus=tmp_path/'corpus.jsonl'
    corpus.write_text(json.dumps(record)+'\n')
    class Searcher:
        def search(self, query, topn):
            return [('54072', 1.0)]
    client=TestClient(create_app(corpus,searcher=Searcher()))
    hit=client.post('/search',json={'query':'Pokémon','topn':1}).json()['results'][0]
    assert hit['url'] == record['url'] and hit['title'] == document.title
    assert client.post('/get_content',json={'url':hit['url']}).json()['content'] == text
