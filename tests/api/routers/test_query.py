from fastapi.testclient import TestClient

from api.main import app
from api.routers import query as query_router_module
from rag.search_service import SearchResult


def _fake_results() -> list[SearchResult]:
    return [
        SearchResult(
            text="chunk text",
            url="https://example.com/page",
            title="Page title",
            heading_path="Intro",
            score=0.9,
        )
    ]


def test_search_returns_results(monkeypatch) -> None:
    monkeypatch.setattr(
        query_router_module, "hybrid_search", lambda query, top_k=5: _fake_results()
    )

    client = TestClient(app)
    response = client.post("/search", json={"query": "hello"})

    assert response.status_code == 200
    results = response.json()["results"]
    assert results == [
        {
            "text": "chunk text",
            "url": "https://example.com/page",
            "title": "Page title",
            "heading_path": "Intro",
            "score": 0.9,
        }
    ]


def test_answer_returns_citations_but_no_generated_answer(monkeypatch) -> None:
    # LLM provider for /answer is still an open decision (see build_log.md) --
    # this asserts the current, honest behavior: real citations, empty answer.
    monkeypatch.setattr(
        query_router_module, "hybrid_search", lambda query, top_k=5: _fake_results()
    )

    client = TestClient(app)
    response = client.post("/answer", json={"query": "hello"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == ""
    assert body["citations"] == [
        {"url": "https://example.com/page", "title": "Page title", "heading_path": "Intro"}
    ]
