from fastapi.testclient import TestClient

from api.main import app
from api.routers import query as query_router_module
from rag.answer_service import AnswerResult
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


def test_answer_returns_generated_answer_with_ordered_citations(monkeypatch) -> None:
    # generate_answer is the seam the router depends on -- patched where it's
    # used. Its own Gemini call is covered separately in test_answer_service.
    def fake_generate_answer(query, top_k=5):
        return AnswerResult(answer="Contextual retrieval helps [1].", sources=_fake_results())

    monkeypatch.setattr(query_router_module, "generate_answer", fake_generate_answer)

    client = TestClient(app)
    response = client.post("/answer", json={"query": "hello"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Contextual retrieval helps [1]."
    assert body["citations"] == [
        {"url": "https://example.com/page", "title": "Page title", "heading_path": "Intro"}
    ]
