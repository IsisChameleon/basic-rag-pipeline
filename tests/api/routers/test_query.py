from fastapi.testclient import TestClient

from api.main import app
from rag.models import ChunkRecord
from tests.fakes import build_fake_container


def _stored_record() -> ChunkRecord:
    return ChunkRecord(
        id="https://example.com/page#0",
        uri="https://example.com/page",
        title="Page title",
        heading_path="Intro",
        chunk_index=0,
        text="chunk text",
        content_hash="hash",
        fetched_at="2026-01-01T00:00:00+00:00",
    )


def test_search_returns_results() -> None:
    app.state.container = build_fake_container(records=[_stored_record()])

    client = TestClient(app)
    response = client.post("/search", json={"query": "chunk"})

    assert response.status_code == 200
    results = response.json()["results"]
    assert results == [
        {
            "text": "chunk text",
            "uri": "https://example.com/page",
            "title": "Page title",
            "heading_path": "Intro",
            "score": 1.0,  # FakeReranker: one query term overlaps the text
        }
    ]


def test_answer_returns_generated_answer_with_ordered_citations() -> None:
    app.state.container = build_fake_container(
        records=[_stored_record()], llm_response="Contextual retrieval helps [1]."
    )

    client = TestClient(app)
    response = client.post("/answer", json={"query": "chunk"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Contextual retrieval helps [1]."
    assert body["citations"] == [
        {"uri": "https://example.com/page", "title": "Page title", "heading_path": "Intro"}
    ]
