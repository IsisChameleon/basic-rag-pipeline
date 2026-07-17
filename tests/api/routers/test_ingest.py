from fastapi.testclient import TestClient

from api.main import app
from rag.models import Document
from tests.fakes import build_fake_container


def test_ingest_returns_202_and_job_completes_in_background() -> None:
    app.state.container = build_fake_container(
        documents=[
            Document(uri="https://example.com/docs/a", title="A", markdown="# One\n\nalpha beta"),
            Document(uri="https://example.com/docs/b", title="B", markdown="# Two\n\ngamma delta"),
            None,  # one discovered page whose fetch failed
        ]
    )

    client = TestClient(app)
    response = client.post("/ingest", json={"url": "https://example.com/docs"})

    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "pending"
    job_id = body["job_id"]

    # TestClient runs background tasks to completion before .post() returns
    # (Starlette executes them within the same ASGI call), so the job is
    # already done by the time we poll here.
    status_response = client.get(f"/ingest/{job_id}")
    assert status_response.status_code == 200
    assert status_response.json() == {
        "job_id": job_id,
        "status": "completed",
        "pages_discovered": 3,
        "pages_ingested": 2,
        "chunks_stored": 2,  # each markdown document above packs into one chunk
        "error": None,
    }


def test_ingest_job_failure_is_reported_via_status() -> None:
    app.state.container = build_fake_container(source_error=ValueError("boom"))

    client = TestClient(app)
    response = client.post("/ingest", json={"url": "https://example.com/docs"})
    job_id = response.json()["job_id"]

    status_response = client.get(f"/ingest/{job_id}")
    assert status_response.json()["status"] == "failed"
    assert status_response.json()["error"] == "boom"


def test_ingest_job_not_found_returns_404() -> None:
    app.state.container = build_fake_container()

    client = TestClient(app)
    response = client.get("/ingest/does-not-exist")
    assert response.status_code == 404
