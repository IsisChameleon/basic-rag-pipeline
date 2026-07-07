from fastapi.testclient import TestClient

from api.main import app
from api.routers import ingest as ingest_router_module
from rag.ingest_service import IngestSummary


def test_ingest_returns_202_and_job_completes_in_background(monkeypatch) -> None:
    async def fake_ingest(url: str) -> IngestSummary:
        assert url == "https://example.com/docs"
        return IngestSummary(pages_discovered=3, pages_ingested=3, chunks_stored=12)

    # Patched where the name is used (ingest_router_module), not where it's
    # defined (rag.ingest_service) -- see memory note on mock.patch targets.
    monkeypatch.setattr(ingest_router_module, "ingest_page_with_url", fake_ingest)

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
        "pages_ingested": 3,
        "chunks_stored": 12,
        "error": None,
    }


def test_ingest_job_failure_is_reported_via_status(monkeypatch) -> None:
    async def failing_ingest(url: str) -> IngestSummary:
        raise ValueError("boom")

    monkeypatch.setattr(ingest_router_module, "ingest_page_with_url", failing_ingest)

    client = TestClient(app)
    response = client.post("/ingest", json={"url": "https://example.com/docs"})
    job_id = response.json()["job_id"]

    status_response = client.get(f"/ingest/{job_id}")
    assert status_response.json()["status"] == "failed"
    assert status_response.json()["error"] == "boom"


def test_ingest_job_not_found_returns_404() -> None:
    client = TestClient(app)
    response = client.get("/ingest/does-not-exist")
    assert response.status_code == 404
