from fastapi.testclient import TestClient

from api.main import app
from api.routers import ingest as ingest_router_module
from rag.ingest_service import IngestSummary


def test_ingest_returns_summary_from_ingest_section(monkeypatch) -> None:
    async def fake_ingest_section(url: str) -> IngestSummary:
        assert url == "https://example.com/docs"
        return IngestSummary(pages_discovered=3, pages_ingested=3, chunks_stored=12)

    # Patched where the name is used (ingest_router_module), not where it's
    # defined (rag.ingest_service) -- see memory note on mock.patch targets.
    monkeypatch.setattr(ingest_router_module, "ingest_section", fake_ingest_section)

    client = TestClient(app)
    response = client.post("/ingest", json={"url": "https://example.com/docs"})

    assert response.status_code == 200
    assert response.json() == {
        "status": "completed",
        "pages_discovered": 3,
        "pages_ingested": 3,
        "chunks_stored": 12,
    }
