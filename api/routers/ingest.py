from __future__ import annotations

import asyncio

from fastapi import APIRouter
from pydantic import BaseModel

from rag.ingest_service import ingest_section

router = APIRouter(tags=["ingest"])


class IngestRequest(BaseModel):
    url: str


class IngestResponse(BaseModel):
    status: str
    pages_discovered: int
    pages_ingested: int
    chunks_stored: int


@router.post("/ingest", response_model=IngestResponse)
def ingest(request: IngestRequest) -> IngestResponse:
    """Discover every page under `request.url`'s path prefix via the site's
    sitemap.xml, then fetch/extract/chunk/embed/store each one.

    Declared as a sync `def`, not `async def`: FastAPI runs sync handlers in
    its worker thread pool, whereas an `async def` handler runs inline on the
    single event loop that also serves /search and /answer. ingest_section
    mixes async network fetches with CPU-bound work (trafilatura extraction,
    embedding, SQLite/Chroma writes) -- running that inline on the event loop
    would stall every other in-flight request for the duration.
    Reference: https://fastapi.tiangolo.com/async/#path-operation-functions
    """
    summary = asyncio.run(ingest_section(request.url))
    return IngestResponse(
        status="completed",
        pages_discovered=summary.pages_discovered,
        pages_ingested=summary.pages_ingested,
        chunks_stored=summary.chunks_stored,
    )
