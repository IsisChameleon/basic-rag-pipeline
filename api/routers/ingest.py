from __future__ import annotations

import asyncio

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

from rag import jobs
from rag.ingest_service import ingest_page_with_url

router = APIRouter(tags=["ingest"])


class IngestRequest(BaseModel):
    url: str


class IngestAcceptedResponse(BaseModel):
    job_id: str
    status: str  # always "pending" at creation time


class IngestJobResponse(BaseModel):
    job_id: str
    status: str  # "pending" | "completed" | "failed"
    pages_discovered: int | None = None
    pages_ingested: int | None = None
    chunks_stored: int | None = None
    error: str | None = None


def _run_ingest_job(job_id: str, url: str) -> None:
    """Runs on FastAPI's background-task thread, after the 202 response has
    already been sent to the client -- see
    https://fastapi.tiangolo.com/tutorial/background-tasks/. Ingesting a
    whole sitemap section (fetch/extract/chunk/embed/store for possibly
    dozens of pages) can take longer than a typical HTTP client or reverse
    proxy timeout, so the work happens here instead of inline in the
    request/response cycle.
    """
    try:
        summary = asyncio.run(ingest_page_with_url(url))
        jobs.mark_completed(job_id, summary)
    except Exception as exc:
        jobs.mark_failed(job_id, str(exc))


@router.post("/ingest", response_model=IngestAcceptedResponse, status_code=202)
def ingest(request: IngestRequest, background_tasks: BackgroundTasks) -> IngestAcceptedResponse:
    """Accepts a section URL and starts ingesting it in the background,
    returning immediately with a job id. Poll GET /ingest/{job_id} for
    status and, once "completed", the resulting page/chunk counts.
    """
    job_id = jobs.create_job()
    background_tasks.add_task(_run_ingest_job, job_id, request.url)
    return IngestAcceptedResponse(job_id=job_id, status="pending")


@router.get("/ingest/{job_id}", response_model=IngestJobResponse)
def get_ingest_job(job_id: str) -> IngestJobResponse:
    job = jobs.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")

    return IngestJobResponse(
        job_id=job_id,
        status=job.status,
        pages_discovered=job.result.pages_discovered if job.result else None,
        pages_ingested=job.result.pages_ingested if job.result else None,
        chunks_stored=job.result.chunks_stored if job.result else None,
        error=job.error,
    )
