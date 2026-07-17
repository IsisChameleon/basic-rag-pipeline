from __future__ import annotations

import asyncio
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from api.dependencies import get_ingest_service, get_job_store
from api.schemas import IngestAcceptedResponse, IngestJobResponse, IngestRequest
from rag.job_store import JobStore
from rag.services import IngestService

router = APIRouter(tags=["ingest"])


def _run_ingest_job(
    ingest_service: IngestService, job_store: JobStore, job_id: str, url: str
) -> None:
    """Runs on FastAPI's background-task thread, after the 202 response has
    already been sent to the client -- see
    https://fastapi.tiangolo.com/tutorial/background-tasks/. Ingesting a
    whole sitemap section (fetch/extract/chunk/embed/store for possibly
    dozens of pages) can take longer than a typical HTTP client or reverse
    proxy timeout, so the work happens here instead of inline in the
    request/response cycle.
    """
    try:
        summary = asyncio.run(ingest_service.ingest(url))
        job_store.mark_completed(job_id, summary)
    except Exception as exc:
        job_store.mark_failed(job_id, str(exc))


@router.post("/ingest", response_model=IngestAcceptedResponse, status_code=202)
def ingest(
    request: IngestRequest,
    background_tasks: BackgroundTasks,
    ingest_service: Annotated[IngestService, Depends(get_ingest_service)],
    job_store: Annotated[JobStore, Depends(get_job_store)],
) -> IngestAcceptedResponse:
    """Accepts a section URL and starts ingesting it in the background,
    returning immediately with a job id. Poll GET /ingest/{job_id} for
    status and, once "completed", the resulting page/chunk counts.
    """
    job_id = job_store.create()
    background_tasks.add_task(_run_ingest_job, ingest_service, job_store, job_id, request.url)
    return IngestAcceptedResponse(job_id=job_id, status="pending")


@router.get("/ingest/{job_id}", response_model=IngestJobResponse)
def get_ingest_job(
    job_id: str,
    job_store: Annotated[JobStore, Depends(get_job_store)],
) -> IngestJobResponse:
    job = job_store.get(job_id)
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
