"""FastAPI's dependency adapters: they pull already-built objects off the
ApiContainer that lifespan put on app.state, so routers depend on a service,
not on the container. The wiring itself lives in api/container.py."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from api.container import ApiContainer
from rag.job_store import JobStore
from rag.services import AnswerService, IngestService, SearchService


def get_container(request: Request) -> ApiContainer:
    return request.app.state.container


def get_search_service(
    container: Annotated[ApiContainer, Depends(get_container)],
) -> SearchService:
    return container.rag.search_service


def get_answer_service(
    container: Annotated[ApiContainer, Depends(get_container)],
) -> AnswerService:
    return container.rag.answer_service


def get_ingest_service(
    container: Annotated[ApiContainer, Depends(get_container)],
) -> IngestService:
    return container.rag.ingest_service


def get_job_store(
    container: Annotated[ApiContainer, Depends(get_container)],
) -> JobStore:
    return container.job_store
