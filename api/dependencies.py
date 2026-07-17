"""The API's composition root: the only code in the API that knows concrete
wiring. build_container() takes the shared RAG graph from rag.factory and
adds what only the API needs; everything else receives its dependencies.
Tests replace the whole graph by putting a fake-filled Container on
app.state.container."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, Request

from core.settings import Settings
from rag.factory import RagServices, build_rag_services
from rag.job_store import JobStore
from rag.services import AnswerService, IngestService, SearchService


@dataclass
class Container:
    settings: Settings
    rag: RagServices
    # JobStore is API-only: it backs the async ingest job endpoints, and is
    # in-process, so it has no meaning outside this process.
    job_store: JobStore

    def warm_up(self) -> None:
        self.rag.warm_up()


def build_container(settings: Settings | None = None) -> Container:
    settings = settings or Settings()
    return Container(
        settings=settings,
        rag=build_rag_services(settings),
        job_store=JobStore(),
    )


def get_container(request: Request) -> Container:
    return request.app.state.container


def get_search_service(
    container: Annotated[Container, Depends(get_container)],
) -> SearchService:
    return container.rag.search_service


def get_answer_service(
    container: Annotated[Container, Depends(get_container)],
) -> AnswerService:
    return container.rag.answer_service


def get_ingest_service(
    container: Annotated[Container, Depends(get_container)],
) -> IngestService:
    return container.rag.ingest_service


def get_job_store(
    container: Annotated[Container, Depends(get_container)],
) -> JobStore:
    return container.job_store
