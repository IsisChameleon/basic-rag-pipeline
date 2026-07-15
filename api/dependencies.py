"""Composition root: the only code in the app that knows concrete wiring.
build_container() constructs the object graph; everything else receives its
dependencies. Tests replace the whole graph by putting a fake-filled
Container on app.state.container."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, Request

from core.settings import Settings
from rag.chunk_repository import ChunkRepository
from rag.document_source import DocumentSource
from rag.encoders import Embedder, Reranker
from rag.job_store import JobStore
from rag.llm import LLMClient
from rag.markdown_chunker import MarkdownChunker
from rag.services import AnswerService, IngestService, SearchService
from rag.vector_store import VectorStore


@dataclass
class Container:
    settings: Settings
    search_service: SearchService
    answer_service: AnswerService
    ingest_service: IngestService
    job_store: JobStore
    # Held for warm_up only: eager model loading is composition policy (the
    # API wants fail-fast boot and a fast first request), so it is triggered
    # here, not inside the services.
    embedder: Embedder
    reranker: Reranker

    def warm_up(self) -> None:
        self.embedder.warm_up()
        self.reranker.warm_up()


def build_container(settings: Settings | None = None) -> Container:
    settings = settings or Settings()
    embedder = Embedder(settings.embedding_model)  # cheap; model loads lazily
    reranker = Reranker(settings.reranker_model)  # cheap; model loads lazily
    repository = ChunkRepository(settings.db_path)
    vector_store = VectorStore(settings.chroma_dir)
    chunker = MarkdownChunker(
        count_tokens=embedder.count_tokens,
        target_tokens=settings.chunk_target_tokens,
        overlap_tokens=settings.chunk_overlap_tokens,
    )
    llm = LLMClient(api_key=settings.google_api_key, model=settings.llm_model)
    search_service = SearchService(repository, vector_store, embedder, reranker)
    return Container(
        settings=settings,
        search_service=search_service,
        answer_service=AnswerService(search_service, llm),
        ingest_service=IngestService(
            DocumentSource(), chunker, embedder, repository, vector_store
        ),
        job_store=JobStore(),
        embedder=embedder,
        reranker=reranker,
    )


def get_container(request: Request) -> Container:
    return request.app.state.container


def get_search_service(
    container: Annotated[Container, Depends(get_container)],
) -> SearchService:
    return container.search_service


def get_answer_service(
    container: Annotated[Container, Depends(get_container)],
) -> AnswerService:
    return container.answer_service


def get_ingest_service(
    container: Annotated[Container, Depends(get_container)],
) -> IngestService:
    return container.ingest_service


def get_job_store(
    container: Annotated[Container, Depends(get_container)],
) -> JobStore:
    return container.job_store
