"""Builds the RAG object graph, shared by every entrypoint.

This is a factory, not a composition root: a composition root belongs to a
*process* (api/dependencies.py for the API, and later the agent's own), and
each one calls this for the RAG half of its graph, then adds what only it
needs. Keeping the shared half here means a second entrypoint cannot drag
the first one's dependencies -- or its required config -- along with it."""

from __future__ import annotations

from dataclasses import dataclass

from core.settings import Settings
from rag.chunk_repository import ChunkRepository
from rag.document_source import DocumentSource
from rag.encoders import Embedder, Reranker
from rag.llm import LLMClient
from rag.markdown_chunker import MarkdownChunker
from rag.services import AnswerService, IngestService, SearchService
from rag.vector_store import VectorStore


@dataclass
class RagServices:
    search_service: SearchService
    answer_service: AnswerService
    ingest_service: IngestService
    # Held for warm_up only: eager model loading is composition policy (an
    # entrypoint may want fail-fast boot and a fast first request), so it is
    # triggered by the caller, not inside the services.
    embedder: Embedder
    reranker: Reranker

    def warm_up(self) -> None:
        self.embedder.warm_up()
        self.reranker.warm_up()


def build_rag_services(settings: Settings) -> RagServices:
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
    return RagServices(
        search_service=search_service,
        answer_service=AnswerService(search_service, llm),
        ingest_service=IngestService(
            DocumentSource(), chunker, embedder, repository, vector_store
        ),
        embedder=embedder,
        reranker=reranker,
    )
