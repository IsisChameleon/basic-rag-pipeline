from __future__ import annotations

import hashlib
from datetime import UTC, datetime

from langfuse import observe
from loguru import logger

from rag.chunk_repository import ChunkRepository
from rag.document_source import DocumentSource
from rag.encoders import Embedder, Reranker
from rag.llm import LLMClient
from rag.markdown_chunker import MarkdownChunker
from rag.models import AnswerResult, ChunkRecord, IngestSummary, SearchResult
from rag.observability import ObservationType
from rag.vector_store import VectorStore

# Each pipeline stage keeps its own @observe-decorated method so a Langfuse
# trace shows one span per stage (sparse -> dense -> fusion -> rerank). When
# Langfuse is not configured the decorators are no-ops and these run
# unchanged.


class SearchService:
    """Hybrid retrieval: keyword (repository) + vector (store) candidates,
    fused by Reciprocal Rank Fusion, reordered by the cross-encoder."""

    def __init__(
        self,
        repository: ChunkRepository,
        vector_store: VectorStore,
        embedder: Embedder,
        reranker: Reranker,
        *,
        candidate_pool_size: int = 20,
        rrf_k: int = 60,
    ) -> None:
        self._repository = repository
        self._vector_store = vector_store
        self._embedder = embedder
        self._reranker = reranker
        self._candidate_pool_size = candidate_pool_size
        self._rrf_k = rrf_k

    @observe(name="hybrid-search")
    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        candidates = self._rrf_fuse(self._keyword_retrieve(query), self._vector_retrieve(query))
        if not candidates:
            return []
        return self._rerank(query, candidates)[:top_k]

    @observe(name="bm25-retrieve", as_type=ObservationType.RETRIEVER)
    def _keyword_retrieve(self, query: str) -> list[ChunkRecord]:
        return self._repository.search_keyword(query, limit=self._candidate_pool_size)

    @observe(name="vector-retrieve", as_type=ObservationType.RETRIEVER)
    def _vector_retrieve(self, query: str) -> list[ChunkRecord]:
        query_vector = self._embedder.embed_query(query)
        return self._vector_store.search_similar(query_vector, limit=self._candidate_pool_size)

    @observe(name="rrf-fuse")
    def _rrf_fuse(
        self, keyword_hits: list[ChunkRecord], vector_hits: list[ChunkRecord]
    ) -> list[ChunkRecord]:
        """RRF uses rank order only -- neither store's raw score survives, so
        the two retrieval methods need no score calibration against each
        other."""
        fused_scores: dict[str, float] = {}
        records: dict[str, ChunkRecord] = {}

        for hits in (keyword_hits, vector_hits):
            for rank, record in enumerate(hits):
                fused_scores[record.id] = (
                    fused_scores.get(record.id, 0.0) + 1.0 / (self._rrf_k + rank + 1)
                )
                records.setdefault(record.id, record)

        ordered = sorted(fused_scores, key=lambda chunk_id: fused_scores[chunk_id], reverse=True)
        return [records[chunk_id] for chunk_id in ordered[: self._candidate_pool_size]]

    @observe(name="rerank")
    def _rerank(self, query: str, candidates: list[ChunkRecord]) -> list[SearchResult]:
        scores = self._reranker.rerank(query, [c.text for c in candidates])
        ranked = sorted(
            zip(candidates, scores, strict=True), key=lambda pair: pair[1], reverse=True
        )
        return [SearchResult(chunk=record, score=float(score)) for record, score in ranked]


class IngestService:
    def __init__(
        self,
        source: DocumentSource,
        chunker: MarkdownChunker,
        embedder: Embedder,
        repository: ChunkRepository,
        vector_store: VectorStore,
    ) -> None:
        self._source = source
        self._chunker = chunker
        self._embedder = embedder
        self._repository = repository
        self._vector_store = vector_store

    async def ingest(self, source_url: str) -> IngestSummary:
        logger.info("Starting ingest for {}", source_url)
        documents = await self._source.load(source_url)
        if not documents:
            logger.warning("Nothing to ingest for {} -- no pages discovered", source_url)
            return IngestSummary(pages_discovered=0, pages_ingested=0, chunks_stored=0)

        pages_ingested = 0
        chunks_stored = 0
        for document in documents:
            if document is None:
                continue  # fetch/extract failed; already logged by the source

            chunks = self._chunker.chunk(document.markdown)
            if not chunks:
                logger.debug("Skipping {} -- produced no chunks", document.uri)
                continue

            content_hash = hashlib.sha256(document.markdown.encode("utf-8")).hexdigest()
            fetched_at = datetime.now(UTC).isoformat()
            records = [
                ChunkRecord(
                    id=f"{document.uri}#{index}",
                    uri=document.uri,
                    title=document.title,
                    heading_path=chunk.heading_path,
                    chunk_index=index,
                    text=chunk.text,
                    content_hash=content_hash,
                    fetched_at=fetched_at,
                )
                for index, chunk in enumerate(chunks)
            ]

            # Deterministic ids require delete-before-add per document on
            # re-ingest. Only one document's chunks/vectors are held in
            # memory at a time -- at 384-dim float32 a vector is ~1.5KB, so
            # even a page with dozens of chunks stays well under a megabyte.
            self._repository.delete_document(document.uri)
            self._vector_store.delete_document(document.uri)
            vectors = self._embedder.embed_documents([record.text for record in records])
            self._vector_store.add_vectors(records, vectors)
            self._repository.add(records)
            pages_ingested += 1
            chunks_stored += len(records)

        logger.info(
            "Ingest complete for {}: {}/{} pages ingested, {} chunks stored",
            source_url,
            pages_ingested,
            len(documents),
            chunks_stored,
        )
        return IngestSummary(
            pages_discovered=len(documents),
            pages_ingested=pages_ingested,
            chunks_stored=chunks_stored,
        )


class AnswerService:
    """Full RAG answer step: retrieve the query's most relevant chunks, hand
    them to the LLM as numbered sources, and ask it to answer with
    scientific-paper-style [n] citations that index into the returned
    `sources` list. The citation system prompt lives here -- it is domain
    logic, not an LLM detail."""

    _SYSTEM_INSTRUCTION = (
        "You are a research assistant. Answer the user's question using ONLY the "
        "numbered sources provided. Cite every claim with the bracketed number of "
        "the source it comes from, the way a scientific paper cites references -- "
        "e.g. 'Contextual Retrieval reduced failures by 67% [2].' A sentence may "
        "cite multiple sources like [1][3]. Do not use any knowledge beyond the "
        "sources. If the sources do not contain enough information to answer, say "
        "so plainly instead of guessing."
    )

    def __init__(self, search_service: SearchService, llm: LLMClient) -> None:
        self._search_service = search_service
        self._llm = llm

    @observe(name="answer")
    def answer(self, query: str, top_k: int = 5) -> AnswerResult:
        sources = self._search_service.search(query, top_k=top_k)
        if not sources:
            return AnswerResult(
                answer="No relevant sources were found for this query.", sources=[]
            )

        prompt = f"Question: {query}\n\nSources:\n{self._format_sources(sources)}"
        answer = self._llm.generate(prompt, system=self._SYSTEM_INSTRUCTION)
        return AnswerResult(answer=answer, sources=sources)

    @staticmethod
    def _format_sources(sources: list[SearchResult]) -> str:
        blocks = []
        for i, source in enumerate(sources, start=1):
            chunk = source.chunk
            location = f"{chunk.title} > {chunk.heading_path}" if chunk.heading_path else chunk.title
            blocks.append(f"[{i}] {location} ({chunk.uri})\n{chunk.text}")
        return "\n\n".join(blocks)
