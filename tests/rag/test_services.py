"""Service tests: real service classes + real MarkdownChunker (our own
logic stays real), fakes only at the external boundaries. No mock.patch --
everything arrives through constructors."""

from __future__ import annotations

import asyncio

from rag.markdown_chunker import MarkdownChunker
from rag.models import ChunkRecord, Document
from rag.services import AnswerService, IngestService, SearchService
from tests.fakes import (
    FakeChunkRepository,
    FakeDocumentSource,
    FakeEmbedder,
    FakeLLMClient,
    FakeReranker,
    FakeVectorStore,
)


def _record(uri: str, index: int, text: str) -> ChunkRecord:
    return ChunkRecord(
        id=f"{uri}#{index}",
        uri=uri,
        title="Title",
        heading_path="Intro",
        chunk_index=index,
        text=text,
        content_hash="hash",
        fetched_at="2026-01-01T00:00:00+00:00",
    )


def _search_service(
    repository: FakeChunkRepository | None = None,
    vector_store: FakeVectorStore | None = None,
) -> SearchService:
    return SearchService(
        repository or FakeChunkRepository(),
        vector_store or FakeVectorStore(),
        FakeEmbedder(),
        FakeReranker(),
    )


class TestSearchService:
    def test_returns_reranked_results_capped_at_top_k(self) -> None:
        repository = FakeChunkRepository()
        repository.add(
            [
                _record("https://a", 0, "nothing relevant here"),
                _record("https://b", 0, "contextual retrieval details"),
                _record("https://c", 0, "more about contextual retrieval and retrieval quality"),
            ]
        )
        service = _search_service(repository=repository)

        results = service.search("contextual retrieval", top_k=2)

        assert len(results) == 2
        # FakeReranker scores by term overlap, so the two chunks mentioning
        # the query terms outrank the irrelevant one.
        assert {r.chunk.uri for r in results} == {"https://b", "https://c"}
        assert results[0].score >= results[1].score

    def test_fuses_keyword_and_vector_hits_without_duplicates(self) -> None:
        shared = _record("https://a", 0, "appears in both retrievers")
        vector_only = _record("https://b", 0, "vector only appears here")
        repository = FakeChunkRepository()
        repository.add([shared])
        vector_store = FakeVectorStore()
        vector_store.add_vectors([shared, vector_only], [[1.0, 0.0], [0.0, 1.0]])
        service = _search_service(repository=repository, vector_store=vector_store)

        results = service.search("appears", top_k=5)

        ids = [r.chunk.id for r in results]
        assert sorted(ids) == sorted({*ids})  # dedup by record.id: no chunk appears twice
        assert set(ids) == {shared.id, vector_only.id}

    def test_returns_empty_when_nothing_matches(self) -> None:
        assert _search_service().search("anything", top_k=5) == []


class TestIngestService:
    def _service(
        self, documents: list[Document | None]
    ) -> tuple[IngestService, FakeChunkRepository, FakeVectorStore]:
        repository = FakeChunkRepository()
        vector_store = FakeVectorStore()
        embedder = FakeEmbedder()
        service = IngestService(
            FakeDocumentSource(documents),
            MarkdownChunker(count_tokens=embedder.count_tokens),
            embedder,
            repository,
            vector_store,
        )
        return service, repository, vector_store

    def test_ingests_documents_and_counts_failures(self) -> None:
        documents = [
            Document(uri="https://a", title="A", markdown="# One\n\nalpha beta gamma"),
            None,  # a page whose fetch failed
            Document(uri="https://b", title="B", markdown="# Two\n\ndelta epsilon"),
        ]
        service, repository, vector_store = self._service(documents)

        summary = asyncio.run(service.ingest("https://site/docs"))

        assert summary.pages_discovered == 3
        assert summary.pages_ingested == 2
        assert summary.chunks_stored == len(repository.records)
        assert [r.id for r in repository.records] == ["https://a#0", "https://b#0"]
        # Both stores hold the same records under the same deterministic ids.
        assert {r.id for r in vector_store.records} == {r.id for r in repository.records}
        assert repository.records[0].heading_path == "One"
        assert repository.records[0].content_hash  # sha256 of the markdown

    def test_reingest_replaces_instead_of_duplicating(self) -> None:
        documents = [Document(uri="https://a", title="A", markdown="# One\n\nalpha beta")]
        service, repository, vector_store = self._service(documents)

        asyncio.run(service.ingest("https://site/docs"))
        asyncio.run(service.ingest("https://site/docs"))

        assert [r.id for r in repository.records] == ["https://a#0"]
        assert [r.id for r in vector_store.records] == ["https://a#0"]


class TestAnswerService:
    def test_answers_with_numbered_sources_in_prompt(self) -> None:
        repository = FakeChunkRepository()
        repository.add([_record("https://a", 0, "contextual retrieval explained")])
        llm = FakeLLMClient(response="an answer [1]")
        service = AnswerService(_search_service(repository=repository), llm)

        result = service.answer("contextual retrieval", top_k=3)

        assert result.answer == "an answer [1]"
        assert [s.chunk.uri for s in result.sources] == ["https://a"]
        prompt = llm.prompts[0]
        assert "Question: contextual retrieval" in prompt
        assert "[1] Title > Intro (https://a)" in prompt
        assert "contextual retrieval explained" in prompt
        assert "cite" in llm.systems[0].lower()

    def test_no_sources_short_circuits_without_calling_llm(self) -> None:
        llm = FakeLLMClient()
        service = AnswerService(_search_service(), llm)

        result = service.answer("anything")

        assert result.sources == []
        assert "No relevant sources" in result.answer
        assert llm.prompts == []  # the LLM was never invoked
