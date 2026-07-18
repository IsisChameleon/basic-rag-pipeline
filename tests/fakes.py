"""In-memory stand-ins for the external boundaries (stores, encoders, LLM,
web source). They mirror the real classes' method signatures structurally --
no inheritance, no mock.patch -- and preserve real data semantics (e.g. the
fake stores really hold and return ChunkRecords), per the repo's testing
discipline: mock at external boundaries, keep our own logic real."""

from __future__ import annotations

from rag.models import ChunkRecord, Document


class FakeChunkRepository:
    def __init__(self) -> None:
        self.records: list[ChunkRecord] = []

    def add(self, chunks: list[ChunkRecord]) -> None:
        self.records.extend(chunks)

    def delete_document(self, uri: str) -> None:
        self.records = [r for r in self.records if r.uri != uri]

    def search_keyword(self, query: str, limit: int) -> list[ChunkRecord]:
        tokens = query.lower().split()
        hits = [r for r in self.records if any(t in r.text.lower() for t in tokens)]
        return hits[:limit]


class FakeVectorStore:
    def __init__(self) -> None:
        self.records: list[ChunkRecord] = []
        self.vectors: dict[str, list[float]] = {}

    def add_vectors(self, chunks: list[ChunkRecord], vectors: list[list[float]]) -> None:
        assert len(chunks) == len(vectors)
        self.records.extend(chunks)
        self.vectors.update({c.id: v for c, v in zip(chunks, vectors, strict=True)})

    def delete_document(self, uri: str) -> None:
        removed = [r.id for r in self.records if r.uri == uri]
        self.records = [r for r in self.records if r.uri != uri]
        for chunk_id in removed:
            self.vectors.pop(chunk_id, None)

    def search_similar(self, vector: list[float], limit: int) -> list[ChunkRecord]:
        # Insertion order stands in for similarity order; tests control it.
        return self.records[:limit]


class FakeEmbedder:
    """Deterministic 2-dim 'embeddings'; also a plausible token counter."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(t)), 1.0] for t in texts]

    def embed_query(self, query: str) -> list[float]:
        return [float(len(query)), 1.0]

    def count_tokens(self, text: str) -> int:
        return len(text.split())


class FakeReranker:
    """Scores by naive term overlap with the query -- enough to make rerank
    ordering observable and controllable from test data."""

    def rerank(self, query: str, candidates: list[str]) -> list[float]:
        query_tokens = set(query.lower().split())
        return [
            float(len(query_tokens & set(candidate.lower().split())))
            for candidate in candidates
        ]


class FakeLLMClient:
    def __init__(self, response: str = "a generated answer [1]") -> None:
        self.response = response
        self.prompts: list[str] = []
        self.systems: list[str] = []

    def generate(self, prompt: str, *, system: str) -> str:
        self.prompts.append(prompt)
        self.systems.append(system)
        return self.response


class FakeDocumentSource:
    def __init__(
        self, documents: list[Document | None], error: Exception | None = None
    ) -> None:
        self.documents = documents
        self.error = error
        self.loaded_urls: list[str] = []

    async def load(self, source_url: str) -> list[Document | None]:
        self.loaded_urls.append(source_url)
        if self.error is not None:
            raise self.error
        return self.documents


class _FakeRequest:
    """googleapiclient builds a request object you then .execute()."""

    def __init__(self, result: dict) -> None:
        self.result = result

    def execute(self) -> dict:
        return self.result


class FakeGmailMessages:
    """The .users().messages() resource: list returns one page of ids, get
    returns one raw message, exactly as Gmail does.

    `listing` is the single list response ({"messages": [{"id": ...}, ...]});
    `messages` maps message id to its get response. Records the `maxResults`
    each list call passes so a test can assert the cap reached the API.
    """

    def __init__(self, listing: dict, messages: dict[str, dict]) -> None:
        self.listing = listing
        self.messages = messages
        self.queries: list[str] = []
        self.max_results: list[int] = []
        self.fetched_ids: list[str] = []

    def list(self, *, userId: str, q: str, maxResults: int) -> _FakeRequest:
        self.queries.append(q)
        self.max_results.append(maxResults)
        return _FakeRequest(self.listing)

    def get(self, *, userId: str, id: str, format: str) -> _FakeRequest:
        self.fetched_ids.append(id)
        return _FakeRequest(self.messages[id])


class FakeGmailService:
    """Stands in for the built googleapiclient Gmail resource -- the external
    boundary. Mirrors the chained .users().messages() shape, so EmailSource's
    own MIME walk, header lookup, cap slicing and query building all still
    really run."""

    def __init__(self, listing: dict, messages: dict[str, dict]) -> None:
        self.resource = FakeGmailMessages(listing, messages)

    def users(self) -> FakeGmailService:
        return self

    def messages(self) -> FakeGmailMessages:
        return self.resource


def build_fake_container(
    *,
    records: list[ChunkRecord] | None = None,
    documents: list[Document | None] | None = None,
    source_error: Exception | None = None,
    llm_response: str = "a generated answer [1]",
):
    """An ApiContainer wired entirely from fakes (plus our real chunker and
    services), for API tests: `app.state.container = build_fake_container(...)`.
    `records` pre-populates retrieval; `documents`/`source_error` script the
    ingest source."""
    from api.container import ApiContainer
    from core.settings import Settings
    from rag.container import RagContainer
    from rag.job_store import JobStore
    from rag.markdown_chunker import MarkdownChunker
    from rag.services import AnswerService, IngestService, SearchService

    repository = FakeChunkRepository()
    repository.add(records or [])
    vector_store = FakeVectorStore()
    embedder = FakeEmbedder()
    reranker = FakeReranker()
    search_service = SearchService(repository, vector_store, embedder, reranker)
    return ApiContainer(
        settings=Settings(),
        rag=RagContainer(
            search_service=search_service,
            answer_service=AnswerService(search_service, FakeLLMClient(response=llm_response)),
            ingest_service=IngestService(
                FakeDocumentSource(documents or [], error=source_error),
                MarkdownChunker(count_tokens=embedder.count_tokens),
                embedder,
                repository,
                vector_store,
            ),
            embedder=embedder,
            reranker=reranker,
        ),
        job_store=JobStore(),
    )
