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
    def __init__(self, documents: list[Document | None]) -> None:
        self.documents = documents
        self.loaded_urls: list[str] = []

    async def load(self, source_url: str) -> list[Document | None]:
        self.loaded_urls.append(source_url)
        return self.documents
