from __future__ import annotations

from pathlib import Path

import chromadb
from chromadb.api import ClientAPI

from rag.models import ChunkRecord

_COLLECTION_NAME = "chunks"


class VectorStore:
    """Derived, rebuildable similarity index over chunk records, backed by
    Chroma. Vectors are computed by the caller (the Embedder), never by
    Chroma itself. Metadata carries every ChunkRecord field so retrieval can
    return complete records without a lookup in the repository."""

    def __init__(self, persist_dir: str | Path) -> None:
        self._persist_dir = Path(persist_dir)
        self._client: ClientAPI | None = None

    def _collection(self):
        if self._client is None:
            self._persist_dir.parent.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(self._persist_dir))
        return self._client.get_or_create_collection(_COLLECTION_NAME)

    def add_vectors(self, chunks: list[ChunkRecord], vectors: list[list[float]]) -> None:
        if not chunks:
            return
        self._collection().add(
            ids=[c.id for c in chunks],
            embeddings=vectors,
            documents=[c.text for c in chunks],
            metadatas=[c.model_dump(exclude={"id", "text"}) for c in chunks],
        )

    def delete_document(self, uri: str) -> None:
        self._collection().delete(where={"uri": uri})

    def search_similar(self, vector: list[float], limit: int) -> list[ChunkRecord]:
        """Best-first by similarity; order is the only signal the RRF fusion
        downstream uses."""
        result = self._collection().query(
            query_embeddings=[vector],
            n_results=limit,
            include=["documents", "metadatas"],
        )
        return [
            ChunkRecord(id=chunk_id, text=document, **metadata)
            for chunk_id, document, metadata in zip(
                result["ids"][0], result["documents"][0], result["metadatas"][0], strict=True
            )
        ]
