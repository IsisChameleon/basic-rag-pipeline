from __future__ import annotations

import chromadb
from chromadb.api import ClientAPI

from rag.config import CHROMA_DIR

_COLLECTION_NAME = "chunks"
_client: ClientAPI | None = None


def get_collection():
    global _client
    if _client is None:
        CHROMA_DIR.parent.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return _client.get_or_create_collection(_COLLECTION_NAME)


def add_chunks(
    ids: list[str],
    embeddings: list[list[float]],
    documents: list[str],
    metadatas: list[dict],
) -> None:
    if not ids:
        return
    get_collection().add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)


def delete_by_url(url: str) -> None:
    get_collection().delete(where={"url": url})


def query(embedding: list[float], n_results: int) -> list[dict]:
    result = get_collection().query(
        query_embeddings=[embedding],
        n_results=n_results,
        include=["documents", "metadatas"],
    )
    ids = result["ids"][0]
    documents = result["documents"][0]
    metadatas = result["metadatas"][0]
    return [
        {"id": chunk_id, "text": document, **metadata}
        for chunk_id, document, metadata in zip(ids, documents, metadatas, strict=True)
    ]
