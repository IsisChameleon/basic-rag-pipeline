"""VectorStore tests run against real embedded Chroma in tmp_path: the class
is a thin wrapper, and the thing worth proving is that a full ChunkRecord
round-trips through Chroma's id/document/metadata split (content_hash and
fetched_at included -- they ride in metadata so retrieval never needs a
repository lookup)."""

from pathlib import Path

import pytest

from rag.models import ChunkRecord
from rag.vector_store import VectorStore


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


@pytest.fixture
def store(tmp_path: Path) -> VectorStore:
    return VectorStore(tmp_path / "chroma")


def test_add_and_search_roundtrips_full_record(store: VectorStore) -> None:
    near = _record("https://example.com/a", 0, "close to the query")
    far = _record("https://example.com/b", 0, "far away")
    store.add_vectors([near, far], [[1.0, 0.0], [0.0, 1.0]])

    results = store.search_similar([0.9, 0.1], limit=2)

    assert results[0] == near  # every field survives the metadata round-trip
    assert [r.id for r in results] == [near.id, far.id]


def test_delete_document_removes_all_its_chunks(store: VectorStore) -> None:
    uri = "https://example.com/a"
    store.add_vectors(
        [_record(uri, 0, "first"), _record(uri, 1, "second"), _record("https://x.com", 0, "keep")],
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]],
    )

    store.delete_document(uri)

    remaining = store.search_similar([1.0, 0.0], limit=5)
    assert [r.uri for r in remaining] == ["https://x.com"]
