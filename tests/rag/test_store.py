from rag import store
from rag.store import ChunkMetadata


def test_insert_and_bm25_search_roundtrip() -> None:
    conn = store.get_connection(":memory:")
    store.insert_chunk(
        conn,
        metadata=ChunkMetadata(
            url="https://example.com/a", title="A", heading_path="Intro", chunk_index=0
        ),
        text="the quick brown fox jumps over the lazy dog",
        content_hash="hash-a",
        fetched_at="2026-01-01T00:00:00+00:00",
    )
    conn.commit()

    results = store.search_bm25(conn, "fox", limit=5)
    assert len(results) == 1
    assert results[0]["url"] == "https://example.com/a"


def test_bm25_search_handles_query_with_fts5_special_chars() -> None:
    """Regression: raw user input is a query with FTS5 operators/punctuation --
    a question ending in '?' used to raise 'fts5: syntax error near \"?\"'."""
    conn = store.get_connection(":memory:")
    store.insert_chunk(
        conn,
        metadata=ChunkMetadata(
            url="https://example.com/a", title="A", heading_path="Intro", chunk_index=0
        ),
        text="contextual retrieval reduces failed retrievals",
        content_hash="hash-a",
        fetched_at="2026-01-01T00:00:00+00:00",
    )
    conn.commit()

    # None of these should raise; the first should still find the chunk.
    assert len(store.search_bm25(conn, "what is contextual retrieval?", limit=5)) == 1
    assert store.search_bm25(conn, "error-code TS-999", limit=5) == []
    assert store.search_bm25(conn, "name: value", limit=5) == []
    assert store.search_bm25(conn, "   ", limit=5) == []


def test_delete_chunks_for_url_removes_from_fts_index() -> None:
    """Regression check for the FTS5 external-content sync triggers: without
    them, a delete on `chunks` would silently leave `chunks_fts` stale."""
    conn = store.get_connection(":memory:")
    store.insert_chunk(
        conn,
        metadata=ChunkMetadata(
            url="https://example.com/b", title="B", heading_path="Intro", chunk_index=0
        ),
        text="unique searchable phrase",
        content_hash="hash-b",
        fetched_at="2026-01-01T00:00:00+00:00",
    )
    conn.commit()
    assert store.search_bm25(conn, "searchable", limit=5)

    store.delete_chunks_for_url(conn, "https://example.com/b")

    assert store.search_bm25(conn, "searchable", limit=5) == []
