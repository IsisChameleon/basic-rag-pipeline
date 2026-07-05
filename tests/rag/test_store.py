from rag import store


def test_insert_and_bm25_search_roundtrip() -> None:
    conn = store.get_connection(":memory:")
    store.insert_chunk(
        conn,
        url="https://example.com/a",
        title="A",
        heading_path="Intro",
        chunk_index=0,
        text="the quick brown fox jumps over the lazy dog",
        content_hash="hash-a",
        fetched_at="2026-01-01T00:00:00+00:00",
    )
    conn.commit()

    results = store.search_bm25(conn, "fox", limit=5)
    assert len(results) == 1
    assert results[0]["url"] == "https://example.com/a"


def test_delete_chunks_for_url_removes_from_fts_index() -> None:
    """Regression check for the FTS5 external-content sync triggers: without
    them, a delete on `chunks` would silently leave `chunks_fts` stale."""
    conn = store.get_connection(":memory:")
    store.insert_chunk(
        conn,
        url="https://example.com/b",
        title="B",
        heading_path="Intro",
        chunk_index=0,
        text="unique searchable phrase",
        content_hash="hash-b",
        fetched_at="2026-01-01T00:00:00+00:00",
    )
    conn.commit()
    assert store.search_bm25(conn, "searchable", limit=5)

    store.delete_chunks_for_url(conn, "https://example.com/b")

    assert store.search_bm25(conn, "searchable", limit=5) == []
