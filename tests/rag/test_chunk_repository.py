"""ChunkRepository tests run against a real SQLite file (tmp_path, not
:memory: -- the repository opens a short-lived connection per operation, and
an in-memory database would vanish between operations).

The FTS5 tests are the critical ones for the TEXT-primary-key schema: FTS5
external-content tables key on the integer rowid, and with `id TEXT PRIMARY
KEY` the chunks table keeps an *implicit* rowid, so the triggers and the
search JOIN must use `rowid`, not `id`.
"""

from pathlib import Path

import pytest

from rag.chunk_repository import ChunkRepository
from rag.models import ChunkRecord


def _record(uri: str, index: int = 0, text: str = "some text") -> ChunkRecord:
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
def repository(tmp_path: Path) -> ChunkRepository:
    return ChunkRepository(tmp_path / "test.db")


def test_add_and_keyword_search_roundtrip(repository: ChunkRepository) -> None:
    stored = _record(
        "https://example.com/a", text="the quick brown fox jumps over the lazy dog"
    )
    repository.add([stored])

    results = repository.search_keyword("fox", limit=5)
    assert len(results) == 1
    assert results[0] == stored  # full ChunkRecord round-trips, not a subset


def test_keyword_search_handles_fts5_special_chars(repository: ChunkRepository) -> None:
    """Regression: raw user input is a query with FTS5 operators/punctuation --
    a question ending in '?' used to raise 'fts5: syntax error near \"?\"'."""
    repository.add(
        [_record("https://example.com/a", text="contextual retrieval reduces failed retrievals")]
    )

    assert len(repository.search_keyword("what is contextual retrieval?", limit=5)) == 1
    assert repository.search_keyword("error-code TS-999", limit=5) == []
    assert repository.search_keyword("name: value", limit=5) == []
    assert repository.search_keyword("   ", limit=5) == []


def test_delete_document_removes_from_fts_index(repository: ChunkRepository) -> None:
    """Regression check for the FTS5 external-content sync triggers on the
    implicit rowid: without them, a delete on `chunks` would silently leave
    `chunks_fts` stale and keyword search would keep returning ghosts."""
    repository.add([_record("https://example.com/b", text="unique searchable phrase")])
    assert repository.search_keyword("searchable", limit=5)

    repository.delete_document("https://example.com/b")

    assert repository.search_keyword("searchable", limit=5) == []


def test_delete_then_readd_same_ids(repository: ChunkRepository) -> None:
    """Re-ingest flow: deterministic ids ({uri}#{index}) mean the same primary
    keys are reinserted after delete_document -- must not violate the TEXT
    PRIMARY KEY constraint or desync the FTS index."""
    uri = "https://example.com/c"
    repository.add([_record(uri, 0, "first version alpha"), _record(uri, 1, "second chunk beta")])
    repository.delete_document(uri)
    repository.add([_record(uri, 0, "first version gamma"), _record(uri, 1, "second chunk beta")])

    assert repository.search_keyword("alpha", limit=5) == []
    hits = repository.search_keyword("gamma", limit=5)
    assert [h.id for h in hits] == [f"{uri}#0"]


def test_search_ranks_better_match_first(repository: ChunkRepository) -> None:
    repository.add(
        [
            _record("https://example.com/d", 0, "retrieval retrieval retrieval systems"),
            _record("https://example.com/e", 0, "a passing mention of retrieval among many other words here"),
        ]
    )

    hits = repository.search_keyword("retrieval", limit=5)
    assert len(hits) == 2
    assert hits[0].uri == "https://example.com/d"
