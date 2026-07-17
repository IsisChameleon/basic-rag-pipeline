from __future__ import annotations

import sqlite3
from pathlib import Path

from rag.models import ChunkRecord

# `id` is TEXT (deterministic f"{uri}#{chunk_index}", minted by the ingest
# service) so chunk identity no longer depends on SQLite's rowid. FTS5
# external-content tables can only key on an integer rowid; a TEXT primary
# key leaves the table's *implicit* rowid in place, so the FTS table and the
# sync triggers reference `rowid` explicitly (this is the load-bearing detail
# -- see tests/rag/test_chunk_repository.py).
#
# Schema + triggers follow the external-content pattern from the FTS5 docs,
# §4.4.3 External Content Tables: https://sqlite.org/fts5.html#external_content_tables
_SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    uri TEXT NOT NULL,
    title TEXT NOT NULL,
    heading_path TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    text TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    fetched_at TEXT NOT NULL
);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text, content=chunks, content_rowid=rowid
);

CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, text) VALUES (new.rowid, new.text);
END;

CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.rowid, old.text);
END;

CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.rowid, old.text);
    INSERT INTO chunks_fts(rowid, text) VALUES (new.rowid, new.text);
END;
"""

_RECORD_COLUMNS = "id, uri, title, heading_path, chunk_index, text, content_hash, fetched_at"


def _to_fts_match_query(query: str) -> str:
    """Turn free-text user input into a safe FTS5 MATCH expression.

    FTS5 parses the MATCH argument as a query *expression*, not plain text:
    characters like ? : - ( ) " and the words AND/OR/NOT are operators, so raw
    user input (e.g. a question ending in "?") raises `fts5: syntax error`.
    Wrapping each whitespace-separated token in double quotes makes it a literal
    string token, immune to that grammar (embedded quotes are doubled to escape
    them).

    Tokens are joined with OR, not FTS5's implicit AND: a natural-language
    question ("what is contextual retrieval?") would otherwise require *every*
    word -- including stopwords -- to appear in a chunk, so BM25 would return
    nothing for almost any question. With OR, any matching term qualifies a
    chunk and bm25() ranks by how well it matches; the reranker downstream
    sorts out the noise.
    """
    tokens = [token.replace('"', '""') for token in query.split()]
    return " OR ".join(f'"{token}"' for token in tokens)


class ChunkRepository:
    """Source of truth for chunk records, backed by SQLite; also serves
    keyword search via FTS5. Holds the db path, not a connection: each
    operation opens a short-lived connection, which keeps the class safe
    across FastAPI's threadpool and the background ingest thread."""

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = str(db_path)

    def _connect(self) -> sqlite3.Connection:
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        conn.executescript(_SCHEMA)
        return conn

    def add(self, chunks: list[ChunkRecord]) -> None:
        if not chunks:
            return
        conn = self._connect()
        try:
            conn.executemany(
                f"INSERT INTO chunks ({_RECORD_COLUMNS}) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        c.id,
                        c.uri,
                        c.title,
                        c.heading_path,
                        c.chunk_index,
                        c.text,
                        c.content_hash,
                        c.fetched_at,
                    )
                    for c in chunks
                ],
            )
            conn.commit()
        finally:
            conn.close()

    def delete_document(self, uri: str) -> None:
        conn = self._connect()
        try:
            conn.execute("DELETE FROM chunks WHERE uri = ?", (uri,))
            conn.commit()
        finally:
            conn.close()

    def search_keyword(self, query: str, limit: int) -> list[ChunkRecord]:
        """Best-first by BM25 (bm25() scores are negative; more negative is a
        better match, hence ORDER BY ... ASC). Order is the only signal the
        RRF fusion downstream uses."""
        match_query = _to_fts_match_query(query)
        if not match_query:
            return []
        conn = self._connect()
        try:
            rows = conn.execute(
                f"""
                SELECT {", ".join("c." + col for col in _RECORD_COLUMNS.split(", "))},
                       bm25(chunks_fts) AS score
                FROM chunks_fts
                JOIN chunks c ON c.rowid = chunks_fts.rowid
                WHERE chunks_fts MATCH ?
                ORDER BY score ASC
                LIMIT ?
                """,
                (match_query, limit),
            ).fetchall()
        finally:
            conn.close()
        return [
            ChunkRecord(**{key: row[key] for key in row.keys() if key != "score"}) for row in rows
        ]
