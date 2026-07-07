from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from rag.config import DB_PATH


@dataclass
class ChunkMetadata:
    """The fields describing a chunk that must stay identical wherever a
    chunk is stored (SQLite `chunks` table and the Chroma vector metadata) --
    a single source of truth instead of two hand-written copies drifting
    apart."""

    url: str
    title: str
    heading_path: str
    chunk_index: int

_SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY,
    url TEXT NOT NULL,
    title TEXT NOT NULL,
    heading_path TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    text TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    fetched_at TEXT NOT NULL
);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text, content=chunks, content_rowid=id
);

CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
END;

CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.id, old.text);
END;

CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.id, old.text);
    INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
END;
"""


def get_connection(db_path: str | None = None) -> sqlite3.Connection:
    path = db_path or str(DB_PATH)
    if path != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    return conn


def delete_chunks_for_url(conn: sqlite3.Connection, url: str) -> None:
    conn.execute("DELETE FROM chunks WHERE url = ?", (url,))
    conn.commit()


def insert_chunk(
    conn: sqlite3.Connection,
    *,
    metadata: ChunkMetadata,
    text: str,
    content_hash: str,
    fetched_at: str,
) -> int:
    cursor = conn.execute(
        """
        INSERT INTO chunks (url, title, heading_path, chunk_index, text, content_hash, fetched_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            metadata.url,
            metadata.title,
            metadata.heading_path,
            metadata.chunk_index,
            text,
            content_hash,
            fetched_at,
        ),
    )
    return cursor.lastrowid


def search_bm25(conn: sqlite3.Connection, query: str, limit: int) -> list[dict]:
    """FTS5 MATCH query, ranked best-first. bm25() scores are negative and
    smaller (more negative) means a better match, hence ORDER BY ... ASC."""
    rows = conn.execute(
        """
        SELECT c.id, c.url, c.title, c.heading_path, c.text, bm25(chunks_fts) AS score
        FROM chunks_fts
        JOIN chunks c ON c.id = chunks_fts.rowid
        WHERE chunks_fts MATCH ?
        ORDER BY score ASC
        LIMIT ?
        """,
        (query, limit),
    ).fetchall()
    return [dict(row) for row in rows]
