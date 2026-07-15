from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class Document(BaseModel):
    """A document from any source, normalised to markdown. `uri` is its
    identity: https://... (crawled), file://... (PDF), upload://... (upload)."""

    uri: str
    title: str
    markdown: str


class Chunk(BaseModel):
    """Chunker output: a piece of a document, not yet stored, no identity."""

    text: str
    heading_path: str


class ChunkRecord(BaseModel):
    """The canonical chunk shape -- what gets stored and what retrieval
    returns. `id` is deterministic: f"{uri}#{chunk_index}", minted by
    IngestService, so it is identical in SQLite and Chroma and never depends
    on either store's row identity."""

    id: str
    uri: str
    title: str
    heading_path: str
    chunk_index: int
    text: str
    content_hash: str
    fetched_at: str


class SearchResult(BaseModel):
    chunk: ChunkRecord
    score: float  # cross-encoder rerank score, higher is more relevant


class AnswerResult(BaseModel):
    answer: str
    # Sources in citation order: sources[0] is reference [1], sources[1] is
    # [2], and so on -- the numbers the model uses in `answer` index into
    # this list.
    sources: list[SearchResult]


class IngestSummary(BaseModel):
    # Field names stay `pages_*` because they are exposed on the public
    # /ingest/{job_id} response; rename to documents_* when a non-web source
    # actually lands.
    pages_discovered: int
    pages_ingested: int
    chunks_stored: int


JobStatus = Literal["pending", "completed", "failed"]


class IngestJob(BaseModel):
    status: JobStatus
    result: IngestSummary | None = None
    error: str | None = None
