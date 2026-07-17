"""Request/response wire models for every route, consolidated. Routers map
domain objects to these shapes. A chunk's source page is `uri` here to match
the domain naming (ChunkRecord.uri); `IngestRequest.url` stays `url` because
it really is a URL to go fetch, not a stored chunk's identifier."""

from __future__ import annotations

from pydantic import BaseModel


class IngestRequest(BaseModel):
    url: str


class IngestAcceptedResponse(BaseModel):
    job_id: str
    status: str  # always "pending" at creation time


class IngestJobResponse(BaseModel):
    job_id: str
    status: str  # "pending" | "completed" | "failed"
    pages_discovered: int | None = None
    pages_ingested: int | None = None
    chunks_stored: int | None = None
    error: str | None = None


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


class SearchResult(BaseModel):
    text: str  # the chunk's content, e.g. "Contextual Retrieval reduces..."
    uri: str  # the source page, e.g. "https://www.anthropic.com/engineering/contextual-retrieval"
    title: str  # the source page's title, e.g. "Introducing Contextual Retrieval"
    heading_path: str  # the chunk's position in the page's heading structure,
    # e.g. "Introducing Contextual Retrieval > Implementation" for a chunk
    # nested under an "Implementation" subheading; top-level chunks have no
    # " > " separator, e.g. just "Introducing Contextual Retrieval"
    score: float  # rerank score (cross-encoder), higher is more relevant


class SearchResponse(BaseModel):
    results: list[SearchResult]


class AnswerRequest(BaseModel):
    query: str


class Citation(BaseModel):
    uri: str  # e.g. "https://www.anthropic.com/engineering/contextual-retrieval"
    title: str  # e.g. "Introducing Contextual Retrieval"
    heading_path: str  # e.g. "Introducing Contextual Retrieval > Implementation"


class AnswerResponse(BaseModel):
    answer: str
    citations: list[Citation]
