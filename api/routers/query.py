from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from rag.search_service import hybrid_search

router = APIRouter(tags=["query"])


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


class SearchResult(BaseModel):
    text: str  # the chunk's content, e.g. "Contextual Retrieval reduces..."
    url: str  # the source page, e.g. "https://www.anthropic.com/engineering/contextual-retrieval"
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
    url: str  # e.g. "https://www.anthropic.com/engineering/contextual-retrieval"
    title: str  # e.g. "Introducing Contextual Retrieval"
    heading_path: str  # e.g. "Introducing Contextual Retrieval > Implementation"


class AnswerResponse(BaseModel):
    answer: str
    citations: list[Citation]


# Sync `def` handlers, not `async def`: hybrid_search is entirely blocking
# work (SQLite, Chroma, sentence-transformers encode/rerank), so FastAPI's
# worker thread pool is what keeps that off the shared event loop -- see
# api/routers/ingest.py for the same reasoning.


@router.post("/search", response_model=SearchResponse)
def search(request: SearchRequest) -> SearchResponse:
    results = hybrid_search(request.query, top_k=request.top_k)
    return SearchResponse(
        results=[
            SearchResult(
                text=r.text,
                url=r.url,
                title=r.title,
                heading_path=r.heading_path,
                score=r.score,
            )
            for r in results
        ]
    )


@router.post("/answer", response_model=AnswerResponse)
def answer(request: AnswerRequest) -> AnswerResponse:
    # Retrieval is fully implemented; the LLM provider for answer generation
    # is still an open decision (see build_log.md), so `answer` stays empty
    # for now while `citations` reflects real retrieval results.
    results = hybrid_search(request.query, top_k=5)
    citations = [Citation(url=r.url, title=r.title, heading_path=r.heading_path) for r in results]
    return AnswerResponse(answer="", citations=citations)
