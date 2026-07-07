from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from rag.answer_service import generate_answer
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
    # Full RAG: retrieve the query's most relevant chunks, then have Gemini
    # generate an answer that cites them with [n] references. `citations` is
    # in reference order -- citation [n] in `answer` is the n-th entry.
    result = generate_answer(request.query, top_k=5)
    citations = [
        Citation(url=s.url, title=s.title, heading_path=s.heading_path) for s in result.sources
    ]
    return AnswerResponse(answer=result.answer, citations=citations)
