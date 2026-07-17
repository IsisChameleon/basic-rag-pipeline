from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from api import schemas
from api.dependencies import get_answer_service, get_search_service
from rag.services import AnswerService, SearchService

router = APIRouter(tags=["query"])

# Sync `def` handlers, not `async def`: search is entirely blocking work
# (SQLite, Chroma, sentence-transformers encode/rerank), so FastAPI's worker
# thread pool is what keeps that off the shared event loop -- see
# api/routers/ingest.py for the same reasoning.

# Domain -> wire mapping happens here.


@router.post("/search", response_model=schemas.SearchResponse)
def search(
    request: schemas.SearchRequest,
    search_service: Annotated[SearchService, Depends(get_search_service)],
) -> schemas.SearchResponse:
    results = search_service.search(request.query, top_k=request.top_k)
    return schemas.SearchResponse(
        results=[
            schemas.SearchResult(
                text=r.chunk.text,
                uri=r.chunk.uri,
                title=r.chunk.title,
                heading_path=r.chunk.heading_path,
                score=r.score,
            )
            for r in results
        ]
    )


@router.post("/answer", response_model=schemas.AnswerResponse)
def answer(
    request: schemas.AnswerRequest,
    answer_service: Annotated[AnswerService, Depends(get_answer_service)],
) -> schemas.AnswerResponse:
    # Full RAG: retrieve the query's most relevant chunks, then have the LLM
    # generate an answer that cites them with [n] references. `citations` is
    # in reference order -- citation [n] in `answer` is the n-th entry.
    result = answer_service.answer(request.query, top_k=5)
    citations = [
        schemas.Citation(
            uri=s.chunk.uri, title=s.chunk.title, heading_path=s.chunk.heading_path
        )
        for s in result.sources
    ]
    return schemas.AnswerResponse(answer=result.answer, citations=citations)
