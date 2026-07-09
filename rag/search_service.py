from __future__ import annotations

from dataclasses import dataclass

from langfuse import observe

from rag import embeddings, store, vectorstore
from rag.observability import ObservationType

# Reciprocal Rank Fusion constant (standard default) and how many candidates
# each retrieval method contributes before reranking narrows it down.
_RRF_K = 60
_CANDIDATE_POOL_SIZE = 20


@dataclass
class SearchResult:
    text: str
    url: str
    title: str
    heading_path: str
    score: float


# Each stage is its own @observe-decorated function so a Langfuse trace shows one
# span per pipeline stage (sparse -> dense -> fusion -> rerank). When Langfuse is
# not configured (no keys) the decorator is a no-op and these run unchanged.


@observe(name="bm25-retrieve", as_type=ObservationType.RETRIEVER)
def _bm25_retrieve(query: str) -> list[dict]:
    conn = store.get_connection()
    try:
        return store.search_bm25(conn, query, limit=_CANDIDATE_POOL_SIZE)
    finally:
        conn.close()


@observe(name="vector-retrieve", as_type=ObservationType.RETRIEVER)
def _vector_retrieve(query: str) -> list[dict]:
    query_vector = embeddings.embed_query(query)
    return vectorstore.query(query_vector, n_results=_CANDIDATE_POOL_SIZE)


@observe(name="rrf-fuse")
def _rrf_fuse(bm25_hits: list[dict], vector_hits: list[dict]) -> list[dict]:
    fused_scores: dict[str, float] = {}
    chunk_data: dict[str, dict] = {}

    for rank, hit in enumerate(bm25_hits):
        chunk_id = str(hit["id"])
        fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + 1.0 / (_RRF_K + rank + 1)
        chunk_data[chunk_id] = hit

    for rank, hit in enumerate(vector_hits):
        chunk_id = str(hit["id"])
        fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + 1.0 / (_RRF_K + rank + 1)
        chunk_data.setdefault(chunk_id, hit)

    ordered = sorted(fused_scores, key=lambda cid: fused_scores[cid], reverse=True)
    return [chunk_data[cid] for cid in ordered[:_CANDIDATE_POOL_SIZE]]


@observe(name="rerank")
def _rerank(query: str, candidates: list[dict]) -> list[tuple[dict, float]]:
    scores = embeddings.rerank(query, [c["text"] for c in candidates])
    return sorted(zip(candidates, scores, strict=True), key=lambda pair: pair[1], reverse=True)


@observe(name="hybrid-search")
def hybrid_search(query: str, top_k: int = 5) -> list[SearchResult]:
    """BM25 (SQLite FTS5) + vector (Chroma) candidates, merged by Reciprocal
    Rank Fusion, then reordered by the cross-encoder reranker."""
    candidates = _rrf_fuse(_bm25_retrieve(query), _vector_retrieve(query))
    if not candidates:
        return []

    ranked = _rerank(query, candidates)

    return [
        SearchResult(
            text=c["text"],
            url=c["url"],
            title=c["title"],
            heading_path=c.get("heading_path", ""),
            score=float(score),
        )
        for c, score in ranked[:top_k]
    ]
