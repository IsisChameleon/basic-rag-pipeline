from __future__ import annotations

from dataclasses import dataclass

from rag import embeddings, store, vectorstore

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


def hybrid_search(query: str, top_k: int = 5) -> list[SearchResult]:
    """BM25 (SQLite FTS5) + vector (Chroma) candidates, merged by Reciprocal
    Rank Fusion, then reordered by the cross-encoder reranker."""
    conn = store.get_connection()
    bm25_hits = store.search_bm25(conn, query, limit=_CANDIDATE_POOL_SIZE)
    conn.close()

    query_vector = embeddings.embed_query(query)
    vector_hits = vectorstore.query(query_vector, n_results=_CANDIDATE_POOL_SIZE)

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

    if not fused_scores:
        return []

    candidate_ids = sorted(fused_scores, key=lambda cid: fused_scores[cid], reverse=True)
    candidates = [chunk_data[cid] for cid in candidate_ids[:_CANDIDATE_POOL_SIZE]]

    rerank_scores = embeddings.rerank(query, [c["text"] for c in candidates])
    ranked = sorted(
        zip(candidates, rerank_scores, strict=True), key=lambda pair: pair[1], reverse=True
    )

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
