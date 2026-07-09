from rag import search_service


class _FakeConn:
    def close(self) -> None:
        pass


def test_hybrid_search_fuses_and_reranks(monkeypatch):
    bm25 = [{"id": "a", "text": "alpha", "url": "u", "title": "t", "heading_path": ""}]
    vec = [{"id": "b", "text": "beta", "url": "u", "title": "t", "heading_path": ""}]

    monkeypatch.setattr(search_service.store, "get_connection", lambda: _FakeConn())
    monkeypatch.setattr(search_service.store, "search_bm25", lambda c, q, limit: bm25)
    monkeypatch.setattr(search_service.embeddings, "embed_query", lambda q: [0.0])
    monkeypatch.setattr(search_service.vectorstore, "query", lambda v, n_results: vec)
    # rerank scores beta above alpha so the output order is deterministic
    monkeypatch.setattr(
        search_service.embeddings,
        "rerank",
        lambda q, texts: [2.0 if t == "beta" else 1.0 for t in texts],
    )

    results = search_service.hybrid_search("query", top_k=2)

    assert [r.text for r in results] == ["beta", "alpha"]
    assert results[0].score == 2.0


def test_hybrid_search_empty_when_no_candidates(monkeypatch):
    monkeypatch.setattr(search_service.store, "get_connection", lambda: _FakeConn())
    monkeypatch.setattr(search_service.store, "search_bm25", lambda c, q, limit: [])
    monkeypatch.setattr(search_service.embeddings, "embed_query", lambda q: [0.0])
    monkeypatch.setattr(search_service.vectorstore, "query", lambda v, n_results: [])

    assert search_service.hybrid_search("query") == []
