from types import SimpleNamespace

from rag import answer_service
from rag.search_service import SearchResult


def _sources() -> list[SearchResult]:
    return [
        SearchResult(
            text="Contextual Retrieval reduced failed retrievals by 67%.",
            url="https://example.com/contextual-retrieval",
            title="Contextual Retrieval",
            heading_path="Introducing Contextual Retrieval > Performance",
            score=0.9,
        ),
        SearchResult(
            text="BM25 uses lexical matching for exact terms.",
            url="https://example.com/contextual-retrieval",
            title="Contextual Retrieval",
            heading_path="A primer on RAG",
            score=0.7,
        ),
    ]


def test_format_sources_numbers_from_one() -> None:
    block = answer_service._format_sources(_sources())
    assert block.startswith("[1] Contextual Retrieval > Introducing Contextual Retrieval")
    assert "[2] Contextual Retrieval > A primer on RAG" in block


def test_generate_answer_passes_numbered_sources_and_returns_them_in_order(monkeypatch) -> None:
    sent = {}

    class FakeModels:
        def generate_content(self, *, model, contents, config):
            sent["model"] = model
            sent["prompt"] = contents
            return SimpleNamespace(text="Retrieval improves accuracy [1][2].")

    monkeypatch.setattr(answer_service, "hybrid_search", lambda query, top_k=5: _sources())
    monkeypatch.setattr(
        answer_service, "get_client_gemini", lambda: SimpleNamespace(models=FakeModels())
    )

    result = answer_service.generate_answer("How does contextual retrieval help?")

    assert result.answer == "Retrieval improves accuracy [1][2]."
    # sources come back in citation order: [1] is sources[0], [2] is sources[1]
    assert [s.heading_path for s in result.sources] == [
        "Introducing Contextual Retrieval > Performance",
        "A primer on RAG",
    ]
    # the model was handed the numbered sources and the question
    assert "[1]" in sent["prompt"] and "[2]" in sent["prompt"]
    assert "How does contextual retrieval help?" in sent["prompt"]
    assert sent["model"] == "gemini-2.5-flash"


def test_generate_answer_skips_llm_when_no_sources(monkeypatch) -> None:
    def _fail() -> None:
        raise AssertionError("Gemini must not be called when retrieval is empty")

    monkeypatch.setattr(answer_service, "hybrid_search", lambda query, top_k=5: [])
    monkeypatch.setattr(answer_service, "get_client_gemini", _fail)

    result = answer_service.generate_answer("nothing matches this")

    assert result.sources == []
    assert "No relevant sources" in result.answer
