from __future__ import annotations

from sentence_transformers import CrossEncoder, SentenceTransformer

# BGE's asymmetric-retrieval convention: prepend this instruction to queries,
# not to documents, per the model card's "Usage" section --
# https://huggingface.co/BAAI/bge-small-en-v1.5#usage. This ST-hub build of
# the model ships empty `.prompts` (verified empirically), so it isn't
# applied automatically -- has to be done by hand.
_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder:
    """Bi-encoder embeddings (sentence-transformers). The constructor is
    cheap; the ~8s model load happens lazily on first use, or eagerly via
    warm_up() where the composition root wants boot-time loading. Lazy first
    load is not thread-safe -- the API calls warm_up() in the lifespan,
    before any request thread exists."""

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._model: SentenceTransformer | None = None

    def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self._model_name)
        return self._model

    def warm_up(self) -> None:
        self._get_model()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._get_model().encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, query: str) -> list[float]:
        return self._get_model().encode(_QUERY_PREFIX + query, normalize_embeddings=True).tolist()

    def count_tokens(self, text: str) -> int:
        """Token counter used by MarkdownChunker to stay under the
        bi-encoder's max_seq_length (512) -- ties chunk sizing to the model
        that will actually embed the chunks, rather than an approximate
        word/char heuristic."""
        tokenizer = self._get_model().tokenizer
        return len(tokenizer.encode(text, add_special_tokens=False))


class Reranker:
    """Cross-encoder reranker; independently swappable from the Embedder
    (different model, e.g. a hosted rerank API later). Same lazy-load and
    warm_up() contract as Embedder."""

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._model: CrossEncoder | None = None

    def _get_model(self) -> CrossEncoder:
        if self._model is None:
            self._model = CrossEncoder(self._model_name)
        return self._model

    def warm_up(self) -> None:
        self._get_model()

    def rerank(self, query: str, candidates: list[str]) -> list[float]:
        pairs = [(query, candidate) for candidate in candidates]
        return self._get_model().predict(pairs).tolist()
