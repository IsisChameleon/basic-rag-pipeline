from __future__ import annotations

from sentence_transformers import CrossEncoder, SentenceTransformer

_BI_ENCODER_NAME = "BAAI/bge-small-en-v1.5"
_CROSS_ENCODER_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# BGE's asymmetric-retrieval convention: prepend this instruction to queries,
# not to documents, per the model card's "Usage" section --
# https://huggingface.co/BAAI/bge-small-en-v1.5#usage. This ST-hub build of
# the model ships empty `.prompts` (verified empirically), so it isn't
# applied automatically -- has to be done by hand.
_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

_bi_encoder: SentenceTransformer | None = None
_cross_encoder: CrossEncoder | None = None


def get_bi_encoder() -> SentenceTransformer:
    global _bi_encoder
    if _bi_encoder is None:
        _bi_encoder = SentenceTransformer(_BI_ENCODER_NAME)
    return _bi_encoder


def get_cross_encoder() -> CrossEncoder:
    global _cross_encoder
    if _cross_encoder is None:
        _cross_encoder = CrossEncoder(_CROSS_ENCODER_NAME)
    return _cross_encoder


def embed_documents(texts: list[str]) -> list[list[float]]:
    model = get_bi_encoder()
    return model.encode(texts, normalize_embeddings=True).tolist()


def embed_query(query: str) -> list[float]:
    model = get_bi_encoder()
    return model.encode(_QUERY_PREFIX + query, normalize_embeddings=True).tolist()


def rerank(query: str, candidates: list[str]) -> list[float]:
    model = get_cross_encoder()
    pairs = [(query, candidate) for candidate in candidates]
    return model.predict(pairs).tolist()


def count_tokens(text: str) -> int:
    """Token counter used by rag.chunk to stay under the bi-encoder's
    max_seq_length (512) -- ties chunk sizing to the model that will actually
    embed the chunks, rather than an approximate word/char heuristic."""
    tokenizer = get_bi_encoder().tokenizer
    return len(tokenizer.encode(text, add_special_tokens=False))
