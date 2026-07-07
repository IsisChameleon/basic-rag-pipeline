from __future__ import annotations

from dataclasses import dataclass

from google.genai import Client, types

from rag import config
from rag.search_service import SearchResult, hybrid_search

_client: Client | None = None

_SYSTEM_INSTRUCTION = (
    "You are a research assistant. Answer the user's question using ONLY the "
    "numbered sources provided. Cite every claim with the bracketed number of "
    "the source it comes from, the way a scientific paper cites references -- "
    "e.g. 'Contextual Retrieval reduced failures by 67% [2].' A sentence may "
    "cite multiple sources like [1][3]. Do not use any knowledge beyond the "
    "sources. If the sources do not contain enough information to answer, say "
    "so plainly instead of guessing."
)


@dataclass
class AnswerResult:
    answer: str
    # Sources in citation order: sources[0] is reference [1], sources[1] is [2],
    # and so on -- the numbers the model uses in `answer` index into this list.
    sources: list[SearchResult]


def get_client() -> Client:
    global _client
    if _client is None:
        if not config.GOOGLE_API_KEY:
            raise RuntimeError(
                "GOOGLE_API_KEY is not set -- /answer needs it to call Gemini. "
                "Add it to your local .env (see .env.example)."
            )
        _client = Client(api_key=config.GOOGLE_API_KEY)
    return _client


def _format_sources(sources: list[SearchResult]) -> str:
    blocks = []
    for i, s in enumerate(sources, start=1):
        location = f"{s.title} > {s.heading_path}" if s.heading_path else s.title
        blocks.append(f"[{i}] {location} ({s.url})\n{s.text}")
    return "\n\n".join(blocks)


def generate_answer(query: str, top_k: int = 5) -> AnswerResult:
    """Full RAG answer step: embed + retrieve the query's most relevant chunks
    (hybrid_search), hand them to Gemini as numbered sources, and ask it to
    answer with scientific-paper-style [n] citations that index into the
    returned `sources` list."""
    sources = hybrid_search(query, top_k=top_k)
    if not sources:
        return AnswerResult(answer="No relevant sources were found for this query.", sources=[])

    prompt = f"Question: {query}\n\nSources:\n{_format_sources(sources)}"

    response = get_client().models.generate_content(
        model=config.LLM_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=_SYSTEM_INSTRUCTION,
            temperature=0.0,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    return AnswerResult(answer=response.text or "", sources=sources)
