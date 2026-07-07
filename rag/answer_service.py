from __future__ import annotations

from dataclasses import dataclass

from google.genai import Client, types
from langfuse import get_client, observe

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


def get_client_gemini() -> Client:
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


@observe(as_type="generation", name="gemini-generate")
def _call_gemini(prompt: str) -> str:
    response = get_client_gemini().models.generate_content(
        model=config.LLM_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=_SYSTEM_INSTRUCTION,
            temperature=0.0,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    # Record model + token usage on the generation span when Langfuse is active.
    # google-genai exposes counts on response.usage_metadata; skip if absent
    # (e.g. the fake client used in tests) -- update_current_generation no-ops
    # when tracing is disabled.
    usage = getattr(response, "usage_metadata", None)
    if usage is not None:
        get_client().update_current_generation(
            model=config.LLM_MODEL,
            usage_details={
                "input": getattr(usage, "prompt_token_count", None),
                "output": getattr(usage, "candidates_token_count", None),
            },
        )
    return response.text or ""


@observe(name="answer")
def generate_answer(query: str, top_k: int = 5) -> AnswerResult:
    """Full RAG answer step: embed + retrieve the query's most relevant chunks
    (hybrid_search), hand them to Gemini as numbered sources, and ask it to
    answer with scientific-paper-style [n] citations that index into the
    returned `sources` list."""
    sources = hybrid_search(query, top_k=top_k)
    if not sources:
        return AnswerResult(answer="No relevant sources were found for this query.", sources=[])

    prompt = f"Question: {query}\n\nSources:\n{_format_sources(sources)}"
    return AnswerResult(answer=_call_gemini(prompt), sources=sources)
