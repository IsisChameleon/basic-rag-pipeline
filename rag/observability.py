from __future__ import annotations

from enum import StrEnum


class ObservationType(StrEnum):
    """Valid values for `@observe(as_type=...)`.

    Langfuse ships an `ObservationType` enum too, but its values are uppercase
    ("GENERATION") while the `@observe` decorator validates against lowercase
    strings and silently falls back to "span" on a mismatch. This mirrors the
    lowercase set the decorator actually accepts (langfuse 4.x, sans "event").
    """

    GENERATION = "generation"
    SPAN = "span"
    AGENT = "agent"
    TOOL = "tool"
    CHAIN = "chain"
    RETRIEVER = "retriever"
    EMBEDDING = "embedding"
    EVALUATOR = "evaluator"
    GUARDRAIL = "guardrail"
