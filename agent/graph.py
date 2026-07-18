from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from loguru import logger

from agent.models import Email, EmailInProgress, FailedEmail, ProcessedEmail


class DigestState(TypedDict):
    """Map-reduce shape: every email leaves pending and is appended -- once,
    complete -- to exactly one of processed or failed. In-flight work never
    lives here; it travels with the worker, so parallel branches have nothing
    shared to conflict over.

    pending has no reducer on purpose: it is consumed (overwritten), and
    operator.add can only grow a list."""

    pending: list[Email]
    current_item: EmailInProgress  # plan A's single in-flight slot; deleted
    # when the loop becomes a Send fan-out (plan B), where it rides the payload
    processed: Annotated[list[ProcessedEmail], operator.add]
    failed: Annotated[list[FailedEmail], operator.add]


def next_item(state: DigestState) -> dict:
    """Pop the queue head and wrap it in the in-flight carrier: this is the
    one point where raw mail becomes a work item. Only runs when pending is
    non-empty: route_check_end_or_next guards every edge into this node."""
    pending = state["pending"]
    return {
        "current_item": EmailInProgress(email=pending[0]),
        "pending": pending[1:],
    }


def categorise(state: DigestState) -> dict:
    # STUB for the first quarantined LLM (no tools): decides
    # other / security_risk / newsletter over the raw body. The
    # List-Unsubscribe flag routes the golden set down both main branches;
    # the stub can never say security_risk.
    item = state["current_item"]
    label = "newsletter" if item.email.list_unsubscribe else "other"
    return {"current_item": item.model_copy(update={"category": label})}


def security_risk(state: DigestState) -> dict:
    logger.warning(
        "Security risk mail dropped: {} from {}",
        state["current_item"].email.subject,
        state["current_item"].email.sender,
    )
    return {}


def add_to_processed(state: DigestState) -> dict:
    """Every path ends here: mint the ProcessedEmail from the carrier and
    append it (the delta only -- operator.add does the rest). This is the one
    place an email is declared done, so counts always add up."""
    item = state["current_item"]
    return {
        "processed": [
            ProcessedEmail(
                id=item.email.id,
                received_at=item.email.received_at,
                category=item.category,
                topics=item.topics,
                summary=item.summary,
            )
        ]
    }


def summarize(state: DigestState) -> dict:
    # STUB for the second quarantined LLM (no tools): summary + topics over
    # the raw body. The subject is the cheapest possible stand-in for both.
    item = state["current_item"]
    return {
        "current_item": item.model_copy(
            update={"topics": [item.email.subject], "summary": item.email.subject}
        )
    }


def rag(state: DigestState) -> dict:
    # Not dictated yet: relevance check of topics against the corpus.
    return {}


def route_check_end_or_next(state: DigestState) -> str:
    """The drain check, BEFORE popping: current_email is never cleared, so
    emptiness must be judged on pending alone."""
    return "next_item" if state["pending"] else END


def route_category(state: DigestState) -> str:
    category = state["current_item"].category
    if category == "newsletter":
        return "summarize"
    if category == "security_risk":
        return "security_risk"
    # "other": recorded in processed (category tells it apart), then the
    # drain check runs from there.
    return "add_to_processed"


def build_digest_graph():
    graph = StateGraph(DigestState)
    graph.add_node("next_item", next_item)
    graph.add_node("categorise", categorise)
    graph.add_node("security_risk", security_risk)
    graph.add_node("summarize", summarize)
    graph.add_node("rag", rag)
    graph.add_node("add_to_processed", add_to_processed)

    graph.add_conditional_edges(START, route_check_end_or_next)
    graph.add_edge("next_item", "categorise")
    graph.add_conditional_edges("categorise", route_category)
    graph.add_edge("security_risk", "add_to_processed")
    graph.add_edge("summarize", "rag")
    graph.add_edge("rag", "add_to_processed")
    graph.add_conditional_edges("add_to_processed", route_check_end_or_next)
    return graph.compile()
