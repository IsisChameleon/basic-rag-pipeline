from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


class Email(BaseModel):
    """One Gmail message from the digest window, normalised to markdown.

    `sender`, `subject` and `body` are UNTRUSTED -- an attacker picks their
    contents simply by sending you mail. They may only be read by LLM nodes
    with no tools bound; retrieval runs on the summary such a node produced,
    never on these fields directly.
    """

    id: str  # Gmail message id: stable, and the digest's link back to the mail
    sender: str  # raw From header, e.g. 'Jane Doe <jane@example.com>'
    subject: str
    received_at: datetime
    list_unsubscribe: bool  # List-Unsubscribe header present: a newsletter tell
    body: str  # markdown; headings survive so a summary can cite a section


class EmailInProgress(BaseModel):
    """The in-flight carrier: next_item wraps the popped Email in one of
    these, and categorise/summarize fill the fields as the email moves
    through the pipeline. pending itself stays pure list[Email]."""

    email: Email
    category: str | None = None
    topics: list[str] = []
    summary: str | None = None


class ProcessedEmail(BaseModel):
    """Every email ends up as one of these, whatever its category; topics and
    summary only exist for the categories that go through summarize."""

    id: str  # same Gmail message id as Email.id
    received_at: datetime
    category: str  # tightens to a Literal when classify becomes an LLM
    topics: list[str] = []
    summary: str | None = None


class FailedEmail(BaseModel):
    """Dead letter: the raw email plus why the pipeline dropped it."""

    email: Email
    failure: str
