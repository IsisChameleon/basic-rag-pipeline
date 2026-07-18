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
