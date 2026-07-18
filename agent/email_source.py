from __future__ import annotations

import asyncio
import base64
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator

from loguru import logger
from markdownify import markdownify

from agent.models import Email


def _header(headers: list[dict[str, str]], name: str) -> str:
    """Gmail returns headers as a list, and RFC 5322 header names are
    case-insensitive, so neither position nor casing can be relied on."""
    wanted = name.lower()
    for header in headers:
        if header.get("name", "").lower() == wanted:
            return header.get("value", "")
    return ""


def _walk(part: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """A payload is a MIME tree: multipart/mixed wrapping multipart/alternative
    wrapping the text parts is routine, so the text is at no fixed depth."""
    yield part
    for child in part.get("parts", []):
        yield from _walk(child)


def _decode(part: dict[str, Any]) -> str:
    data = part.get("body", {}).get("data")
    if not data:
        return ""  # an attachment or a container part carries no inline data
    return base64.urlsafe_b64decode(data).decode("utf-8", errors="replace")


def _extract_body(payload: dict[str, Any]) -> str:
    """Prefer text/html converted to markdown: a newsletter's text/plain
    alternative is often a stripped-down stub, and the headings only survive
    on the HTML side -- they are what lets a summary cite a section."""
    parts = list(_walk(payload))
    for part in parts:
        if part.get("mimeType") == "text/html":
            html = _decode(part)
            if html:
                return markdownify(html, heading_style="ATX").strip()
    for part in parts:
        if part.get("mimeType") == "text/plain":
            text = _decode(part)
            if text:
                return text.strip()
    return ""


def _to_email(raw: dict[str, Any]) -> Email | None:
    payload = raw.get("payload", {})
    body = _extract_body(payload)
    if not body:
        # Routine, not exceptional: calendar invites and attachment-only mail
        # have no inline text at all.
        logger.debug("Skipping message {} -- no text body", raw.get("id"))
        return None

    headers = payload.get("headers", [])
    return Email(
        id=raw["id"],
        sender=_header(headers, "From"),
        subject=_header(headers, "Subject"),
        # internalDate (epoch ms), not the Date header: Date is written by the
        # sender and may be skewed, malformed, or missing.
        received_at=datetime.fromtimestamp(int(raw["internalDate"]) / 1000, tz=UTC),
        list_unsubscribe=bool(_header(headers, "List-Unsubscribe")),
        body=body,
    )


class GoldenEmailSource:
    """The email set captured by scripts/smoke_gmail.py, replayed from disk:
    same load() shape as EmailSource, zero Gmail calls, identical input every
    run -- for developing the graph without burning API quota on live mail.

    `since` is accepted but ignored on purpose: the capture window was already
    applied when the file was written, and filtering a frozen set against a
    moving "now" would return [] within a day of capturing.
    """

    def __init__(self, path: str | Path = "data/golden_emails.json") -> None:
        self._path = Path(path)

    async def load(self, since: datetime, max_emails: int = 10) -> list[Email]:
        raw = json.loads(self._path.read_text())
        return [Email.model_validate(entry) for entry in raw[:max_emails]]


class EmailSource:
    """Gmail source: lists the message ids in the window, fetches each one, and
    normalises it to an Email. A future IMAP/Outlook source only has to produce
    Emails the same way.

    Takes an already-built googleapiclient Gmail resource. Acquiring the
    credentials behind it belongs to the container, so moving from a local
    token.json to a managed token store changes nothing in here.
    """

    def __init__(self, service: Any) -> None:
        self._service = service

    async def load(self, since: datetime, max_emails: int = 10) -> list[Email]:
        """The most recent up to `max_emails` messages received since `since`.

        `max_emails` caps the messages *fetched* -- each is a separate API call,
        so this bounds the cost against a busy mailbox (200 emails today would
        otherwise be 200 gets). The result can be shorter than the cap: messages
        with no text body are logged and dropped.

        The client is synchronous, so the whole list+get sequence runs in one
        worker thread: it keeps the event loop free without assuming the client
        is safe to share across threads.
        """
        return await asyncio.to_thread(self._load_blocking, since, max_emails)

    def _load_blocking(self, since: datetime, max_emails: int) -> list[Email]:
        messages = self._service.users().messages()
        # Epoch seconds, not YYYY/MM/DD: Gmail reads a date-style query as
        # midnight PST, which would not be a 24h window from here.
        query = f"after:{int(since.timestamp())}"

        # One page suffices: ids come back newest-first, so the newest
        # max_emails are all on the first page -- no cursor to follow. The
        # slice enforces the cap even if the API returns more than maxResults.
        #
        # Gmail paginates messages.list: a page holds up to 500 ids (default
        # 100) and, when more match, the response carries a `nextPageToken` to
        # pass back as `pageToken` for the next page, looping until no token is
        # returned. We deliberately don't, because max_emails (<=500) never
        # spans more than one page. If a cap above 500 is ever needed, restore
        # the loop -- see the pagination example at
        # https://developers.google.com/workspace/gmail/api/guides/list-messages
        response = messages.list(userId="me", q=query, maxResults=max_emails).execute()
        refs = response.get("messages", [])[:max_emails]

        emails: list[Email] = []
        for ref in refs:
            # list returns ids only; each message needs its own get.
            raw = messages.get(userId="me", id=ref["id"], format="full").execute()
            email = _to_email(raw)
            if email is not None:
                emails.append(email)
        logger.info(
            "Loaded {} email(s) from {} fetched since {}",
            len(emails),
            len(refs),
            since.isoformat(),
        )
        return emails
