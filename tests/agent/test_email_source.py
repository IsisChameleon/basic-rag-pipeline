from __future__ import annotations

import base64
import json
from datetime import UTC, datetime
from pathlib import Path

from agent.email_source import EmailSource, GoldenEmailSource
from agent.models import Email
from tests.fakes import FakeGmailService

SINCE = datetime(2026, 7, 16, 9, 0, 0, tzinfo=UTC)
RECEIVED = datetime(2026, 7, 16, 14, 30, 0, tzinfo=UTC)


def encode(text: str) -> str:
    return base64.urlsafe_b64encode(text.encode()).decode()


def raw_message(message_id: str, *, payload: dict, received_at: datetime = RECEIVED) -> dict:
    # Gmail's internalDate is epoch milliseconds, as a string.
    internal_date = str(int(received_at.timestamp() * 1000))
    return {"id": message_id, "internalDate": internal_date, "payload": payload}


def headers(**named: str) -> list[dict[str, str]]:
    return [{"name": name, "value": value} for name, value in named.items()]


def listing(*message_ids: str) -> dict:
    return {"messages": [{"id": i} for i in message_ids]}


async def test_prefers_html_part_and_keeps_headings_as_markdown():
    payload = {
        "mimeType": "multipart/alternative",
        "headers": headers(From="News <hi@news.com>", Subject="Weekly"),
        "parts": [
            {"mimeType": "text/plain", "body": {"data": encode("View in browser")}},
            {
                "mimeType": "text/html",
                "body": {"data": encode("<h2>ANN indexing</h2><p>HNSW vs IVF</p>")},
            },
        ],
    }
    service = FakeGmailService(listing("m1"), {"m1": raw_message("m1", payload=payload)})

    emails = await EmailSource(service).load(SINCE)

    assert len(emails) == 1
    # The heading survives as markdown -- this is what lets a summary say
    # "read section 2" -- and the text/plain stub was not used.
    assert "## ANN indexing" in emails[0].body
    assert "HNSW vs IVF" in emails[0].body
    assert "View in browser" not in emails[0].body


async def test_falls_back_to_plain_text_when_there_is_no_html_part():
    payload = {
        "mimeType": "text/plain",
        "headers": headers(From="Jane <jane@x.com>", Subject="Lunch?"),
        "body": {"data": encode("Free at 1pm?")},
    }
    service = FakeGmailService(listing("m1"), {"m1": raw_message("m1", payload=payload)})

    emails = await EmailSource(service).load(SINCE)

    assert emails[0].body == "Free at 1pm?"


async def test_finds_text_nested_below_the_top_level_parts():
    payload = {
        "mimeType": "multipart/mixed",
        "headers": headers(From="Jane <jane@x.com>", Subject="Report"),
        "parts": [
            {
                "mimeType": "multipart/alternative",
                "parts": [{"mimeType": "text/html", "body": {"data": encode("<p>Body</p>")}}],
            },
            {"mimeType": "application/pdf", "body": {"attachmentId": "a1"}},
        ],
    }
    service = FakeGmailService(listing("m1"), {"m1": raw_message("m1", payload=payload)})

    emails = await EmailSource(service).load(SINCE)

    assert emails[0].body == "Body"


async def test_reads_metadata_from_headers_and_internal_date():
    payload = {
        "mimeType": "text/plain",
        # Lower-cased header names are legal, and List-Unsubscribe is the
        # newsletter tell.
        "headers": headers(**{"from": "News <hi@news.com>", "subject": "Weekly"})
        + [{"name": "List-Unsubscribe", "value": "<https://news.com/u>"}],
        "body": {"data": encode("Hello")},
    }
    service = FakeGmailService(listing("m1"), {"m1": raw_message("m1", payload=payload)})

    emails = await EmailSource(service).load(SINCE)

    assert emails[0].id == "m1"
    assert emails[0].sender == "News <hi@news.com>"
    assert emails[0].subject == "Weekly"
    assert emails[0].list_unsubscribe is True
    assert emails[0].received_at == RECEIVED


async def test_list_unsubscribe_is_false_when_the_header_is_absent():
    payload = {
        "mimeType": "text/plain",
        "headers": headers(From="Jane <jane@x.com>", Subject="Lunch?"),
        "body": {"data": encode("Free at 1pm?")},
    }
    service = FakeGmailService(listing("m1"), {"m1": raw_message("m1", payload=payload)})

    emails = await EmailSource(service).load(SINCE)

    assert emails[0].list_unsubscribe is False


async def test_queries_gmail_with_exact_epoch_seconds():
    service = FakeGmailService({"messages": []}, {})

    await EmailSource(service).load(SINCE)

    # Epoch seconds, not YYYY/MM/DD: a date-style query would be read as
    # midnight PST rather than the 24h window asked for.
    assert service.resource.queries == [f"after:{int(SINCE.timestamp())}"]


async def test_caps_the_number_of_messages_fetched():
    payload = {
        "mimeType": "text/plain",
        "headers": headers(From="Jane <jane@x.com>", Subject="Hi"),
        "body": {"data": encode("Body")},
    }
    ids = [f"m{i}" for i in range(7)]
    service = FakeGmailService(listing(*ids), {i: raw_message(i, payload=payload) for i in ids})

    emails = await EmailSource(service).load(SINCE, max_emails=5)

    # The cap bounds the get calls -- the expensive part -- not just the
    # returned list: only the first 5 ids are ever fetched, and the cap is
    # passed to the API as maxResults too.
    assert len(emails) == 5
    assert service.resource.fetched_ids == ids[:5]
    assert service.resource.max_results == [5]


async def test_drops_messages_with_no_text_body():
    # A calendar invite or attachment-only mail: nothing to summarise.
    attachment_only = {
        "mimeType": "multipart/mixed",
        "headers": headers(From="Jane <jane@x.com>", Subject="Slides"),
        "parts": [{"mimeType": "application/pdf", "body": {"attachmentId": "a1"}}],
    }
    readable = {
        "mimeType": "text/plain",
        "headers": headers(From="Jane <jane@x.com>", Subject="Hi"),
        "body": {"data": encode("Body")},
    }
    service = FakeGmailService(
        listing("m1", "m2"),
        {
            "m1": raw_message("m1", payload=attachment_only),
            "m2": raw_message("m2", payload=readable),
        },
    )

    emails = await EmailSource(service).load(SINCE)

    assert [e.id for e in emails] == ["m2"]


async def test_returns_empty_when_the_window_holds_no_mail():
    service = FakeGmailService({"messages": []}, {})

    assert await EmailSource(service).load(SINCE) == []


def email(message_id: str) -> Email:
    return Email(
        id=message_id,
        sender="News <hi@news.com>",
        subject="Weekly",
        received_at=RECEIVED,
        list_unsubscribe=True,
        body="## ANN indexing\n\nHNSW vs IVF",
    )


def golden_file(tmp_path: Path, emails: list[Email]) -> Path:
    # The exact format scripts/smoke_gmail.py writes: a JSON array of
    # model_dump(mode="json") dicts. This is the writer/reader contract.
    path = tmp_path / "golden_emails.json"
    path.write_text(json.dumps([e.model_dump(mode="json") for e in emails]))
    return path


async def test_golden_source_round_trips_the_captured_emails(tmp_path):
    saved = [email("m1"), email("m2")]
    source = GoldenEmailSource(golden_file(tmp_path, saved))

    # `since` is deliberately ignored: it lies AFTER the captured mail, and a
    # real filter would return [] -- the frozen set must load regardless.
    after_capture = datetime(2026, 7, 20, tzinfo=UTC)
    emails = await source.load(after_capture)

    assert emails == saved
    assert emails[0].received_at.tzinfo is not None


async def test_golden_source_caps_at_max_emails(tmp_path):
    saved = [email(f"m{i}") for i in range(5)]
    source = GoldenEmailSource(golden_file(tmp_path, saved))

    emails = await source.load(SINCE, max_emails=3)

    assert [e.id for e in emails] == ["m0", "m1", "m2"]
