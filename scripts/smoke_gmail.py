"""Smoke test for the Gmail path: real OAuth, real mailbox, no mocks.

What the unit tests can't tell us, this can:
  1. Does the OAuth flow actually complete end-to-end? (browser consent ->
     token.json written -> an authorised Gmail service)
  2. What does markdownify do with a *real* newsletter -- usable Markdown with
     headings, or a wall of layout-table noise? That answer shapes the
     summarize node, so we look at a real body with our own eyes here.

It deliberately does NOT build the full agent container: no RAG models, no LLM
key needed. It exercises exactly "up to here" -- build_gmail_service +
EmailSource.load -- and nothing below.

Run from the repo root:
    uv run python scripts/smoke_gmail.py

First run opens a browser once for consent and writes token.json; later runs
reuse it silently.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from agent.container import build_gmail_service
from agent.email_source import EmailSource
from core.settings import Settings


async def main() -> None:
    settings = Settings()

    # build_gmail_service is the one function AgentCore replaces on day 2. Here
    # it runs the local InstalledAppFlow: load token.json, refresh, or consent.
    print("Authorising with Gmail (browser may open on first run)...")
    service = build_gmail_service(settings)

    # The same 24h window the digest will use. timezone-aware UTC so the epoch
    # conversion inside load() is unambiguous.
    since = datetime.now(UTC) - timedelta(hours=24)
    print(f"Loading mail since {since.isoformat()}\n")

    # max_emails caps the messages fetched (one get call each), so a busy
    # mailbox can't turn this smoke test into 200 API calls.
    emails = await EmailSource(service).load(since, max_emails=10)
    if not emails:
        print("No mail in the last 24h. Widen the window in this script to test.")
        return

    # Keep what we fetched: a fixed input set for developing the graph against,
    # instead of hitting the Gmail API on every run. model_dump(mode="json")
    # round-trips through Email.model_validate, so a loader gets identical
    # objects back. data/ is gitignored -- real mail never enters the repo.
    golden = Path("data/golden_emails.json")
    golden.write_text(json.dumps([e.model_dump(mode="json") for e in emails], indent=2))
    print(f"Saved {len(emails)} email(s) to {golden}\n")

    # One-line overview: this is the shape classify/summarize will consume.
    print(f"{len(emails)} email(s):\n")
    for email in emails:
        flag = "[newsletter]" if email.list_unsubscribe else "            "
        print(f"  {flag} {email.received_at:%m-%d %H:%M}  {email.sender[:40]:40}  {email.subject[:50]}")

    # The payoff: eyeball one real body as Markdown. Prefer a newsletter, since
    # that is the case whose headings the "read section 2" feature depends on.
    sample = next((e for e in emails if e.list_unsubscribe), emails[0])
    print("\n" + "=" * 70)
    print(f"BODY of: {sample.subject!r}  (list_unsubscribe={sample.list_unsubscribe})")
    print(f"length: {len(sample.body)} chars")
    print("=" * 70)
    print(sample.body[:1500])


if __name__ == "__main__":
    asyncio.run(main())
