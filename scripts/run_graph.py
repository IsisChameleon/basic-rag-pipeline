"""Run the digest graph against the golden set (data/golden_emails.json).

Run from the repo root:
    uv run python scripts/run_graph.py
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

from agent.email_source import GoldenEmailSource
from agent.graph import build_digest_graph


def main() -> None:
    emails = asyncio.run(GoldenEmailSource().load(datetime.now(UTC)))
    result = build_digest_graph().invoke(
        {"pending": emails, "processed": [], "failed": []}
    )

    print(f"pending={len(result['pending'])} processed={len(result['processed'])} "
          f"failed={len(result['failed'])}")
    for p in result["processed"]:
        print(f"  {p.category:12} topics={p.topics} summary={(p.summary or '')[:50]!r}")
    for f in result["failed"]:
        print(f"  FAILED {f.email.subject[:50]!r}: {f.failure}")


if __name__ == "__main__":
    main()
