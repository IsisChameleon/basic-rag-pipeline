git config --global push.autoSetupRemote true

uv run python -c "..."

  import asyncio
  from datetime import UTC, datetime
  from agent.email_source import GoldenEmailSource
  from agent.graph import build_digest_graph

  emails = asyncio.run(GoldenEmailSource().load(datetime.now(UTC)))
  result = build_digest_graph().invoke({"pending": emails, "processed": [], "failed": []})Why