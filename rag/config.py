from __future__ import annotations

import os
from pathlib import Path

# No pydantic-settings layering yet (see build_log.md) -- these are the only
# two settings that exist so far, both just override the local-dev default
# via env var (RAG_DATA_DIR, set in docker-compose.yaml).
DATA_DIR = Path(os.environ.get("RAG_DATA_DIR", "data"))
DB_PATH = DATA_DIR / "rag.db"
CHROMA_DIR = DATA_DIR / "chroma"
