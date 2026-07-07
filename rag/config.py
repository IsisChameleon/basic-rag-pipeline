from __future__ import annotations

import os
from pathlib import Path

# No pydantic-settings layering yet (see build_log.md) -- these settings just
# override the local-dev default via env var (RAG_DATA_DIR set in
# docker-compose.yaml; GOOGLE_API_KEY from the local .env).
DATA_DIR = Path(os.environ.get("RAG_DATA_DIR", "data"))
DB_PATH = DATA_DIR / "rag.db"
CHROMA_DIR = DATA_DIR / "chroma"

# LLM used by /answer to generate the final answer from retrieved sources.
# Same model the readme book pipeline uses.
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
LLM_MODEL = "gemini-2.5-flash"
