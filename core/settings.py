from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All runtime configuration in one place (env vars / .env). Values flow
    to the classes that need them via each entrypoint's container -- nothing
    outside a composition root reads settings."""

    # validation_alias keeps the pre-existing env var name RAG_DATA_DIR
    # (set in docker-compose.yaml).
    data_dir: Path = Field(default=Path("data"), validation_alias="RAG_DATA_DIR")
    google_api_key: str = ""
    llm_model: str = "gemini-2.5-flash"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    chunk_target_tokens: int = 350
    chunk_overlap_tokens: int = 50

    # extra="ignore": the local .env also holds keys read directly by other
    # libraries (LANGFUSE_*), not by Settings.
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @property
    def db_path(self) -> Path:
        return self.data_dir / "rag.db"

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"
