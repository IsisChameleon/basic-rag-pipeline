"""The API's composition root: the only code in the API that knows concrete
wiring. It composes the shared RagContainer and adds what only the API needs;
everything else receives its dependencies. Tests replace the whole graph by
putting a fake-filled ApiContainer on app.state.container."""

from __future__ import annotations

from dataclasses import dataclass

from core.settings import Settings
from rag.container import RagContainer, build_rag_container
from rag.job_store import JobStore


@dataclass
class ApiContainer:
    settings: Settings
    rag: RagContainer
    # JobStore is API-only: it backs the async ingest job endpoints, and is
    # in-process, so it has no meaning outside this process.
    job_store: JobStore

    def warm_up(self) -> None:
        self.rag.warm_up()


def build_api_container(settings: Settings | None = None) -> ApiContainer:
    settings = settings or Settings()
    return ApiContainer(
        settings=settings,
        rag=build_rag_container(settings),
        job_store=JobStore(),
    )
