from __future__ import annotations

import threading
import uuid

from rag.models import IngestJob, IngestSummary


class JobStore:
    """In-memory only: job state is lost on process restart and isn't shared
    across multiple worker processes. Acceptable at this project's current
    scope (single dev process); a real deployment would need a durable store
    (e.g. a task queue) instead."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, IngestJob] = {}

    def create(self) -> str:
        job_id = str(uuid.uuid4())
        with self._lock:
            self._jobs[job_id] = IngestJob(status="pending")
        return job_id

    def mark_completed(self, job_id: str, result: IngestSummary) -> None:
        with self._lock:
            self._jobs[job_id] = IngestJob(status="completed", result=result)

    def mark_failed(self, job_id: str, error: str) -> None:
        with self._lock:
            self._jobs[job_id] = IngestJob(status="failed", error=error)

    def get(self, job_id: str) -> IngestJob | None:
        with self._lock:
            return self._jobs.get(job_id)
