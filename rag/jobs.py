from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass
from typing import Literal

from rag.ingest_service import IngestSummary

JobStatus = Literal["pending", "completed", "failed"]

# In-memory only: job state is lost on process restart and isn't shared
# across multiple worker processes. Acceptable at this project's current
# scope (single dev process); a real deployment would need a durable store
# (e.g. a task queue) instead.
_lock = threading.Lock()
_jobs: dict[str, "IngestJob"] = {}


@dataclass
class IngestJob:
    status: JobStatus
    result: IngestSummary | None = None
    error: str | None = None


def create_job() -> str:
    job_id = str(uuid.uuid4())
    with _lock:
        _jobs[job_id] = IngestJob(status="pending")
    return job_id


def mark_completed(job_id: str, result: IngestSummary) -> None:
    with _lock:
        _jobs[job_id] = IngestJob(status="completed", result=result)


def mark_failed(job_id: str, error: str) -> None:
    with _lock:
        _jobs[job_id] = IngestJob(status="failed", error=error)


def get_job(job_id: str) -> IngestJob | None:
    with _lock:
        return _jobs.get(job_id)
