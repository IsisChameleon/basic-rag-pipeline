# Build log

Running log of tools used, decisions made, and why — including changes made
in response to user feedback during the build.

## 2026-07-05 — Project kickoff & scaffold design

**Context gathering.** Read `~/src/readme` (`docker-compose.yaml`, `CLAUDE.md`,
`ruff.toml`, `server/pyproject.toml`, `server/dev.Dockerfile`,
`server/api/main.py`, `server/api/deps.py`, `server/api/routers/start.py`,
`server/shared/config.py`, `server/shared/logging.py`,
`server/tests/api/conftest.py`, `AGENTS.md`) to establish the Python/FastAPI
conventions to follow: uv-managed Python 3.13, FastAPI with routers split
from `main.py`, loguru logging, ruff (line-length 100, `select = ["I"]`),
`pydantic-settings` for layered config, pytest + `TestClient`, and a
docker-compose pattern of bind-mounted source + `uv sync --frozen` +
`uvicorn --reload`.

**Scope correction.** Initial brainstorming pass over-assumed the RAG
implementation itself — asked the user to pick a vector store, an
embedding/LLM provider, and an ingestion strategy (single page vs. crawl).
The user corrected this: the actual request was scaffolding only — working
FastAPI project structure, docker-compose, and three **empty stub**
endpoints (`/ingest`, `/search`, `/answer`), not the RAG logic. Those
architecture questions are deferred, not answered, and are recorded as
out-of-scope in the design spec
(`docs/superpowers/specs/2026-07-05-rag-api-scaffold-design.md`).

**Router grouping decision.** Originally proposed one router per endpoint
(3 routers). User pushed back: at most 2 routers. Landed on grouping by
read/write concern — `ingest.py` (write path) and `query.py` (`/search` +
`/answer`, both read/query-time operations against the same index).

**Repo layout decision.** Unlike readme (`server/api`, `server/bot`,
top-level `client/`), this repo *is* the server — no sibling client/bot — so
`api/` sits at the repo root rather than under a `server/` subdirectory.

**Config machinery deferred.** Skipped readme's `pydantic-settings`
layered-TOML `shared/config.py` pattern for now: there is nothing to
configure yet (no API keys, no vector store, no DB) since endpoints are
stubs. Will introduce it when the first real setting (e.g. an API key)
shows up.
