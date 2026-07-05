# RAG API scaffold — design

Date: 2026-07-05

## Goal

Stand up a FastAPI project skeleton for a RAG pipeline: three routes
(`/ingest`, `/search`, `/answer`) as empty stubs, plus the project scaffolding
(docker-compose, uv/pyproject, ruff, tests) needed to run and test it. No RAG
logic (chunking, embeddings, vector store, LLM calls) is implemented yet —
that is deliberately out of scope until a follow-up design.

Conventions are carried over from `~/src/readme/server`: uv-managed Python
3.13, FastAPI with routers split out of `main.py`, loguru, ruff
(line-length 100, `select = ["I"]`), pytest + `TestClient`.

## Layout

Repo root is the server itself — there is no separate client or bot in this
repo, so no `server/` subdirectory is needed (unlike readme, which hosts
`client/`, `server/api`, and `server/bot` side by side).

```
api/
  main.py              # FastAPI app, CORS, /health
  routers/
    ingest.py          # POST /ingest
    query.py           # POST /search, POST /answer
tests/
  api/
    test_main.py
    routers/
      test_ingest.py
      test_query.py
pyproject.toml         # uv, python>=3.13, fastapi/uvicorn/loguru/pydantic
ruff.toml              # line-length 100, select = ["I"]
dev.Dockerfile         # uv/python3.13-bookworm base
docker-compose.yaml    # single `api` service, ${API_PORT:-8000}
build_log.md           # running decision log for this project
```

## Router grouping

`/search` and `/answer` are both query-time operations against the same
index; `/ingest` is the write path. Grouped as two routers, not three:
`ingest.py` (write) and `query.py` (`/search` + `/answer`, read).

## Endpoints (stubs only)

Request and response bodies are typed with pydantic `BaseModel` classes
(not bare dicts), matching readme's `api/routers/start.py` pattern
(`StartSessionRequest` / `StartSessionResponse`):

- `GET /health` → `{"status": "ok"}` (unchanged, untyped — matches readme's
  own `/health`)
- `POST /ingest` — `IngestRequest(url: str)` →
  `IngestResponse(status: str)`, stub returns `status="not_implemented"`
- `POST /search` — `SearchRequest(query: str)` →
  `SearchResponse(results: list)`, stub returns `results=[]`
- `POST /answer` — `AnswerRequest(query: str)` →
  `AnswerResponse(answer: str, citations: list)`, stub returns
  `answer="", citations=[]`

Request/response models are intentionally minimal (just the one field each
endpoint obviously needs) so the contract isn't over-specified before the
real implementation design happens.

## Error handling

None beyond FastAPI's automatic 422 on invalid request bodies. No RAG logic
exists yet to fail, so no further error handling is warranted at this stage.

## Testing

One smoke test per endpoint (status code + response shape) plus a health
check test, using `fastapi.testclient.TestClient`, mirroring readme's
`tests/api/test_main.py` pattern.

## Docker

`docker-compose.yaml` with a single `api` service: bind-mounted source,
`uv sync --frozen && uv run uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload`,
port configurable via `${API_PORT:-8000}`. No database or vector-store
service yet — deferred to the real RAG implementation design, which will
also decide the vector store, embedding/LLM provider, and ingestion scope
(single page vs. crawl).

## Explicitly out of scope (deferred)

These were raised during design discussion and deliberately deferred, not
decided:

- Vector store backend (embedded Chroma vs. Postgres/pgvector vs. other)
- LLM provider for `/answer` (Anthropic/OpenAI/Google)
- Ingestion scope (single page vs. same-domain crawl)
- `pydantic-settings`-style layered config (nothing to configure yet)

Embedding/reranking models are decided (recorded in `build_log.md`) but
**not implemented**: `BAAI/bge-small-en-v1.5` (bi-encoder, retrieval) +
`cross-encoder/ms-marco-MiniLM-L-6-v2` (reranker), both local
sentence-transformers-compatible models.

These will be revisited/implemented when the actual RAG logic is designed.
