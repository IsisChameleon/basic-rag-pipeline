# Langfuse Observability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Langfuse as an on-demand Docker Compose profile and instrument the RAG pipeline so every stage (sparse, dense, fusion, rerank, generation) emits a nested trace span.

**Architecture:** Self-hosted Langfuse v3 stack (6 containers) lives behind `profiles: ["langfuse"]` so the default `docker compose up` is unchanged. The FastAPI `api` service gets Langfuse SDK env vars sourced from `.env` with empty defaults, so tracing is a silent no-op unless configured. Pipeline stages are extracted into `@observe`-decorated helpers to form a clean trace tree.

**Tech Stack:** Docker Compose profiles, Langfuse v3 (self-host) + `langfuse` Python SDK v4, FastAPI, uv.

## Global Constraints

- Python `>=3.13`; package manager is **uv** (`uv sync --frozen` / `uv add`).
- Add deps to `pyproject.toml` `dependencies`; keep `torch` on the CPU wheel index (unchanged).
- Langfuse SDK package name: `langfuse`, floor `>=3`.
- Langfuse infra secrets are **fixed local-dev values**, inline in compose, commented as non-secret/local-only.
- `api` must NOT `depends_on` any Langfuse service.
- Instrumentation must be **behavior-preserving**: all existing tests pass unchanged with tracing disabled (empty keys).
- Commit after each task. Never commit to `main` (we are on `add-langfuse-observability`).
- Docker convention: run via `docker compose`, restart the stack after config changes (per user's CLAUDE.md).

---

### Task 1: Add the `langfuse` dependency

**Files:**
- Modify: `pyproject.toml` (dependencies list)
- Modify: `uv.lock` (regenerated)

**Interfaces:**
- Produces: `import langfuse`, `from langfuse import observe, get_client` available to all `rag/*` modules.

- [ ] **Step 1: Add the dependency**

In `pyproject.toml`, add to `dependencies` (after `"google-genai>=1.63.0",`):
```toml
    "langfuse>=3",
```

- [ ] **Step 2: Lock + install**

Run: `uv sync`
Expected: resolves and installs `langfuse` (+ its OpenTelemetry deps); `uv.lock` updated.

- [ ] **Step 3: Verify import**

Run: `uv run python -c "from langfuse import observe, get_client; print('ok')"`
Expected: prints `ok` (no keys set → no error on import).

- [ ] **Step 4: Commit**
```bash
git add pyproject.toml uv.lock
git commit -m "Add langfuse dependency"
```

---

### Task 2: Instrument `hybrid_search` with per-stage spans

**Files:**
- Modify: `rag/search_service.py`
- Test: `tests/rag/test_search_service.py` (create) — but primary guard is existing behavior.

**Interfaces:**
- Consumes: `store.search_bm25`, `embeddings.embed_query`, `vectorstore.query`, `embeddings.rerank` (unchanged).
- Produces: `hybrid_search(query, top_k)` unchanged public signature/return; new private helpers `_bm25_retrieve`, `_vector_retrieve`, `_rrf_fuse`, `_rerank`.

- [ ] **Step 1: Write a behavior-preservation test**

Create `tests/rag/test_search_service.py`. This test fakes the four external boundaries and asserts `hybrid_search` still fuses + reranks correctly (tracing disabled = no keys in test env, so `@observe` is a no-op):

```python
from rag import search_service


def test_hybrid_search_fuses_and_reranks(monkeypatch):
    bm25 = [{"id": "a", "text": "alpha", "url": "u", "title": "t", "heading_path": ""}]
    vec = [{"id": "b", "text": "beta", "url": "u", "title": "t", "heading_path": ""}]

    conn = object()
    monkeypatch.setattr(search_service.store, "get_connection", lambda: conn)
    monkeypatch.setattr(search_service.store, "search_bm25", lambda c, q, limit: bm25)
    # close() must be callable on the fake connection
    monkeypatch.setattr(search_service.store, "get_connection", lambda: type("C", (), {"close": lambda self: None})())
    monkeypatch.setattr(search_service.embeddings, "embed_query", lambda q: [0.0])
    monkeypatch.setattr(search_service.vectorstore, "query", lambda v, n_results: vec)
    # rerank: score b higher than a so order is deterministic
    monkeypatch.setattr(
        search_service.embeddings,
        "rerank",
        lambda q, texts: [2.0 if t == "beta" else 1.0 for t in texts],
    )

    results = search_service.hybrid_search("query", top_k=2)

    assert [r.text for r in results] == ["beta", "alpha"]
    assert results[0].score == 2.0
```

- [ ] **Step 2: Run it against the CURRENT code to confirm it passes first**

Run: `uv run pytest tests/rag/test_search_service.py -v`
Expected: PASS (this pins current behavior before the refactor).

- [ ] **Step 3: Refactor into `@observe` helpers**

Rewrite `rag/search_service.py` keeping the same logic, split into decorated stages:

```python
from __future__ import annotations

from dataclasses import dataclass

from langfuse import observe

from rag import embeddings, store, vectorstore

_RRF_K = 60
_CANDIDATE_POOL_SIZE = 20


@dataclass
class SearchResult:
    text: str
    url: str
    title: str
    heading_path: str
    score: float


@observe(name="bm25-retrieve")
def _bm25_retrieve(query: str) -> list[dict]:
    conn = store.get_connection()
    try:
        return store.search_bm25(conn, query, limit=_CANDIDATE_POOL_SIZE)
    finally:
        conn.close()


@observe(name="vector-retrieve")
def _vector_retrieve(query: str) -> list[dict]:
    query_vector = embeddings.embed_query(query)
    return vectorstore.query(query_vector, n_results=_CANDIDATE_POOL_SIZE)


@observe(name="rrf-fuse")
def _rrf_fuse(bm25_hits: list[dict], vector_hits: list[dict]) -> list[dict]:
    fused_scores: dict[str, float] = {}
    chunk_data: dict[str, dict] = {}
    for rank, hit in enumerate(bm25_hits):
        cid = str(hit["id"])
        fused_scores[cid] = fused_scores.get(cid, 0.0) + 1.0 / (_RRF_K + rank + 1)
        chunk_data[cid] = hit
    for rank, hit in enumerate(vector_hits):
        cid = str(hit["id"])
        fused_scores[cid] = fused_scores.get(cid, 0.0) + 1.0 / (_RRF_K + rank + 1)
        chunk_data.setdefault(cid, hit)
    ordered = sorted(fused_scores, key=lambda cid: fused_scores[cid], reverse=True)
    return [chunk_data[cid] for cid in ordered[:_CANDIDATE_POOL_SIZE]]


@observe(name="rerank")
def _rerank(query: str, candidates: list[dict]) -> list[tuple[dict, float]]:
    scores = embeddings.rerank(query, [c["text"] for c in candidates])
    return sorted(zip(candidates, scores, strict=True), key=lambda p: p[1], reverse=True)


@observe(name="hybrid-search")
def hybrid_search(query: str, top_k: int = 5) -> list[SearchResult]:
    """BM25 (SQLite FTS5) + vector (Chroma) candidates, merged by Reciprocal
    Rank Fusion, then reordered by the cross-encoder reranker."""
    candidates = _rrf_fuse(_bm25_retrieve(query), _vector_retrieve(query))
    if not candidates:
        return []
    ranked = _rerank(query, candidates)
    return [
        SearchResult(
            text=c["text"],
            url=c["url"],
            title=c["title"],
            heading_path=c.get("heading_path", ""),
            score=float(score),
        )
        for c, score in ranked[:top_k]
    ]
```

Note: the empty-result guard moves to "no candidates after fusion" (equivalent to the old `if not fused_scores` since fusion returns `[]` for empty input).

- [ ] **Step 4: Run new + existing tests**

Run: `uv run pytest tests/rag/test_search_service.py tests/rag/test_answer_service.py -v`
Expected: PASS (behavior preserved).

- [ ] **Step 5: Commit**
```bash
git add rag/search_service.py tests/rag/test_search_service.py
git commit -m "Instrument hybrid_search with per-stage Langfuse spans"
```

---

### Task 3: Instrument answer generation

**Files:**
- Modify: `rag/answer_service.py`
- Test: `tests/rag/test_answer_service.py` (existing tests must still pass)

**Interfaces:**
- Consumes: `hybrid_search` (now traced), `get_client`, `config.LLM_MODEL`.
- Produces: `generate_answer` unchanged signature/return; new `@observe(as_type="generation")` wrapper around the Gemini call.

- [ ] **Step 1: Decorate the entry point and extract the generation call**

In `rag/answer_service.py`, add `from langfuse import get_client, observe` and change `generate_answer` so the Gemini call lives in a decorated helper:

```python
@observe(as_type="generation", name="gemini-generate")
def _call_gemini(prompt: str) -> str:
    response = get_client_gemini().models.generate_content(
        model=config.LLM_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=_SYSTEM_INSTRUCTION,
            temperature=0.0,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    usage = getattr(response, "usage_metadata", None)
    if usage is not None:
        get_client().update_current_generation(
            model=config.LLM_MODEL,
            usage_details={
                "input": getattr(usage, "prompt_token_count", None),
                "output": getattr(usage, "candidates_token_count", None),
            },
        )
    return response.text or ""


@observe(name="answer")
def generate_answer(query: str, top_k: int = 5) -> AnswerResult:
    sources = hybrid_search(query, top_k=top_k)
    if not sources:
        return AnswerResult(answer="No relevant sources were found for this query.", sources=[])
    prompt = f"Question: {query}\n\nSources:\n{_format_sources(sources)}"
    return AnswerResult(answer=_call_gemini(prompt), sources=sources)
```

Rename the existing `get_client` (the Gemini client factory) to `get_client_gemini` to avoid colliding with Langfuse's `get_client`. Update the two internal references and any test monkeypatch target.

- [ ] **Step 2: Update the existing test's monkeypatch target**

In `tests/rag/test_answer_service.py`, the tests patch `answer_service.get_client`. After the rename they must patch `answer_service.get_client_gemini` (and the no-sources test patches `_fail` onto `get_client_gemini`). The `_call_gemini` helper reads the client via `get_client_gemini()`, so patching that name keeps the fake in place.

Change both `monkeypatch.setattr(answer_service, "get_client", ...)` lines to `"get_client_gemini"`.

- [ ] **Step 3: Run the answer-service tests**

Run: `uv run pytest tests/rag/test_answer_service.py -v`
Expected: PASS. (Tracing disabled → `@observe`/`update_current_generation` no-op; the fake Gemini client returns `SimpleNamespace(text=...)` with no `usage_metadata`, so the usage branch is skipped.)

- [ ] **Step 4: Commit**
```bash
git add rag/answer_service.py tests/rag/test_answer_service.py
git commit -m "Instrument answer generation as a Langfuse generation span"
```

---

### Task 4: Flush traces on API shutdown

**Files:**
- Modify: `api/main.py`

**Interfaces:**
- Consumes: `langfuse.get_client()`.
- Produces: FastAPI lifespan that flushes on shutdown.

- [ ] **Step 1: Add a lifespan handler**

Rewrite `api/main.py` to add a lifespan that flushes Langfuse on shutdown (guarded so it never raises if tracing is disabled):

```python
from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.routers.ingest import router as ingest_router
from api.routers.query import router as query_router
from core.logging_config import configure_logging

configure_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    try:
        from langfuse import get_client

        get_client().flush()
    except Exception:
        pass


app = FastAPI(title="Basic RAG Pipeline API", version="0.1.0", lifespan=lifespan)

app.include_router(ingest_router)
app.include_router(query_router)


@app.get("/health")
async def health():
    return {"status": "ok"}
```

- [ ] **Step 2: Verify the app still imports and health works**

Run: `uv run pytest tests/api/test_main.py -v`
Expected: PASS.

- [ ] **Step 3: Commit**
```bash
git add api/main.py
git commit -m "Flush Langfuse traces on API shutdown"
```

---

### Task 5: Add the Langfuse stack to Docker Compose under a profile

**Files:**
- Modify: `docker-compose.yaml`

**Interfaces:**
- Produces: `docker compose --profile langfuse up` starts `langfuse-web` (host :3000) + worker + postgres + clickhouse + redis + minio; `api` gets `LANGFUSE_HOST/PUBLIC_KEY/SECRET_KEY` env.

- [ ] **Step 1: Add Langfuse env vars to the `api` service**

In the `api` service `environment:` block, add (sourced from `.env`, empty default = tracing off):
```yaml
      LANGFUSE_HOST: ${LANGFUSE_HOST:-http://langfuse-web:3000}
      LANGFUSE_PUBLIC_KEY: ${LANGFUSE_PUBLIC_KEY:-}
      LANGFUSE_SECRET_KEY: ${LANGFUSE_SECRET_KEY:-}
```

- [ ] **Step 2: Add the six Langfuse services + volumes**

Adapt the [official self-host compose](https://raw.githubusercontent.com/langfuse/langfuse/main/docker-compose.yml). Every service gets `profiles: ["langfuse"]`. Use fixed local-dev secrets. Add `LANGFUSE_INIT_*` on `langfuse-web` to provision the org/project/user + the dev key pair used in `.env.example`. Add named volumes `langfuse_postgres_data`, `langfuse_clickhouse_data`, `langfuse_clickhouse_logs`, `langfuse_minio_data`. Only publish `langfuse-web` on host `3000`; keep the rest internal.

(Exact service block is transcribed from the upstream file at implementation time — do not hand-write ClickHouse/MinIO/worker env from memory; copy upstream and add `profiles` + `LANGFUSE_INIT_*`.)

- [ ] **Step 3: Validate compose config parses (both profiles)**

Run: `docker compose config >/dev/null && docker compose --profile langfuse config >/dev/null && echo ok`
Expected: `ok` (no YAML/interpolation errors).

- [ ] **Step 4: Confirm default `up` still starts only `api`**

Run: `docker compose config --services` and `docker compose --profile langfuse config --services`
Expected: first lists only `api`; second lists `api` + the 6 Langfuse services.

- [ ] **Step 5: Commit**
```bash
git add docker-compose.yaml
git commit -m "Add Langfuse self-host stack behind the langfuse compose profile"
```

---

### Task 6: Env template + demo runbook docs

**Files:**
- Modify: `.env.example`
- Create/Modify: docs (a `docs/langfuse.md` runbook)

**Interfaces:**
- Produces: documented fixed dev keys + `docker compose --profile langfuse up` runbook.

- [ ] **Step 1: Extend `.env.example`**

Append the three Langfuse vars with the fixed dev keys (same values as the compose `LANGFUSE_INIT_PROJECT_*`) and a comment explaining empty = tracing off:
```bash
# Optional: enable Langfuse tracing. Leave blank to disable (the SDK no-ops).
# To demo: start with `docker compose --profile langfuse up`, then fill these
# with the dev keys the langfuse-web container provisions on first boot.
LANGFUSE_HOST=http://langfuse-web:3000
LANGFUSE_PUBLIC_KEY=
LANGFUSE_SECRET_KEY=
```

- [ ] **Step 2: Write `docs/langfuse.md`**

Runbook: what the profile starts, `docker compose --profile langfuse up`, open `http://localhost:3000` (login with the init user email/password), the fixed dev keys to paste into `.env`, restart `api`, `POST /answer`, and the expected trace tree (answer → hybrid-search → {bm25, vector, rrf, rerank} + gemini-generate). Add a short "Phase 2: evaluations" pointer to the spec.

- [ ] **Step 3: Commit**
```bash
git add .env.example docs/langfuse.md
git commit -m "Document the Langfuse profile and demo runbook"
```

---

### Task 7: End-to-end verification

**Files:** none (verification only; findings recorded in `build_log.md`).

- [ ] **Step 1: Bring up the full stack**

Run: `docker compose --profile langfuse up -d` then wait for `langfuse-web` healthy.
Expected: all 7 containers running; `curl -s localhost:3000/api/public/health` returns ok.

- [ ] **Step 2: Configure keys + restart api**

Put the provisioned dev keys in `.env`, `docker compose --profile langfuse up -d api` (recreate to pick up env).

- [ ] **Step 3: Drive the pipeline**

Run: `curl -s -X POST localhost:8000/answer -H 'content-type: application/json' -d '{"query":"what is contextual retrieval?"}'`
Expected: a cited answer.

- [ ] **Step 4: Confirm the trace tree**

Query the Langfuse API (or UI) for the latest trace; assert observations named `answer`, `hybrid-search`, `bm25-retrieve`, `vector-retrieve`, `rrf-fuse`, `rerank`, `gemini-generate` exist and nest correctly.

- [ ] **Step 5: Confirm graceful degradation**

Run: `docker compose down` then `docker compose up -d` (no profile). `POST /answer` still returns an answer with tracing off (no errors in `docker compose logs api`).

- [ ] **Step 6: Log results**

Append a dated entry to `build_log.md` (symptom/decision/verification format) recording what was verified vs. what remains, then commit.

---

## Self-Review

**Spec coverage:**
- §1 compose profile → Task 5 ✓; headless init → Task 5 Step 2 ✓; graceful no-`depends_on` → Global Constraints + Task 5 ✓.
- §2 per-stage tracing → Tasks 2–3 ✓; flush on shutdown → Task 4 ✓; graceful degradation (empty keys) → Task 5 Step 1 + tests run with no keys ✓.
- §3 env/docs → Task 6 ✓.
- §4 Phase 2 evals → design-only, referenced in Task 6 Step 2; no build tasks (correct).
- Assumption A1 (empty keys no-op) is exercised by Tasks 2–4 tests running with no keys; if any test errors on import/decoration, the fallback `LANGFUSE_TRACING_ENABLED` gate is added in that task.

**Placeholder scan:** Task 5 Step 2 intentionally defers the verbatim upstream service block to implementation (copying upstream compose is safer than hand-writing ClickHouse/MinIO env from memory) — this is a deliberate "copy source X" instruction, not a vague TODO.

**Type consistency:** `hybrid_search`/`generate_answer` signatures unchanged; `get_client` (Gemini) renamed to `get_client_gemini` consistently across `answer_service.py` and its test (Task 3 Steps 1–2); Langfuse `get_client` used only in Tasks 3–4.
