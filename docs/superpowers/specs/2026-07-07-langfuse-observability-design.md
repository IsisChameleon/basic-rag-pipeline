# Langfuse observability for the RAG pipeline — design

**Date:** 2026-07-07
**Branch:** `add-langfuse-observability`
**Status:** approved (design), Phase 1 implementation in progress

## Goal

Add [Langfuse](https://langfuse.com) to the project so we can (1) start it on demand
as a Docker Compose **profile** alongside the API, and (2) collect **per-stage traces**
of the RAG pipeline: sparse retrieval, dense/semantic retrieval, rank fusion, re-rank,
and answer generation. Evaluation metrics (NDCG@k, recall@k, etc.) are designed here
as **Phase 2** but not built now.

Context: this is for an **interview demo**. Priorities are *free*, *self-contained*
(no live-network dependency mid-demo), and *easy to run*. That drove the "self-host
behind a profile" choice over Langfuse Cloud.

## Current pipeline (what we're instrumenting)

`rag/search_service.py::hybrid_search` runs, in order:

1. **BM25 sparse** retrieval — SQLite FTS5 (`store.search_bm25`), 20 candidates.
2. **Dense/semantic** retrieval — embed query + Chroma vector search (`embeddings.embed_query` + `vectorstore.query`), 20 candidates.
3. **Reciprocal Rank Fusion (RRF)** — merge the two candidate lists (`_RRF_K = 60`).
4. **Cross-encoder re-rank** — `embeddings.rerank`, reorder the fused pool.
5. Return top-k `SearchResult`s.

`rag/answer_service.py::generate_answer` calls `hybrid_search`, formats the chunks as
numbered sources, and calls **Gemini** (`google-genai`, `gemini-2.5-flash`) to produce
a cited answer.

Both `/search` and `/answer` handlers are **sync `def`** (run in FastAPI's threadpool)
so the blocking SQLite/Chroma/sentence-transformers work stays off the event loop.

## Scope

- **Phase 1 (build now):** compose profile + per-stage tracing (sections 1–3 below).
- **Phase 2 (design only):** evaluation harness (section 4). Documentation, not code.

---

## 1. Langfuse as a Compose profile

Add the Langfuse v3 self-host stack to `docker-compose.yaml`. Every Langfuse service is
tagged `profiles: ["langfuse"]` so it only starts when explicitly requested. Verified
service list (from the [official compose](https://raw.githubusercontent.com/langfuse/langfuse/main/docker-compose.yml)):

| Service | Image | Host port | Purpose |
|---|---|---|---|
| `langfuse-web` | `langfuse/langfuse:3` | 3000 | UI + ingestion API |
| `langfuse-worker` | `langfuse/langfuse-worker:3` | (internal) | async event processing |
| `postgres` | `postgres:17` | (internal) | transactional store |
| `clickhouse` | `clickhouse/clickhouse-server` | (internal) | trace/observation OLAP store |
| `redis` | `redis:7` | (internal) | queue/cache |
| `minio` | `minio` | (internal) | S3-compatible blob store for events |

**Naming:** Langfuse's own services (`postgres`, `redis`, etc.) must not collide with
anything in our compose. We currently have only `api`, so no collision — but to be safe
and self-documenting, the Langfuse infra services keep their upstream names and all sit
behind the profile.

**Secrets:** fixed **local-dev** values (Postgres/ClickHouse/MinIO creds, `NEXTAUTH_SECRET`,
`SALT`, `ENCRYPTION_KEY`). These are not real secrets — the instance is local-only and
never exposed. They live inline in the compose file with a comment saying so.

**Zero-click provisioning (key demo affordance):** `langfuse-web` gets `LANGFUSE_INIT_*`
env vars so the org, project, a login user, **and a fixed dev API key pair** are created
on first boot. No logging into the UI to click "create project" before the demo works.
Verified variable names:

- `LANGFUSE_INIT_ORG_ID`, `LANGFUSE_INIT_ORG_NAME`
- `LANGFUSE_INIT_PROJECT_ID`, `LANGFUSE_INIT_PROJECT_NAME`
- `LANGFUSE_INIT_PROJECT_PUBLIC_KEY`, `LANGFUSE_INIT_PROJECT_SECRET_KEY`
- `LANGFUSE_INIT_USER_EMAIL`, `LANGFUSE_INIT_USER_NAME`, `LANGFUSE_INIT_USER_PASSWORD`

Applied only on first startup when the resources don't already exist (idempotent).

### Behaviour

- `docker compose up` → **only `api`** starts (unchanged from today; light).
- `docker compose --profile langfuse up` → `api` + the 6 Langfuse containers.
- All services share the compose **default network**, so `api` reaches Langfuse at
  `http://langfuse-web:3000` (service name, container-to-container — not `localhost`).
- **`api` does NOT `depends_on` Langfuse.** If it did, starting `api` would drag the
  whole stack up. Instead the SDK degrades gracefully (see §2).

---

## 2. Per-stage tracing instrumentation

### Dependency

Add `langfuse>=3` (the current major is v4; `>=3` is a floor — pin the resolved version
in `uv.lock`) to `pyproject.toml` `dependencies`. pip/uv package name is `langfuse`.

### Code changes

**`rag/search_service.py`** — extract each stage of `hybrid_search` into a small
`@observe`-decorated helper so the trace tree is one span per stage:

```python
from langfuse import observe

@observe(name="bm25-retrieve")
def _bm25_retrieve(query: str) -> list[dict]: ...

@observe(name="vector-retrieve")
def _vector_retrieve(query: str) -> list[dict]: ...

@observe(name="rrf-fuse")
def _rrf_fuse(bm25_hits, vector_hits) -> list[dict]: ...   # returns candidate pool

@observe(name="rerank")
def _rerank(query: str, candidates: list[dict]) -> list[tuple[dict, float]]: ...

@observe(name="hybrid-search")
def hybrid_search(query: str, top_k: int = 5) -> list[SearchResult]:
    # orchestrates the four helpers above
```

**`rag/answer_service.py`** — decorate the entry point and the Gemini call:

```python
@observe(name="answer")
def generate_answer(query: str, top_k: int = 5) -> AnswerResult:
    ...
    # wrap the Gemini call so it shows as a generation observation, logging
    # model + token usage via update_current_generation(...)
```

The Gemini call is wrapped in a helper decorated `@observe(as_type="generation",
name="gemini-generate")`; inside it we call `langfuse.update_current_generation(...)`
(or `.update()` on the observation) to record `model`, `input`, `output`, and
`usage_details` (token counts from the `google-genai` response's `usage_metadata`).

Resulting trace tree for `/answer`:

```
answer                       (trace root)
├── hybrid-search
│   ├── bm25-retrieve
│   ├── vector-retrieve
│   ├── rrf-fuse
│   └── rerank
└── gemini-generate          (generation: model, tokens)
```

`/search` yields the same tree without `gemini-generate`.

`@observe` auto-captures function args as span input and the return value as output.
For the retrieval spans we keep the raw hit dicts (chunk id + score + text); if the
serialized payload is noisy we trim to id+score+title via `update_current_span`.

### Graceful degradation when the profile is off

The SDK reads `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST` from the
environment. In `docker-compose.yaml` these are sourced from `.env` with **empty
defaults** (same pattern as `GOOGLE_API_KEY`/`HF_TOKEN`):

```yaml
LANGFUSE_HOST: ${LANGFUSE_HOST:-http://langfuse-web:3000}
LANGFUSE_PUBLIC_KEY: ${LANGFUSE_PUBLIC_KEY:-}
LANGFUSE_SECRET_KEY: ${LANGFUSE_SECRET_KEY:-}
```

Empty keys → the SDK disables tracing and the `@observe` decorators become no-ops
(the decorated functions still run normally). So with the profile off and `.env`
unset, `api` behaves exactly as today. To demo, fill the three values in `.env` with
the fixed dev keys and start with `--profile langfuse`.

> **Assumption to verify in implementation:** that empty keys make the SDK *no-op*
> rather than *raise* on `@observe`/`get_client()`. If it raises or logs noisily, add
> the explicit `LANGFUSE_TRACING_ENABLED` env gate (Langfuse supports it) defaulting to
> `false`, and flip it to `true` only under the profile.

### Flush on shutdown

Uvicorn runs with `--reload` in dev; buffered events can be lost on reload/exit. Add a
FastAPI shutdown hook (lifespan handler in `api/main.py`) that calls `langfuse.flush()`
(or `shutdown()`), guarded so it's a no-op when tracing is disabled.

### Threadpool note

No async/contextvars hazard: the whole `/answer` → `hybrid_search` → stage-helpers chain
runs **synchronously inside a single threadpool worker**, so the `@observe` span context
nests correctly within that thread. (FastAPI copies the request contextvars into the
threadpool when dispatching a sync handler.)

---

## 3. Config, env, docs

- **`.env.example`:** add `LANGFUSE_HOST`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`
  with the fixed dev values documented and a one-line "how to demo" note (the same keys
  the compose `LANGFUSE_INIT_PROJECT_*` vars provision).
- **Docs:** a short section (README or `docs/`) with the demo runbook:
  `docker compose --profile langfuse up` → open `http://localhost:3000` (login with the
  init user) → `POST /answer` → watch the trace tree appear.

---

## 4. Evaluations — Phase 2 (design only, NOT built now)

Langfuse does evaluation, but it splits into two things that need different setup:

**a) Offline retrieval metrics (the ones in `docs/evals.md`: NDCG@k, recall@k, MRR).**
These need **ground-truth relevance labels** — a set of queries each with known relevant
chunk ids/grades — which we don't have yet. Design:

- Store labeled items in a **Langfuse Dataset** (`{input: query, expected: relevant_chunk_ids/grades}`).
- A script `scripts/eval_retrieval.py` iterates the dataset, runs each query through
  `hybrid_search`, computes the metrics from `docs/evals.md`, and pushes them as
  **scores** attached to a Langfuse **dataset run** (so runs are comparable over time,
  e.g. before/after a reranker change).
- This is the natural fit for the pipeline: the reranker's whole job is ordering, and
  NDCG is grade- and position-aware (see `docs/evals.md`).

**b) Online answer-quality scoring (no dataset needed).**
Langfuse **LLM-as-judge** on `/answer` generations — faithfulness (answer grounded in
sources), relevance — configured in the Langfuse UI against the `gemini-generate`
observations produced in Phase 1. Good for a demo because it needs no labels.

**Blocked on:** a small labeled query→relevant-chunk dataset (for (a)). Recommend
hand-labeling ~15–25 queries against the current corpus as the first Phase 2 task.

---

## Assumptions & decisions log

Running log so the reviewer can see *why* each choice was made and which facts are
verified vs. assumed. Newest at the bottom.

### Decisions

- **D1 — Self-host over Langfuse Cloud.** User: interview demo, must be free, "whatever
  is easier." Chose self-host behind a profile because it's fully self-contained (no
  network/account dependency mid-demo), directly delivers the compose-profile skill the
  user asked for, and is free. Cloud is the documented fallback if the WSL2 box can't
  handle the RAM — same instrumentation code, only `LANGFUSE_HOST`+keys change.
- **D2 — All Langfuse services behind `profiles: ["langfuse"]`.** Keeps the default
  `docker compose up` identical to today. Confirmed by user's framing ("start langfuse
  when I start the project with profile langfuse").
- **D3 — `api` does NOT `depends_on` Langfuse.** Avoids forcing the stack up whenever
  the API starts; rely on SDK graceful degradation instead.
- **D4 — Extract stage functions + `@observe`** (user choice) over manual spans. Cleaner
  trace tree; a small, defensible refactor of `hybrid_search` that also improves it.
- **D5 — Phase 1 = tracing only; evals = Phase 2** (user choice). Evals need a labeled
  dataset the user doesn't have yet.
- **D6 — Headless provisioning via `LANGFUSE_INIT_*`** so the demo needs zero UI setup.
- **D7 — Fixed dev keys sourced from `.env` with empty defaults.** Mirrors the existing
  `GOOGLE_API_KEY`/`HF_TOKEN` pattern; empty default = tracing off = quiet when not
  demoing.

### Verified facts

- **V1** — Langfuse self-host is 6 containers (web, worker, postgres, clickhouse, redis,
  minio). Source: official docker-compose.yml.
- **V2** — Langfuse UI default host port is **3000**.
- **V3** — `LANGFUSE_INIT_*` headless-init variable names (listed in §1), applied only on
  first startup if resources don't exist. Source: Langfuse self-hosting docs.
- **V4** — Python SDK package is `langfuse`; current major is **v4** (v3 is legacy).
  `@observe(as_type="generation")`, `update_current_span`/`update_current_generation`,
  `.update(usage_details=..., metadata=...)`, and `flush()`/`shutdown()` all exist.
  Source: Langfuse Python SDK v4 docs.

### Open assumptions (verify during implementation)

- **A1** — Empty `LANGFUSE_PUBLIC_KEY`/`SECRET_KEY` make the SDK **no-op** rather than
  raise. Fallback if false: explicit `LANGFUSE_TRACING_ENABLED=false` gate. (See §2.)
- **A2** — `google-genai` response exposes token counts (`usage_metadata`) we can pass to
  `usage_details`. If the shape differs, log what's available and move on.
- **A3** — The WSL2 box has enough RAM/disk for the 6-container stack (ClickHouse +
  MinIO are the heavy ones). If not, fall back to Langfuse Cloud (D1).
- **A4** — No port collisions on the host: Langfuse only needs to publish 3000; other
  services stay internal to avoid clashing with anything the user runs locally.
