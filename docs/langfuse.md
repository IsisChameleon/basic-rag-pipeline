# Langfuse tracing & evaluation

[Langfuse](https://langfuse.com) gives the RAG pipeline **observability**: every
`/search` and `/answer` call emits a nested trace with one span per stage, so you
can see what each retrieval/rerank/generation step did and how long it took.

It runs **self-hosted, on demand**, behind a Docker Compose profile — the default
`docker compose up` is unchanged and does not start any of it.

## What the profile starts

`docker compose --profile langfuse up` adds six containers to the usual `api`
service (adapted from the [official Langfuse stack](https://github.com/langfuse/langfuse/blob/main/docker-compose.yml)):

| Service | Role |
|---|---|
| `langfuse-web` | UI + ingestion API — the only one with a host port (**http://localhost:3000**) |
| `langfuse-worker` | processes ingested events asynchronously |
| `postgres` | transactional store |
| `clickhouse` | trace/observation analytics store |
| `redis` | queue/cache |
| `minio` | S3-compatible blob store for event payloads |

All the infra secrets are **fixed local-dev values** hard-coded in
`docker-compose.yaml`; the instance is only reachable on the compose network and
`localhost:3000`, so they are not real secrets.

## Demo runbook

1. **Enable tracing in `.env`.** Add these lines (also in `.env.example`). The keys
   match the pair the `langfuse-web` container auto-provisions on first boot:
   ```bash
   LANGFUSE_HOST=http://langfuse-web:3000
   LANGFUSE_PUBLIC_KEY=pk-lf-1234567890
   LANGFUSE_SECRET_KEY=sk-lf-1234567890
   ```
   Leaving the keys blank disables tracing (the SDK no-ops) — that is the default.

2. **Start everything:**
   ```bash
   docker compose --profile langfuse up
   ```
   First boot pulls ~several GB of images and initializes ClickHouse/MinIO, so give
   it a minute. On first boot Langfuse also creates the org/project/user and the dev
   key pair automatically (no UI setup needed).

3. **Open the UI:** http://localhost:3000 — log in with:
   - email `demo@example.com`
   - password `demopassword`

4. **Drive the pipeline** (the api host port is `API_PORT`, default 8000):
   ```bash
   curl -X POST localhost:8000/answer \
     -H 'content-type: application/json' \
     -d '{"query":"what is contextual retrieval?"}'
   ```

5. **See the trace** in the Langfuse UI → *Tracing*. For an `/answer` call:
   ```
   answer                       (trace root)
   ├── hybrid-search
   │   ├── bm25-retrieve        sparse (SQLite FTS5)
   │   ├── vector-retrieve      dense / semantic (Chroma)
   │   ├── rrf-fuse             reciprocal rank fusion
   │   └── rerank               cross-encoder re-rank
   └── gemini-generate          generation (model + token usage)
   ```
   `/search` produces the same tree without `gemini-generate`.

## How the instrumentation works

Each pipeline stage is a small `@observe`-decorated function in
`rag/search_service.py`; the Gemini call in `rag/answer_service.py` is
`@observe(as_type="generation")` and logs model + token usage. When the Langfuse
keys are unset the decorators are **no-ops**, so the code path is identical to
running without Langfuse. `api/main.py` flushes buffered traces on shutdown.

## Phase 2 — evaluations (designed, not built yet)

Langfuse also does evaluation. The design lives in
[`docs/superpowers/specs/2026-07-07-langfuse-observability-design.md`](superpowers/specs/2026-07-07-langfuse-observability-design.md)
(section 4). In short:

- **Offline retrieval metrics** (NDCG@k, recall@k, MRR from [`docs/evals.md`](evals.md)) —
  need a labeled `{query, relevant_chunks}` **dataset**. Stored as a Langfuse
  Dataset; a `scripts/eval_retrieval.py` runs each query through `hybrid_search`,
  computes the metrics, and pushes them as **scores** on a dataset run.
- **Online answer scoring** — Langfuse **LLM-as-judge** (faithfulness, relevance) on
  the `gemini-generate` generations, configured in the UI. No dataset needed.

The blocker for the offline metrics is a hand-labeled query set (~15–25 queries),
which is the recommended first Phase 2 task.
