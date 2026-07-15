# Architecture Refactor: Ports & Adapters — Design

> **DESCOPED 2026-07-15 — see Addendum at the end.** The injection/modeling
> core of this design is being implemented; the ports layer (Protocols,
> src-layout, domain/services/adapters taxonomy) is deferred until a second
> implementation of any seam actually exists.

> Status: **complete — all sections reviewed with user 2026-07-13/14**;
> self-review pass done 2026-07-14 (see Design review notes at the end).
> Domain layer is revision 2 (fewer chunk models, URI-based identity,
> symmetric store contracts, VectorStore name kept, lazy model loading
> inside adapters).

## Goal

Re-model the codebase for modularity and extensibility, designed around the
extension points the user anticipates:

- swap the relational store (SQLite → Postgres, FTS5 → tsvector)
- swap the vector store (Chroma → pgvector/Qdrant/…)
- swap the LLM (Gemini → Claude/OpenAI) and the embedder/reranker
- new document sources beyond sitemap-crawled HTML (PDFs, GitHub, uploads)
- new chunkers (e.g. a LangChain splitter)

Framework stance (agreed 2026-07-13): **frameworks are adopted as adapters
("mode 1")** — e.g. a LangChain splitter behind `Chunker`, LangChain chat
models behind `LLMClient`. The pipeline core stays custom. A LangGraph agent
may later replace/extend `AnswerService`; it would call `SearchService`
through its existing port.

Chosen shape: **ports & adapters (hexagonal), layered folders**. Dependencies
point inward: `domain` imports nothing of ours; `services` import only
`domain`; `adapters` implement domain contracts; `api` wires it all
together (composition root). No DI container library — plain constructor
injection + FastAPI `Depends`.

## Folder layout (approved)

```
src/rag_pipeline/
├─ domain/            # inner layer: no external deps
│  ├─ models.py       # Document, Chunk, ChunkRecord, SearchResult,
│  │                  #   AnswerResult, IngestSummary, IngestJob (pydantic)
│  └─ contracts.py    # Protocols: ChunkRepository, VectorStore,
│                     #   Embedder, Reranker, LLMClient,
│                     #   DocumentSource, Chunker, JobStore
├─ services/          # orchestration, depends on contracts only
│  ├─ ingest_service.py    # class IngestService
│  ├─ search_service.py    # class SearchService (bm25+vector+RRF+rerank)
│  └─ answer_service.py    # class AnswerService
├─ adapters/          # implement the domain ports; modules named by
│  │                  #   TECHNOLOGY, flat until a category grows (Python
│  │                  #   idiom: a module is a cohesive topic, not
│  │                  #   one-file-per-class). Name follows the ports &
│  │                  #   adapters pattern (cf. cosmicpython.com).
│  ├─ sqlite.py            # SqliteChunkRepository
│  ├─ chroma.py            # ChromaVectorStore
│  ├─ encoders.py          # SentenceTransformerEmbedder, CrossEncoderReranker
│  ├─ gemini.py            # GeminiClient
│  ├─ web_source.py        # WebDocumentSource (+ discover/fetch/extract helpers)
│  ├─ markdown_chunker.py  # MarkdownChunker
│  └─ jobs.py              # InMemoryJobStore
├─ api/
│  ├─ main.py         # app factory + lifespan
│  ├─ dependencies.py # composition root (builds container, Depends)
│  ├─ schemas.py      # request/response pydantic models
│  └─ routers/ingest.py, query.py
└─ core/              # settings.py (pydantic-settings), logging,
                      #   observability
```

## Domain layer (approved — revision 2)

Two files, no third-party imports beyond pydantic. Every service types its
constructor parameters against these Protocols; every adapter in
`adapters/` implements one.

### Key decisions from review

- **Chunk identity is minted by the service, not the database**:
  `id = f"{uri}#{chunk_index}"` — deterministic and readable. This collapses
  the stored shape and the retrieved shape into one model (`ChunkRecord`)
  and removes the id-authority coupling to SQLite's rowid.
- **Per-store relevance scores are dropped from retrieval results**: RRF
  fusion uses rank order only (true of the current code too), so keyword and
  vector search both return plain ordered `list[ChunkRecord]`. The only
  score a caller ever sees is the cross-encoder's, on `SearchResult`.
- **Identity is a URI, not a URL**: `https://…` (crawled), `file://…`
  (PDFs), `upload://…` (uploads). `delete_document(uri)` = "remove
  everything belonging to this document" — source-agnostic re-ingest.
- **`ChunkRepository` vs `VectorStore` relationship**: the repository is
  the *source of truth* for chunk records; the vector store is a *derived,
  rebuildable index* over them. On Postgres+pgvector, one adapter class
  implements BOTH protocols (distinct method names — `add` vs `add_vectors`
  — make that possible), and the composition root passes the same instance
  twice. Protocols are capabilities, not databases.

### `domain/models.py`

| Model | Fields | Replaces today |
|---|---|---|
| `Document` | `uri, title, markdown` | `extract.Page` dataclass |
| `Chunk` | `text, heading_path` | `chunk.Chunk` dataclass (chunker output, no identity yet) |
| `ChunkRecord` | `id, uri, title, heading_path, chunk_index, text, content_hash, fetched_at` | `store.ChunkMetadata`, `insert_chunk` kwargs, AND the raw dicts flowing through retrieval |
| `SearchResult` | `chunk: ChunkRecord, score: float` | `search_service.SearchResult` |
| `AnswerResult` | `answer, sources` | `answer_service.AnswerResult` |
| `IngestSummary` / `IngestJob` | as today | dataclasses in `ingest_service` / `jobs` |

```python
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class Document(BaseModel):
    """A document from any source, normalised to markdown. `uri` is its
    identity: https://… (crawled), file://… (PDF), upload://… (upload)."""
    uri: str
    title: str
    markdown: str


class Chunk(BaseModel):
    """Chunker output: a piece of a document, not yet stored, no identity."""
    text: str
    heading_path: str


class ChunkRecord(BaseModel):
    """The canonical chunk shape — what gets stored and what retrieval
    returns. `id` is deterministic: f"{uri}#{chunk_index}"."""
    id: str
    uri: str
    title: str
    heading_path: str
    chunk_index: int
    text: str
    content_hash: str
    fetched_at: str


class SearchResult(BaseModel):
    chunk: ChunkRecord
    score: float                   # cross-encoder rerank score


class AnswerResult(BaseModel):
    answer: str
    sources: list[SearchResult]    # citation order: sources[0] is [1]


class IngestSummary(BaseModel):
    pages_discovered: int
    pages_ingested: int
    chunks_stored: int


JobStatus = Literal["pending", "completed", "failed"]


class IngestJob(BaseModel):
    status: JobStatus
    result: IngestSummary | None = None
    error: str | None = None
```

(`IngestSummary` keeps its `pages_*` field names because they are exposed on
the public `/ingest/{job_id}` response; rename to `documents_*` when a
non-web source actually lands.)

### `domain/contracts.py`

One Protocol per extension seam. Structural typing: adapters do NOT inherit
from these — they just implement the matching methods, and mypy/pyright
verify conformance where an adapter is passed to a service constructor.

```python
from __future__ import annotations

from typing import Protocol

from rag_pipeline.domain.models import (
    Chunk, ChunkRecord, Document, IngestJob, IngestSummary,
)


class ChunkRepository(Protocol):
    """Source of truth for chunk records; also serves keyword search
    (FTS5 today, tsvector on Postgres)."""

    def add(self, chunks: list[ChunkRecord]) -> None: ...
    def delete_document(self, uri: str) -> None: ...
    def search_keyword(self, query: str, limit: int) -> list[ChunkRecord]:
        """Best-first by keyword relevance; order is the only signal RRF uses."""
        ...


class VectorStore(Protocol):
    """Derived, rebuildable similarity index over chunk records. On
    Postgres+pgvector one class may implement this AND ChunkRepository."""

    def add_vectors(self, chunks: list[ChunkRecord], vectors: list[list[float]]) -> None: ...
    def delete_document(self, uri: str) -> None: ...
    def search_similar(self, vector: list[float], limit: int) -> list[ChunkRecord]:
        """Best-first by similarity; order is the only signal RRF uses."""
        ...


class Embedder(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, query: str) -> list[float]: ...
    def count_tokens(self, text: str) -> int: ...


class Reranker(Protocol):
    def rerank(self, query: str, candidates: list[str]) -> list[float]: ...


class LLMClient(Protocol):
    def generate(self, prompt: str, *, system: str) -> str: ...


class DocumentSource(Protocol):
    async def load(self, source_url: str) -> list[Document | None]:
        """Discover, fetch, and extract every document under source_url.
        None entries mark documents that failed to fetch/extract (kept so
        the caller can report discovered vs ingested counts)."""
        ...


class Chunker(Protocol):
    def chunk(self, markdown: str) -> list[Chunk]: ...


class JobStore(Protocol):
    def create(self) -> str: ...
    def mark_completed(self, job_id: str, result: IngestSummary) -> None: ...
    def mark_failed(self, job_id: str, error: str) -> None: ...
    def get(self, job_id: str) -> IngestJob | None: ...
```

### Remaining design notes

- **`Reranker` separate from `Embedder`**: different model, independently
  swappable (e.g. Cohere rerank).
- **`DocumentSource` absorbs discover + fetch + extract** behind one port; a
  future PDF/GitHub source only has to produce `Document`s.
- **Chunker token-coupling solved at construction**:
  `MarkdownChunker(count_tokens=embedder.count_tokens)` — `IngestService`
  just sees a `Chunker`.
- **Schema impact**: `chunks.id` becomes `TEXT PRIMARY KEY` (was INTEGER
  rowid). Stored data is derived from crawling, so no migration script —
  re-ingest after the refactor.

## Services layer (approved)

Three classes in `services/`, one per use-case. They depend ONLY on
`domain` contracts (constructor injection); no adapter or third-party
import appears here. Pipeline tuning constants (`rrf_k`,
`candidate_pool_size`, `top_k` defaults) are constructor parameters with
today's values as defaults, so tests and future callers can vary them
without config plumbing.

Langfuse `@observe` decorators stay on the methods exactly as today
(they work unchanged on methods; no-op when tracing is off).

### `services/search_service.py`

```python
class SearchService:
    """Hybrid retrieval: keyword (repository) + vector (store) candidates,
    fused by Reciprocal Rank Fusion, reordered by the cross-encoder."""

    def __init__(
        self,
        repository: ChunkRepository,
        vector_store: VectorStore,
        embedder: Embedder,
        reranker: Reranker,
        *,
        candidate_pool_size: int = 20,
        rrf_k: int = 60,
    ) -> None: ...

    @observe(name="hybrid-search")
    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        # _keyword_retrieve + _vector_retrieve -> _rrf_fuse -> _rerank
        # each private stage keeps its own @observe span, as today
        ...
```

RRF fusion logic moves over unchanged, but operates on
`list[ChunkRecord]` keyed by `record.id` instead of raw dicts.

### `services/ingest_service.py`

```python
class IngestService:
    def __init__(
        self,
        source: DocumentSource,
        chunker: Chunker,
        embedder: Embedder,
        repository: ChunkRepository,
        vector_store: VectorStore,
    ) -> None: ...

    async def ingest(self, source_url: str) -> IngestSummary:
        # source.load() -> per document:
        #   chunker.chunk(doc.markdown)
        #   build ChunkRecords (id = f"{doc.uri}#{i}", content_hash, fetched_at)
        #   repository.delete_document(uri); vector_store.delete_document(uri)
        #   vector_store.add_vectors(records, embedder.embed_documents(texts))
        #   repository.add(records)
        ...
```

Same one-document-at-a-time memory profile as today.

### `services/answer_service.py`

```python
class AnswerService:
    def __init__(self, search_service: SearchService, llm: LLMClient) -> None: ...

    def answer(self, query: str, top_k: int = 5) -> AnswerResult:
        # search_service.search() -> format numbered sources ->
        # llm.generate(prompt, system=_SYSTEM_INSTRUCTION) -> AnswerResult
        ...
```

The citation system prompt lives here (it is domain logic, not an LLM
detail). `AnswerService` depends on `SearchService` *concretely* —
service-to-service dependency within the same layer is fine; a `Retriever`
port would be speculative (YAGNI). A future LangGraph agent replaces this
class and calls `SearchService` the same way.

## Composition root / DI wiring (approved)

No DI container library. One plain `Container` dataclass + one build
function = the only code in the app that knows concrete adapter classes.

### `core/settings.py` (pydantic-settings)

```python
class Settings(BaseSettings):
    data_dir: Path = Path("data")          # env RAG_DATA_DIR
    google_api_key: str = ""               # env GOOGLE_API_KEY
    llm_model: str = "gemini-2.5-flash"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    chunk_target_tokens: int = 350
    chunk_overlap_tokens: int = 50

    model_config = SettingsConfigDict(env_prefix="", env_file=".env")

    @property
    def db_path(self) -> Path: return self.data_dir / "rag.db"
    @property
    def chroma_dir(self) -> Path: return self.data_dir / "chroma"
```

(`RAG_DATA_DIR` keeps its current env name via a field alias.)

### `api/dependencies.py`

```python
@dataclass
class Container:
    settings: Settings
    search_service: SearchService
    answer_service: AnswerService
    ingest_service: IngestService
    job_store: JobStore


def build_container(settings: Settings | None = None) -> Container:
    settings = settings or Settings()
    embedder = SentenceTransformerEmbedder(settings.embedding_model)   # cheap; lazy model load
    reranker = CrossEncoderReranker(settings.reranker_model)           # cheap; lazy model load
    repository = SqliteChunkRepository(settings.db_path)
    vector_store = ChromaVectorStore(settings.chroma_dir)
    chunker = MarkdownChunker(
        count_tokens=embedder.count_tokens,
        target_tokens=settings.chunk_target_tokens,
        overlap_tokens=settings.chunk_overlap_tokens,
    )
    llm = GeminiClient(api_key=settings.google_api_key, model=settings.llm_model)
    search_service = SearchService(repository, vector_store, embedder, reranker)
    return Container(
        settings=settings,
        search_service=search_service,
        answer_service=AnswerService(search_service, llm),
        ingest_service=IngestService(
            WebDocumentSource(), chunker, embedder, repository, vector_store
        ),
        job_store=InMemoryJobStore(),
    )


def get_container(request: Request) -> Container:
    return request.app.state.container


def get_search_service(
    container: Annotated[Container, Depends(get_container)],
) -> SearchService:
    return container.search_service
# ... same one-liner per service / job_store
```

### `api/main.py` lifespan

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    container = build_container()             # cheap: nothing heavy loads yet
    container.warm_up()                       # API policy: models warm before serving
    app.state.container = container
    yield
    flush_langfuse()
```

Lazy loading survives, but as *instance* state inside the adapters
instead of module globals: `SentenceTransformerEmbedder`/
`CrossEncoderReranker` have cheap constructors and load their model on
first use. Each concrete encoder additionally exposes `warm_up()` — NOT
part of the `Embedder`/`Reranker` ports (warm-up timing is composition
policy, and only the composition root knows concrete classes, so it may
call it). `Container.warm_up()` forwards to them.

- API entry point: warms at boot → fail-fast, first request fast (today's
  behaviour, preserved).
- A future CLI/worker building the same container but never embedding
  pays ~nothing.
- Thread-safety of the lazy first load: in the API process the lifespan
  warm-up runs before any request thread exists, so no race; adapters
  carry a one-line comment noting this assumption.

`GeminiClient` likewise creates its google-genai client lazily on first
`generate()` and raises the "GOOGLE_API_KEY is not set" error there —
preserving today's behaviour that the app boots and ingest/search work
without a key (no warm_up: there is no model to load, just an HTTP
client).

### Evolution path: multiple entry points

Lazy-loading adapters (above) already mean building the container is
cheap, so a small CLI could reuse `build_container()` today without
paying the model-load cost. When a real second entry point appears
(search-only worker, ingest worker, scheduled crawler), apply the
canonical pattern — **one composition root per entry point** (per
executable process): each root wires only the object graph its process
needs (an ingest worker never constructs the reranker or LLM; an ops CLI
builds just the store it reads). Shared subgraphs become small `build_*`
helpers each root reuses. At that moment `build_container` graduates from
`api/dependencies.py` to a neutral `composition.py` module beside `api/`,
and `api/dependencies.py` keeps only the FastAPI `Depends` glue. Nothing
outside the roots changes — no port or service knows concrete classes.

### Threading / connections

`SqliteChunkRepository` holds the db path, not a connection; it opens a
short-lived connection per operation (as the current code does). That
keeps it safe across FastAPI's threadpool for sync handlers and the
background-task thread that runs ingest.

### How tests swap fakes

- **Service unit tests**: construct the service directly with fakes —
  `SearchService(FakeChunkRepository(), FakeVectorStore(), FakeEmbedder(),
  FakeReranker())`. No patching, no globals.
- **API tests**: build the app, then
  `app.state.container = Container(settings=…, search_service=…fakes…, …)`
  (or override `get_container` via `app.dependency_overrides`).
- **SQLite adapter tests**: real `SqliteChunkRepository(":memory:")`.

## API layer (approved)

Thin move, not a redesign. **The wire contract does not change**: same
routes, same request/response JSON (fields still named `url`,
`pages_discovered`, …) — existing clients and any saved curl commands keep
working.

- `api/schemas.py`: all request/response pydantic models consolidated
  (SearchRequest/Response, AnswerRequest/Response, Citation,
  IngestRequest, IngestAcceptedResponse, IngestJobResponse). Routers stop
  defining models inline.
- Routers keep sync `def` handlers (same threadpool reasoning as today)
  and receive services via `Depends` providers from `api/dependencies.py`.
- Mapping domain → wire happens in the router: e.g. domain
  `SearchResult.chunk.uri` → wire field `url`; `AnswerResult.sources` →
  `citations`.
- Ingest background task closes over `container.ingest_service` and
  `container.job_store` instead of importing module globals.

## Error handling & observability

Semantics unchanged, relocated:

- Fetch/extract failures: still skip-and-log per document inside
  `WebDocumentSource` (None entries in `load()`'s result).
- Ingest job failures: caught in the background task, recorded via
  `JobStore.mark_failed`, surfaced by `GET /ingest/{job_id}`.
- Missing `GOOGLE_API_KEY`: raised on first `GeminiClient.generate()`
  (see composition root section).
- `rag/observability.py` (ObservationType) moves to `core/observability.py`.
  `@observe` decorators stay on service methods and on
  `GeminiClient.generate` (which keeps the token-usage reporting).
  Langfuse flush stays in the lifespan.

## Testing strategy

Tests mirror the package; fakes live in `tests/fakes.py` and implement
the domain Protocols (structural — no inheritance needed).

| Today | Becomes |
|---|---|
| `tests/rag/test_chunk.py` | `tests/adapters/test_markdown_chunker.py` (logic unchanged) |
| `tests/rag/test_store.py` | `tests/adapters/test_sqlite.py` — real `SqliteChunkRepository(":memory:")` |
| `tests/rag/test_discover.py`, `test_extract.py` | `tests/adapters/test_web_source.py` |
| `tests/rag/test_search_service.py` | `tests/services/test_search_service.py` — constructor-injected fakes, no monkeypatching |
| `tests/rag/test_answer_service.py` | `tests/services/test_answer_service.py` — `FakeLLMClient`, fake-backed `SearchService` |
| `tests/api/*` | same paths — app with a fake-filled `Container` on `app.state` |

Rules (per repo owner's testing discipline): mock/fake at external
boundaries only (stores, encoders, LLM); the SQLite adapter is tested
real against `:memory:`; no `mock.patch` of module attributes anywhere —
that pattern dies with the module globals.

## Migration plan

Single feature branch (`refactor/ports-and-adapters`), one logical move,
tests green at the end (intermediate commits may be red while files move —
squash-merge or accept that on the branch).

1. Scaffold `src/rag_pipeline/` package; switch `pyproject.toml` to
   src-layout; `uv sync`.
2. `domain/` (models + contracts) — new code.
3. `adapters/` — largely `git mv` + wrap: existing function bodies become
   methods (chunk.py → markdown_chunker.py, store.py → sqlite.py,
   vectorstore.py → chroma.py, embeddings.py → encoders.py,
   discover/fetch/extract → web_source.py, answer_service's gemini bits →
   gemini.py, jobs.py → jobs.py).
4. `services/` — IngestService, SearchService, AnswerService.
5. `core/settings.py` (pydantic-settings, replaces rag/config.py),
   `core/observability.py`, `core/logging_config.py`.
6. `api/` — schemas.py, dependencies.py (Container), routers, main.py.
7. Move/rewrite tests per the table; add `tests/fakes.py`.
8. Update `dev.Dockerfile` / `docker-compose.yaml` uvicorn target →
   `rag_pipeline.api.main:app`; update README/build_log references.
9. Delete old `rag/`, `api/`, `core/` trees; grep for stragglers.
10. Verify: full pytest run; `docker compose up --build`; manual smoke of
    POST /ingest → poll job → /search → /answer.

**Data**: `chunks.id` becomes TEXT (deterministic `{uri}#{index}`), so no
schema migration script — wipe `data/` and re-ingest (all stored data is
re-derivable from crawling).

**Out of scope** (explicitly): behaviour changes to retrieval quality,
new endpoints, Postgres/pgvector implementation, LangGraph agent — this
refactor only creates the seams for them.

## Design review notes (senior self-review, 2026-07-14)

Honest assessment of the choices, recorded so future readers know what
was deliberate and what to watch.

**Pattern lineage.** This is the mainstream Python blend — hexagonal
(ports as Protocols, adapters, dependency rule) + repository pattern +
application services + a hand-rolled composition root — i.e. the
architecture of *Architecture Patterns with Python* (cosmicpython.com),
minus its heavier machinery (no unit-of-work, no message bus, no domain
events: nothing here needs transactional workflows). Known, accepted
deviations from purist forms:

- **Anemic domain**: `domain/models.py` is data-only; behaviour lives in
  services. Deliberate — a RAG pipeline is a dataflow app, not a rich
  business domain. "domain/" here means "models + ports", not DDD.
- **`AnswerService` depends on `SearchService` concretely** (same-layer);
  a `Retriever` port would be speculative.
- **`warm_up()` on concrete adapters, not ports** — composition policy,
  callable only where concretes are known (the root).

**Weakest port**: `JobStore`. Single implementation, and the real
evolution (durable task queue) would change semantics (push vs poll),
not just the backend. Kept because it costs ~6 lines and a Redis-backed
store IS a plausible intermediate step; cut it if it ever gets in the way.

**What "production-ready" still would need** (runtime gaps, not
architecture gaps — the seams make each of these a bounded change):
durable job store / task queue (in-memory jobs die on restart and break
under multiple uvicorn workers — true today as well); auth + rate
limiting on the API; SQLite's single-writer limit and Chroma-embedded
mode for concurrent/multi-instance deployments; blocking inference in
the request threadpool caps throughput (separate inference worker if QPS
grows); an eval harness before touching retrieval quality (docs/evals.md).

**Enforcement (recommended, cheap)**: add `import-linter` with a layers
contract (domain ← services ← adapters/api) to CI so the dependency rule
is machine-checked, not tribal knowledge.

**Implementation watch-items**:
- FTS5 external-content tables key on integer rowid; with `id TEXT
  PRIMARY KEY` the chunks table keeps an implicit rowid, so the FTS
  wiring must use that rowid (verify triggers in the sqlite adapter).
- Deterministic ids `{uri}#{chunk_index}` rely on delete-before-add per
  document; concurrent re-ingest of the same URI can interleave (same
  race exists today; single background worker makes it moot for now).
- Chroma metadata gains `content_hash`/`fetched_at` so retrieval can
  return complete `ChunkRecord`s from either store.

## Addendum — descope decision (2026-07-15, agreed with user)

Implement the **injection and modeling core** of this design; defer the
ports layer until a second implementation of any seam actually exists.

**Kept**: pydantic domain models incl. `ChunkRecord` with URI identity and
deterministic TEXT ids; role-named concrete classes with constructor
injection (`ChunkRepository`/sqlite, `VectorStore`/chroma, `Embedder`,
`Reranker`, `LLMClient`/gemini, `DocumentSource`/web, `MarkdownChunker`,
`JobStore`); the three service classes with tuning params as constructor
defaults; `Container` + `build_container` composition root; pydantic-settings;
fakes-based tests with no `mock.patch`; lazy model loading as instance state
with `warm_up()`; unchanged wire contract.

**Dropped (deferred)**: `domain/contracts.py` Protocols; the src-layout
migration and `domain/services/adapters` folder taxonomy (flat `rag/`,
`api/`, `core/` stay); import-linter layers contract.

**Rationale**: Python's structural typing makes ports retroactively free.
Because concrete classes carry the *role* names, adding a Protocol later is
one mechanical rename (`ChunkRepository` → `SqliteChunkRepository`) plus an
8-line Protocol under the vacated name — zero call-site churn. The part of
the refactor that gets more expensive with time (globals → constructor
injection, touching every call site and test) is done now; the part that is
free to defer (interface declarations) is deferred.

**Migration re-sequenced** to keep every commit green: new classes are built
alongside the old modules, the API flips to the new graph in one commit, and
the old modules are deleted last.
