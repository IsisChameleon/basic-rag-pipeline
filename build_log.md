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

**Embedding/reranking model decision (implementation still deferred).**
User specified the models to use once retrieval is actually implemented:
`BAAI/bge-small-en-v1.5` as the bi-encoder for retrieval, and
`cross-encoder/ms-marco-MiniLM-L-6-v2` for reranking — both local,
sentence-transformers-compatible models (no external embedding API). This
only records the choice; no vector store, embedding, or reranking code is
added yet, per the scaffold-only scope of the current work.

## 2026-07-05 — Ingestion tool spike (research, no code yet)

**Manual extraction spike.** Fetched Anthropic's engineering blog
(`https://www.anthropic.com/engineering`) with plain `curl` to test the
ingest approach the user proposed (httpx + BeautifulSoup + sitemap.xml, no
crawling). Findings, saved as raw HTML/XML under `temp/` (gitignored) for
reference:
- The site is fully server-rendered (Next.js SSR) — no headless browser
  needed, `curl`/`httpx` alone returns full article text.
- `https://www.anthropic.com/sitemap.xml` lists all 482 site URLs including
  every `/engineering/...` post (26 of them) — sufficient to discover every
  post without following links.
- `<article>` scoping gives ~99.9% clean text; the only cruft found was two
  `"Copy"` code-block buttons.
- **Bug found**: naive whole-article `BeautifulSoup(...).get_text()` glues
  table cells together with no row/column boundaries (tested on
  `demystifying-evals-for-ai-agents`, which has 4 tables) — a real
  structure-loss problem, not just cosmetic cruft. Needs table-aware
  extraction (e.g. serialize each `<table>` to Markdown row-by-row).
- Image `alt` text on this site is always empty; `figcaption` text is
  substantive and worth keeping.

**Extraction library decision: Trafilatura (not Crawl4AI/Firecrawl/Scrapy).**
Researched what's commonly used for RAG ingestion (web search + read
https://aioutlooks.com/web-scraping-for-rag-pipelines/). Landscape:
- **Crawl4AI** — Playwright-based, LLM/RAG-focused, outputs clean Markdown,
  Apache-2.0. Strong option but carries a full browser dependency we don't
  need (site is static/SSR).
- **Firecrawl** — managed/self-hosted, handles JS + anti-bot, AGPL-3.0
  (copyleft licensing consideration). Heaviest option, solves problems this
  site doesn't have.
- **Scrapy** — general-purpose crawling framework, not RAG-specific; more
  infrastructure than a single-site sitemap-driven ingest needs.
- **Trafilatura** — MIT-licensed, no browser dependency, single
  `pip install trafilatura`, purpose-built for exactly "HTML → clean
  text/Markdown, strip boilerplate." Directly solves the table-glue bug
  found above (serializes tables sanely instead of concatenating cells).

Chosen: **httpx (fetch) + stdlib `xml.etree.ElementTree` (sitemap parsing,
no BeautifulSoup needed — sitemap XML is simple enough for stdlib) +
Trafilatura (content/table extraction)**, replacing the originally-proposed
manual `BeautifulSoup(html).find("article")` extraction. BeautifulSoup is
not itself an XML/HTML parser — it's a navigation API over an underlying
parser (`html.parser`/`lxml`/`html5lib`); for a well-formed, flat sitemap,
stdlib `ElementTree` is simpler and needs no extra dependency.

Also adopted from the aioutlooks article: convert HTML → Markdown *before*
chunking, and chunk on structural boundaries (headings, and keep each
table/list atomic) rather than blind token-count windows — this is the
mechanical fix for the table-glue bug, not just a downstream parameter to
tune. The rest of that article (vector DB picks, cost tables) is unsupported
marketing filler and was not adopted.

Sources:
- https://aioutlooks.com/web-scraping-for-rag-pipelines/
- https://github.com/unclecode/crawl4ai
- https://www.firecrawl.dev/blog/best-open-source-web-scraping-libraries
- https://www.capsolver.com/blog/AI/crawl4ai-vs-firecrawl
- https://www.sitemaps.org (Sitemaps protocol — explains why sitemap.xml is XML)

Still deferred: vector store backend, LLM provider for `/answer`, whether
ingestion stays single-page/sitemap-driven or adds crawling for sites
without a sitemap. No ingest code has been written yet — this is research
only.

## 2026-07-05 — Storage/retrieval wiring decision

**Full pipeline, decisions marked:**

1. Discover — sitemap.xml → list of page URLs
2. Fetch — httpx
3. Extract — Trafilatura → clean Markdown per page
4. Chunk — split Markdown on structural boundaries (headings, atomic tables/lists)
5. Store chunks — **SQLite** (`chunks` table: source of truth for text + metadata, used for citations)
6. Embed — `BAAI/bge-small-en-v1.5` (bi-encoder)
7. Store embeddings — **Chroma** (embedded, persisted to a volume)
8. Build BM25 index — **SQLite FTS5** virtual table (external-content, synced via triggers)
9. `/search` — BM25 (FTS5) query + vector (Chroma) query → merge candidates → rerank with `cross-encoder/ms-marco-MiniLM-L-6-v2` → return ranked chunks
10. `/answer` — reranked top-k → prompt an LLM (provider still deferred) → answer + citations sourced from stored chunk metadata (`url`, `title`)

**Storage decision: SQLite (chunks + FTS5 for BM25) + Chroma (vectors), not Postgres/pgvector, not Chroma-only.**
Three options were weighed:
- **Chosen — SQLite (chunk store + FTS5 for BM25) + Chroma (vector index).** Both embedded, no
  extra container: SQLite's FTS5 module has a built-in `bm25()` ranking function, so no separate
  BM25 library (`rank_bm25`, `bm25s`) is needed, and the same file doubles as the source-of-truth
  chunk store for citations. Chroma is added only for the one thing SQLite can't do: vector ANN
  search over the bi-encoder embeddings.
- **Rejected — Chroma only (vector-only, no BM25/hybrid search).** Simplest possible, but drops
  lexical/keyword matching entirely, which tends to matter for exact terms, code identifiers, and
  acronyms in technical docs.
- **Rejected — Postgres + pgvector + tsvector/ts_rank.** Most "production-like" and would unify
  storage in one DB, but requires an additional db container — contradicts the "single `api`
  container for now" constraint from the original scaffold discussion. Also, Postgres `ts_rank` is
  not literally BM25, just a similar ranking function.

**FTS5 mechanics** — full explanation (what FTS5 stands for, how/when the index is built,
external-content mode, and the required sync triggers) written to
`docs/superpowers/specs/2026-07-05-rag-storage-retrieval-design.md` rather than duplicated here.

Still deferred: LLM provider for `/answer`; whether ingestion needs a crawl fallback for sites
without a sitemap. No storage/retrieval code has been written yet — this is design only.

## 2026-07-05 — Chunking method decision

**Chosen: hand-rolled chunker, no LangChain/LlamaIndex dependency.**

Considered using `langchain-text-splitters` (the slim splitter-only package, not the full
LangChain framework) or LlamaIndex's `MarkdownNodeParser` for the chunking step (stage 4 of the
pipeline). Decided against both:

- Neither library understands table/list semantics — they split blind to content, by
  character/token count. Protecting a table from being split mid-way (the bug found during the
  ingestion spike, see the `demystifying-evals-for-ai-agents` entry above) still requires
  hand-written logic regardless of which splitter does the "easy part" (header splitting +
  recursive paragraph/sentence sub-splitting for oversized sections). Since that custom logic is
  unavoidable, the library only saves the mechanical recursive-splitting part — a modest saving.
- User has prior hands-on experience with both LangChain and LlamaIndex and flagged the same
  friction independently: adopting either means learning and reading through their `Document`
  object conventions/abstractions to understand what the code is actually doing, for a fairly
  small piece of logic. Confirmed judgment call, not just a cost/benefit guess — keep it simple.

Method: split Markdown on headers (`#`/`##`/`###`) first, tag each resulting section with its
heading path as metadata (e.g. `"Demystifying Evals > Grading strategies > Comparison table"`,
useful for citations); protect each table/list as an atomic span that is never split; sub-split
any oversized section on paragraph boundaries within a token budget (~500–900 tokens, small
overlap), all hand-written — no chunking library dependency.

**Embedding: no framework needed either — this follows directly from the model choice already
made.** `BAAI/bge-small-en-v1.5` and `cross-encoder/ms-marco-MiniLM-L-6-v2` are both loaded via
the `sentence-transformers` library directly (`SentenceTransformer` / `CrossEncoder` classes).
A LangChain/LlamaIndex embeddings wrapper would just add indirection around the same
`sentence-transformers` call with no benefit, since there's no embedding-provider swapping planned.

No chunking or embedding code has been written yet — this is a design decision only.

## 2026-07-05 — Full implementation

Implemented everything decided above, end to end. Confirmed the `/ingest` scope question first
(whole section via sitemap, prefix-filtered — see the question asked and answered inline in
conversation) before writing the ingest router, since that was the one piece genuinely not yet
decided.

**Files added:**
- `api/main.py`, `api/routers/{ingest,query}.py` — HTTP layer
- `rag/{config,discover,fetch,extract,chunk,store,embeddings,vectorstore,ingest_service,search_service}.py`
  — pipeline logic, kept separate from the HTTP layer
- `tests/api/**`, `tests/rag/**` — see testing notes below
- `pyproject.toml`, `dev.Dockerfile`, `docker-compose.yaml`, `.gitignore`

**Implementation-time findings and decisions (beyond what was already written down above):**

- **Bug caught before it shipped: `trafilatura.bare_extraction()`'s `.text` field is `None` for
  `output_format="markdown"`** (verified empirically against trafilatura 2.1.0 — only populated
  for `txt`/`xml`). Fixed by using `trafilatura.extract()` for the Markdown body and the separate,
  lighter `trafilatura.extract_metadata()` for the title. Caught by actually running extraction
  against the saved `temp/*.html` pages rather than trusting the signature.
- **Chunk token budget set to 350, not the 500–900 written above.** `BAAI/bge-small-en-v1.5` has a
  hard `max_seq_length` of 512 subword tokens (verified by loading the model) — the 500–900 figure
  from the aioutlooks article was for general LLM context windows, not this embedding model's
  limit. 350 leaves headroom so a chunk is never silently truncated by the encoder.
- **BGE query-prefix convention applied by hand.** BGE's model card calls for prepending
  `"Represent this sentence for searching relevant passages: "` to queries (not documents) for
  asymmetric retrieval. This ST-hub build of the model ships empty `.prompts` (verified), so it
  isn't applied automatically — done explicitly in `rag/embeddings.py`.
- **Hybrid merge uses Reciprocal Rank Fusion (k=60)** to combine the BM25 (FTS5) and vector
  (Chroma) candidate lists — standard, parameter-light way to combine two differently-scaled
  ranking signals — before handing the top 20 to the cross-encoder reranker.
- **`/ingest`, `/search`, `/answer` route handlers are sync `def`, not `async def`.** All three do
  blocking work (SQLite, Chroma, sentence-transformers encode/rerank, trafilatura); FastAPI runs
  sync handlers in its worker thread pool automatically. An `async def` handler runs inline on the
  single shared event loop, so calling blocking code from one would stall every other in-flight
  request — the same async-correctness concern as never calling sync network code from `async def`.
  `/ingest` uses `asyncio.run(...)` internally so the async httpx fetching inside `ingest_section`
  still gets real concurrency within that thread.
- **Fetch concurrency capped at 5 simultaneous requests** (`rag/fetch.py`) — a sitemap section can
  hand back dozens of URLs; firing them all at once would be an unfriendly way to crawl someone
  else's site.
- **`/ingest`, `/search` response schemas extended with concrete fields** (`pages_discovered`,
  `pages_ingested`, `chunks_stored` on ingest; typed `SearchResult`/`Citation` models) now that
  there's a real implementation to describe — the earlier "don't over-specify" stub contracts were
  deliberately minimal only until this point.
- **`/answer` is partially real, not fully stubbed.** Retrieval (hybrid search) is fully decided and
  implemented, so `/answer` runs it and returns genuine citations; only `answer` stays `""`, since
  the LLM provider is still an open decision. This follows directly from "leave `not_implemented`
  only for what's undecided" rather than stubbing the whole endpoint.
- **Removed the root `ruff.toml` created during scaffolding** — it would have shadowed
  `pyproject.toml`'s `[tool.ruff]` section (ruff picks one config source per directory, doesn't
  merge). Unlike readme, this repo has only one Python subtree at the root, so a second config file
  serves no purpose here.
- **`torch` pulled in as a CPU-only wheel, not the default CUDA build.** sentence-transformers'
  default PyPI `torch` wheel drags in ~2GB of `nvidia-cu*`/`triton`/`cuda-toolkit` packages, useless
  on a machine with no usable GPU (confirmed via a CUDA-driver-too-old warning) or in a plain
  container. Fixed via `[tool.uv.sources]` + `[[tool.uv.index]]` pointing `torch` at
  `download.pytorch.org/whl/cpu`. One real gotcha found along the way: this uv version (0.11.26)
  only honors a source override for a package listed directly in `dependencies`, not one pulled in
  purely transitively (verified with an isolated repro) — so `torch` had to be added as a direct
  dependency for the override to take effect, even though sentence-transformers already requires it.

**Testing approach** (per the "mock at external boundaries" convention): `rag/chunk.py`,
`rag/store.py`, `rag/discover.py` are tested directly against real SQLite (`:memory:`) / a real
Markdown string / a mocked HTTP transport (`httpx.MockTransport`) — no mocking of our own code.
The router tests (`tests/api/routers/*`) mock `ingest_section`/`hybrid_search` themselves, since
those pull in real network calls and a ~130MB embedding model — genuine external boundaries for a
fast unit suite. `uv run pytest` — 10 passed.

**Verification performed:**
- `uv run ruff check .` — clean.
- `uv run pytest` — 10/10 passed.
- FTS5 support confirmed both on the host and inside the actual target image
  (`ghcr.io/astral-sh/uv:python3.13-bookworm`) via `docker run`, before committing to the
  SQLite+FTS5 design.
- **Real end-to-end run** (not mocked): `ingest_section("https://www.anthropic.com/engineering")`
  against the live site — 26/26 pages discovered and ingested, 428 chunks stored. A subsequent
  `hybrid_search("how does contextual retrieval reduce failed retrievals")` returned the correct
  passage top-ranked, with an accurate quote ("...reduced the top-20-chunk retrieval failure rate
  by 67% (5.7% → 1.9%)").
- `docker compose up --build` — built, started, and answered `GET /health` with `{"status":"ok"}`
  from inside the container.

**Known limitation found during the real end-to-end run (CORRECTED 2026-07-07 — see below):**
This note originally claimed Trafilatura only "mislabels `heading_path` for the first chunk(s) of
an affected page." **That was wrong — an unverified assertion.** On later inspection Trafilatura
drops *nearly every* real heading (demotes it to a paragraph) and keeps only a "## Get the
developer newsletter" CTA blurb, so on 4 of 5 test pages the CTA was the *only* heading in the
extracted Markdown and every chunk inherited `heading_path = "Get the developer newsletter"`. The
feature was ~100% broken on this site, not cosmetically off. Root-caused and fixed in the
2026-07-07 extractor-swap section below.

## 2026-07-07 — PR review pass, then `/ingest` request-timeout fix

Addressed 9 self-review comments left on PR #1 (documentation/reference gaps, an
unhandled-fetch-exception behavior gap, chunk metadata duplication, an unnecessary import alias) —
see the PR for detail; not repeated here since none changed the architecture.

**Real gap found afterwards: `/ingest` blocked the HTTP response for the entire batch.** The
sync-`def` decision recorded above (`/ingest`, `/search`, `/answer` are sync `def`) was made purely
so a long `/ingest` call doesn't block the shared event loop for *other* concurrent requests. It
was never about the separate problem that the `/ingest` request itself stayed open for the full
duration of discover→fetch→extract→chunk→embed→store across a whole sitemap section (dozens of
pages) — the client's HTTP connection sat open the whole time, with no uvicorn-level timeout
configured and no visibility into partial progress if a proxy or client library's own default
timeout (commonly 30-60s) fired first.

**Fix: accept-and-poll pattern.**
- `POST /ingest` now returns `202 Accepted` immediately with a `job_id`, and hands the actual
  ingestion off to a FastAPI `BackgroundTasks` callback
  (https://fastapi.tiangolo.com/tutorial/background-tasks/) that runs after the response is sent.
- `GET /ingest/{job_id}` polls status (`pending` / `completed` / `failed`) and, once completed,
  returns the same `pages_discovered`/`pages_ingested`/`chunks_stored` counts the old synchronous
  response used to return directly.
- Job state lives in a new `rag/jobs.py`, an in-memory `dict` guarded by a `threading.Lock` (the
  background task runs on FastAPI's worker thread pool, not the event loop, so concurrent access is
  a real possibility). **Deliberately not durable**: job state is lost on process restart and isn't
  shared across multiple worker processes. This is the tradeoff for staying infra-free (no
  Redis/queue) at this project's current scope; revisit with a real task queue (RQ/Celery) if this
  ever needs to survive restarts or scale past one process.
- Verified (not assumed): `TestClient` runs `BackgroundTasks` to completion synchronously within
  the same ASGI call before `.post()` returns, so `tests/api/routers/test_ingest.py` can poll the
  job status immediately afterward and get a deterministic `"completed"`/`"failed"` result — no
  `sleep`/retry loop needed in the test.

## 2026-07-07 — Extractor swap: Trafilatura → readability-lxml + markdownify

**Why.** The `heading_path` metadata was garbage (see the corrected note above): every chunk on
most pages read `"Get the developer newsletter"`. Root cause, verified three independent ways
against the saved `temp/*.html`:
1. Trafilatura's Markdown output contained exactly **one** heading — the newsletter CTA — despite
   the raw HTML having 20-30 real `<h1>`-`<h3>` headings per page.
2. Trafilatura's own XML output tagged only that one element as `<head>`; real headings like
   "Further boosting performance with Reranking" came out as plain `<p>`.
3. No Trafilatura option recovered them (`include_formatting`, `favor_recall`, both combined —
   still one heading).

So Trafilatura's *content-extraction* step (not the Markdown conversion) was demoting headings.
That's a design mismatch: we were relying on it to carry heading structure and it doesn't for this
site.

**Fix — split extraction from conversion.** The canonical 2026 pattern for HTML→Markdown that
preserves structure is: isolate the main article with a readability-style extractor, then convert
that HTML fragment with a faithful Markdown converter.
- `readability-lxml` (`Document(html).summary()`) isolates the article, dropping nav/footer/CTA
  boilerplate while keeping the heading DOM intact.
- `markdownify(..., heading_style="ATX")` converts it to Markdown, preserving the full heading
  hierarchy, tables, and fenced code blocks.
- Title from `Document.short_title()`, stripping a trailing " \ Anthropic" / " | Site" suffix.

**Empirically verified across all 5 saved pages** (headings recovered / CTA junk gone / tables /
code): building-effective-agents 18/✓/–/–, contextual-retrieval 15/✓/–/✓, demystifying-evals
19/✓/✓/✓, multi-agent-research 8/✓/–/–, swe-bench-sonnet 10/✓/✓/✓. End-to-end on
contextual-retrieval, `heading_path` now shows the real nested structure, e.g.
`"Further boosting performance with Reranking > Performance improvements > Cost and latency
considerations"`, with only the pre-heading intro chunk left (honestly) empty.

- Dependencies: removed `trafilatura`; added `readability-lxml`, `markdownify`, and
  `lxml-html-clean` (readability needs `lxml.html.clean`, split into its own package in lxml 5+).
- Added `tests/rag/test_extract.py` — a fixture-based regression guard (no network, no mocks) that
  asserts real `<h2>`/`<h3>` headings survive as ATX Markdown and the site-name title suffix is
  stripped. This is the test that would have caught the original bug.
- Unchanged pre-existing tradeoff: tables/code blocks stay atomic (never split), so a table larger
  than the embedder's 512-token limit is truncated on encode (observed a 528-token table). Not
  addressed here — it's the existing "keep structured blocks whole" decision, independent of the
  extractor swap.

## 2026-07-07 — `/answer` generation implemented with Gemini (gemini-2.5-flash)

The last deferred decision — the LLM provider for `/answer` — is now made:
**Gemini `gemini-2.5-flash`**, the same model the `~/src/readme` book pipeline uses.
Followed readme's core pattern (`google.genai` `Client`, `generate_content`,
`temperature=0.0`, `thinking_config=ThinkingConfig(thinking_budget=0)`) but deliberately
*not* its heavyweight batch machinery (per-job `GeminiJobContext`, `UsageTracker`,
tenacity retry classification, concurrency semaphores) — that exists to process thousands
of chunks per book; a single `/answer` call needs one plain call. Used the **sync** client
(`client.models.generate_content`), not readme's `client.aio.*`, to stay consistent with
this repo's all-sync-`def` handlers running in FastAPI's thread pool.

**RAG flow** (`rag/answer_service.py`): embed + retrieve the query's top-k chunks via the
existing `hybrid_search` → format them as numbered sources `[1] Title > heading_path (url)\n
text` → send to Gemini with a system instruction to answer using only those sources and cite
claims with bracketed `[n]` numbers *like a scientific paper*, saying so plainly when the
sources don't cover the question. Returns the generated answer plus the sources **in citation
order** — `sources[0]` is reference `[1]` — which the router maps to the `citations` list. If
retrieval returns nothing, it skips the LLM entirely and returns an honest "no relevant
sources" message.

- Dependency added: `google-genai>=1.63.0`.
- Config: `GOOGLE_API_KEY` (from the local `.env`) + `LLM_MODEL` in `rag/config.py`; wired
  through `docker-compose.yaml` and documented in `.env.example`. Credential isolation kept:
  the key comes only from this repo's own `.env`, never from readme. A missing key raises a
  clear `RuntimeError` (verified) rather than a cryptic SDK error.
- Tests (`tests/rag/test_answer_service.py`, updated `tests/api/routers/test_query.py`): mock
  the two real external boundaries — the Gemini client (network) and `hybrid_search`
  (retrieval, which needs models + ingested data) — and assert the numbered-sources prompt,
  the citation ordering, and that an empty retrieval skips the LLM. 17 tests pass, ruff clean.
- **Not yet run against the live Gemini API**: this repo has no `GOOGLE_API_KEY` configured
  and borrowing one from another repo is disallowed. Real end-to-end generation is pending the
  user supplying a key in `.env`.

## 2026-07-07 — Sitemap-index bug fix, sitemap library swap, and logging

**Bug:** ingesting `https://langfuse.com/docs` returned no error but ingested
nothing. Root cause: langfuse's `/sitemap.xml` is a **sitemap index**
(`<sitemapindex>` pointing at child sitemaps), not a flat `<urlset>`. The
hand-rolled parser only looked for `<url>` elements, found zero, and returned
an empty list — so discovery silently yielded nothing. Verified by fetching the
sitemap directly (186 bytes, one `<sitemap><loc>` child) and its child
(`sitemap-0.xml`, a `urlset` with 777 URLs, 104 under `/docs`).

**Fix — use a real sitemap library instead of hand-rolling.** Rather than grow
our own parser to cover indexes (and then gzip, robots.txt-declared sitemaps,
plain-text sitemaps, nesting...), switched `rag/discover.py` to
`ultimate-sitemap-parser` (usp). Verified against the live langfuse site: 777
URLs, 104 under `/docs`, matching the manual dig. usp does its own synchronous
HTTP + sitemap discovery from the homepage, so `discover_section_urls` no longer
takes an httpx client and runs usp via `asyncio.to_thread` to keep it off the
event loop. `discover_section_urls(section_url)` is the new signature; we keep
only the path-prefix filtering and logging.

**Logging (loguru).** Added logging across the ingest path so a silent-zero is
never silent again: discovery logs the sitemap, total URLs, and how many match
the prefix (with a WARNING when zero); ingest logs start, fetched count,
per-page skips (DEBUG: no content / no chunks), and a final
`N/M pages, K chunks` summary. Added `core/logging_config.py` with
`configure_logging()` (called from `api/main.py`) that removes loguru's implicit
default handler and installs a stderr sink at `LOG_LEVEL` (default **DEBUG**, so
the per-page skip logs are actually emitted; override with `LOG_LEVEL=INFO` in
production). Matches readme's `configure_logging()` convention. Lives in a new
`core/` package (cross-cutting infra used by both `api/` and `rag/`), not under
the RAG domain package.

- Dependency added: `ultimate-sitemap-parser>=1.4.0`.
- Test rewrite: `tests/rag/test_discover.py` now mocks usp's
  `sitemap_tree_for_homepage` (the library boundary usp fetches through) and
  asserts our prefix filtering + the empty-match case. 18 tests pass, ruff clean.
- Verified end-to-end: `discover_section_urls("https://langfuse.com/docs")` now
  returns 104 URLs with DEBUG logging active.

## 2026-07-07 — Fix `fts5: syntax error` on queries with punctuation/operators

**Bug:** `/search` and `/answer` raised `sqlite3.OperationalError: fts5: syntax
error near "?"` for ordinary queries. Reproduced: a query ending in `?` (i.e.
almost any natural-language question), and also `:`, a bare `-`, or the words
AND/OR/NOT, all crash.

**Cause (not the SQL placeholder).** The `?` in the error is not the bound `?`
parameter -- it's a character in the *value*. `search_bm25` passed raw user text
straight into `WHERE chunks_fts MATCH ?`, and FTS5 parses that argument as a
query *expression* in its own mini-language, where `? : - ( ) "` and AND/OR/NOT
are operators. So `"...retrieval?"` is a malformed expression → syntax error.

**Fix.** New `_to_fts_match_query()` sanitizes free text into a safe FTS5
expression: each whitespace-separated token is wrapped in double quotes (a
literal FTS5 string, immune to the grammar; embedded quotes doubled to escape),
and tokens are joined with **OR**. OR rather than FTS5's implicit AND because a
full question would otherwise require every word -- including stopwords like
"what"/"is" -- to appear in a chunk, so BM25 would return nothing for nearly any
question; with OR, any matching term qualifies and bm25()/the reranker rank it.
Empty/whitespace-only input returns `[]` (an empty MATCH is itself a syntax
error).

- Regression test in `tests/rag/test_store.py`: a `?`-terminated question finds
  its chunk, and operator/punctuation/empty queries don't raise. 19 tests pass.
- Verified in the running container: `POST /search {"query":"what is contextual
  retrieval?"}` now returns results instead of a 500.

## 2026-07-07 — Persist the HuggingFace model cache across container recreates

**Symptom:** the embedding/reranker weights appeared to reload every time Docker
started. Traced it: sentence-transformers caches downloaded weights at
`/root/.cache/huggingface/hub` (129MB bge-small + ~90MB cross-encoder), but that
path was on the container's ephemeral writable layer -- only `/workspace`,
`.venv`, and `data` were volumes. So every `docker compose up --force-recreate`
/ rebuild wiped the cache and re-downloaded ~220MB (measured: 49s cold load).

**Fix:** added a named volume `hf_cache` mounted at `/root/.cache/huggingface`,
with `HF_HOME` pinned to that path (so it holds regardless of the image's HOME).
Verified: after warming the cache once, a `--force-recreate` keeps all 217MB, and
a subsequent load with `HF_HUB_OFFLINE=1` (zero network) completes in 5.3s.

Note on the "Loading weights" progress bar: that ~5s is the in-RAM load (weights
read from disk into memory), which happens on every worker process start and is
inherent -- uvicorn `--reload` re-loads on code changes. The volume eliminates
the expensive *re-download* (~44s), not the in-RAM load.

## 2026-07-07 — Langfuse observability: compose profile + per-stage tracing

Added Langfuse (self-hosted) as an on-demand Compose profile and instrumented the
RAG pipeline so every `/search` and `/answer` call emits a nested trace. Design +
plan: `docs/superpowers/{specs,plans}/2026-07-07-langfuse-observability*.md`.

**What changed.** `hybrid_search`'s stages were extracted into `@observe`-decorated
helpers (`_bm25_retrieve`/`_vector_retrieve` as `as_type="retriever"`, `_rrf_fuse`,
`_rerank`); the Gemini call in `answer_service.py` became an
`@observe(as_type="generation")` helper that logs model + token usage. The Gemini
client factory `get_client` was renamed `get_client_gemini` to avoid colliding with
`langfuse.get_client`. `api/main.py` gained a lifespan that flushes traces on
shutdown. The 6-container Langfuse v3 stack (web/worker/postgres/clickhouse/redis/
minio) sits behind `profiles: ["langfuse"]`; only `langfuse-web` publishes a host
port (3000), the rest stay on the compose network to avoid host port clashes.

**Design decisions (see spec's log for the full list).**
- Self-host over Cloud: interview demo, free, self-contained (no live-network dep).
- `api` does *not* `depends_on` Langfuse -- relies on SDK graceful degradation.
- api's `LANGFUSE_*` keys default to **empty** so plain `docker compose up` is silent;
  the dev keys go in `.env` (documented) to enable tracing. Chosen over defaulting
  keys on, which would spam connection errors in the non-profile path.
- `LANGFUSE_INIT_*` on `langfuse-web` auto-provisions the org/project/user + a fixed
  dev key pair on first boot, so the demo needs zero UI setup.

**Verified end-to-end** (WSL2, Docker 29.6, 15 GiB RAM). Assumption A1 (empty keys
no-op rather than raise) confirmed at runtime -- the SDK logs a disable warning and
`@observe` becomes a pass-through.
- `docker compose config`: default profile = only `api`; `--profile langfuse` = api
  + the 6 services. All 21 tests pass, ruff clean, behavior preserved.
- `docker compose --profile langfuse up`: all 7 containers healthy; langfuse-web
  `/api/public/health` → `{"status":"OK","version":"3.205.1"}`. Headless init created
  the `basic-rag-pipeline` project under `default-org`; the pk-lf-.../sk-lf-... pair
  authenticates against `/api/public/projects`.
- `POST /answer` produced this trace (via `/api/public/traces/<id>`), 7 observations
  nested exactly as designed -- the sync-handler/threadpool context propagated fine:
  ```
  answer [SPAN]
  ├── hybrid-search [SPAN]
  │   ├── bm25-retrieve   [RETRIEVER]
  │   ├── vector-retrieve [RETRIEVER]
  │   ├── rrf-fuse        [SPAN]
  │   └── rerank          [SPAN]
  └── gemini-generate     [GENERATION] gemini-2.5-flash  in=346 out=48 total=394
  ```
- Graceful degradation: `docker compose down` then plain `docker compose up` (no
  profile, empty keys) -- `/answer` still answers and `docker compose logs api` shows
  **no** Langfuse errors/warnings.

**Not built (Phase 2, designed only):** evaluation harness (Langfuse Datasets +
retrieval metrics from `docs/evals.md`, LLM-as-judge on answers). Blocked on a
hand-labeled query→relevant-chunk dataset. See spec section 4 and `docs/langfuse.md`.

## 2026-07-09 — Benign sitemap parse errors during ingest (not a bug)

**Symptom.** Ingesting a site (e.g. `https://www.plenti.com.au/...`) floods the api
logs with two repeated errors before ingestion actually succeeds:

```
Unable to gunzip response for .../sitemap.xml.gz ...: Not a gzipped file (b'<!')
Parsing sitemap from URL .../sitemap failed: Sitemap contained unexpected
  non-standard XML DOCTYPE. Parsing not supported for security reasons.
```

**These are harmless** -- the run still finishes (`Sitemaps listed 576 URL(s); 29
match prefix ...` → `Fetched 29 page(s)`). The lines come from the
`ultimate-sitemap-parser` (usp) library's own stdlib `logging`, not our code (they
lack our loguru timestamp/level format).

**Root cause.** `discover_section_urls` delegates to `usp.tree.sitemap_tree_for_homepage`
(`rag/discover.py:25`). When robots.txt doesn't declare a sitemap, usp brute-force
**probes ~15 well-known sitemap paths** (`usp/tree.py:23` `_UNPUBLISHED_SITEMAP_PATHS`:
`sitemap.xml(.gz)`, `sitemap_index.xml(.gz)`, `sitemap-news.xml(.gz)`,
`admin/config/search/xmlsitemap`, ...). The site returns an **HTML soft-404 page**
(body starts `<!DOCTYPE html>`) for the paths that don't exist, instead of a 404.
That one fact yields both messages:
- `.gz` candidates: usp tries to gunzip HTML → fails; `b'<!'` is the first two bytes
  of `<!DOCTYPE html>` (`usp/helpers.py:283`), then falls back to XML parsing.
- XML parse of the HTML: usp's security-hardened parser refuses any document with a
  `<!DOCTYPE>` (XXE / billion-laughs guard, `usp/fetch_parse.py:460`).

usp logs each failed guess and moves on; it still finds the real sitemap.

**Decision: keep the noise.** No code change. Silencing is a one-liner if it ever
becomes annoying -- `logging.getLogger("usp").setLevel(logging.CRITICAL)` in
`core/logging_config.py:configure_logging()` (CRITICAL, not ERROR, since these are
logged at error level but are expected).
