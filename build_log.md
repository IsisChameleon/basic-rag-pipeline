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

**Known limitation found during the real end-to-end run, not fixed:** Trafilatura occasionally
leaks a short non-article snippet into the extracted Markdown as a heading — on
`contextual-retrieval`, a "## Get the developer newsletter" CTA blurb appears before the real
content. Since our chunker attributes everything up to the *next* heading to whatever heading
came before it, this mislabels `heading_path` for the first chunk(s) of an affected page (the
retrieved chunk *text* itself is unaffected and was verified correct in the query above — only the
heading-path metadata is wrong). Not fixed: a targeted fix would mean pruning specific boilerplate
patterns, which cuts against keeping ingestion generic across arbitrary doc sites rather than
tuned to Anthropic's site specifically. Left as a documented caveat.
