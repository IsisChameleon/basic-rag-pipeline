# RAG storage & retrieval wiring — design

Date: 2026-07-05

## Goal

Decide where ingested content lives once it's been fetched and extracted
(see `2026-07-05-rag-api-scaffold-design.md` for the endpoint scaffold, and
`build_log.md` for the ingestion-tool spike), and how `/search` and
`/answer` retrieve it. This is a design/decision document — no
storage/retrieval code has been written yet.

## Pipeline

1. **Discover** — `sitemap.xml` → list of page URLs
2. **Fetch** — httpx
3. **Extract** — readability-lxml (isolate the article) → markdownify
   (HTML → Markdown, preserving heading structure) per page. (Originally
   Trafilatura, swapped 2026-07-07 because it dropped nearly all real
   headings — see build_log.md.)
4. **Chunk** — split Markdown on structural boundaries (headings, atomic
   tables/lists — never split a table or list mid-way)
5. **Store chunks** — SQLite (`chunks` table: source of truth for chunk
   text + metadata, used for citations)
6. **Embed** — `BAAI/bge-small-en-v1.5` (bi-encoder)
7. **Store embeddings** — Chroma (embedded, persisted to a volume)
8. **Build BM25 index** — SQLite FTS5 virtual table (external-content mode,
   synced via triggers)
9. **`/search`** — BM25 (FTS5) query + vector (Chroma) query → merge
   candidates → rerank with `cross-encoder/ms-marco-MiniLM-L-6-v2` → return
   ranked chunks
10. **`/answer`** — reranked top-k → prompt an LLM (provider still
    deferred) → answer + citations sourced from stored chunk metadata
    (`url`, `title`)

## Storage decision

Two embedded stores, no extra container:

- **SQLite** — a `chunks` table holds the source-of-truth chunk text and
  metadata (`url`, `title`, `chunk_index`, `content_hash`, `fetched_at`).
  This is also what powers citations in `/answer`.
- **Chroma** — embedded, persisted to a volume, holds the `bge-small-en-v1.5`
  embeddings for vector/ANN search. Chroma is used for exactly one thing:
  the vector index. It is not the source of truth for chunk text.
- **BM25** — provided by SQLite's own FTS5 module (see below), not a
  separate Python library. No `rank_bm25`/`bm25s` dependency.

Rejected alternatives: Chroma-only (drops lexical/keyword matching, which
matters for exact terms/code identifiers/acronyms in technical docs);
Postgres + pgvector + `tsvector`/`ts_rank` (requires an extra db container,
and `ts_rank` isn't literally BM25, just a similar ranking function).

## FTS5

**FTS5** = **F**ull-**T**ext **S**earch, version 5 — SQLite's built-in
full-text search extension module (the current generation; FTS1–FTS4 were
earlier, less capable versions). It ships with SQLite and provides the
`bm25()` ranking function used for lexical retrieval.

### How/when the index gets built

Unlike a normal SQL index, FTS5 is not something SQLite silently maintains
behind an existing table — it's a *virtual table* you declare explicitly,
and the index is built incrementally at write time, not on a schedule or a
separate rebuild pass:

1. **Declaration is explicit.**
   ```sql
   CREATE VIRTUAL TABLE chunks_fts USING fts5(text, content=chunks, content_rowid=id);
   ```
   This creates a special table type whose only job is indexing text.
   Behind the scenes SQLite auto-creates shadow tables (`chunks_fts_data`,
   `chunks_fts_idx`, `chunks_fts_docsize`, `chunks_fts_config`) that hold
   the actual inverted index (term → rows containing it).

2. **Indexing happens at write time, incrementally.** The moment a row is
   inserted into `chunks_fts`, SQLite tokenizes the text (default
   tokenizer `unicode61` — splits on word boundaries, folds
   case/diacritics; `porter` adds English stemming) and updates the
   inverted index immediately. There is no periodic rebuild job and no
   lag — a `MATCH` query reflects whatever was last inserted.

3. **`content=chunks, content_rowid=id` is "external content" mode.** It
   tells FTS5 not to store a second copy of the chunk text — only the
   tokenized index, pointing back at `chunks.id` for the real text. This
   saves disk space (no duplicated text) at the cost of needing manual
   sync: FTS5 has no way to know `chunks` changed unless triggers tell it.

4. **A full rebuild is available on demand** (e.g. after bulk-loading
   `chunks` before indexing) via:
   ```sql
   INSERT INTO chunks_fts(chunks_fts) VALUES ('rebuild');
   ```

### Required sync triggers (external-content mode)

Because `chunks_fts` doesn't store the text itself, every write to
`chunks` needs a matching trigger, or the FTS index silently goes stale:

```sql
CREATE TRIGGER chunks_ai AFTER INSERT ON chunks BEGIN
  INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
END;

CREATE TRIGGER chunks_ad AFTER DELETE ON chunks BEGIN
  INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.id, old.text);
END;

CREATE TRIGGER chunks_au AFTER UPDATE ON chunks BEGIN
  INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.id, old.text);
  INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
END;
```

The delete/update triggers use FTS5's special `'delete'` command form
(passing the *old* rowid + text) because a plain `DELETE FROM chunks_fts
WHERE rowid=...` is not how entries are removed from an external-content
FTS5 index — the module needs the old text to correctly unwind the
inverted index.

## Explicitly out of scope (still deferred)

- LLM provider for `/answer` (Anthropic/OpenAI/Google)
- Whether ingestion needs a crawl fallback for sites without a sitemap
  (sitemap-driven discovery covers the case tested so far)
