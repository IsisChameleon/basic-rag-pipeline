# Oversized table/code blocks are truncated on embedding

**Status:** open — pre-existing tradeoff, not yet addressed
**Found:** 2026-07-07 (during the Trafilatura → readability-lxml + markdownify extractor swap)
**Area:** `rag/chunk.py`, `rag/embeddings.py`

## Problem

The chunker treats tables and fenced code blocks as **atomic** — they are never
split, even when a single one exceeds the target token budget
([rag/chunk.py:167-170](../../rag/chunk.py)). The embedding model
(`BAAI/bge-small-en-v1.5`) has a hard `max_seq_length` of **512 tokens**, and
`SentenceTransformer.encode` silently truncates anything longer rather than
erroring.

So a table (or code block) larger than 512 tokens is embedded from only its
first 512 tokens; the rest never contributes to the vector. The chunk's full
text is still stored in SQLite and returned by BM25/at retrieval time — only the
**vector** for that chunk is computed from a truncated input, degrading semantic
recall for the tail of a large table.

## Evidence

Observed during the real end-to-end run on
`demystifying-evals-for-ai-agents`, where a large comparison table produced:

```
[transformers] Token indices sequence length is longer than the specified
maximum sequence length for this model (528 > 512). Running this sequence
through the model will result in indexing errors
```

The 528-token table is embedded from its first 512 tokens only.

## Why it's left as-is for now

This is a direct consequence of the deliberate "keep structured blocks whole"
decision — splitting a table mid-row or a code block mid-statement would produce
chunks that are individually meaningless. The truncation only affects the
**vector** for oversized structured blocks (a minority of chunks); their text is
intact for lexical (BM25) search and for display. It is independent of the
extractor swap that surfaced it.

## Possible fixes (not yet decided)

- **Row-group tables with a repeated header.** Split a large table into
  multiple chunks, each carrying the header row(s) so every piece is
  self-describing and independently embeddable.
- **Summarize-then-embed.** Embed a short generated summary of the oversized
  block instead of its raw text, while still storing the full text for display.
- **Detect and warn/skip.** At minimum, log which chunks exceed the limit at
  ingest time so truncation isn't silent.
- **Larger-context embedding model.** Switch to an embedding model with a
  higher `max_seq_length` (raises the threshold but doesn't remove it, and
  changes the chunk-size budget in `rag/chunk.py`).
