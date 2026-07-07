# Retrieval evaluation metrics (notepad)

Candidate metrics to measure `/search` retrieval quality (given a set of
queries each with known relevant chunks):

- **Precision@k** — of the top k retrieved, what fraction are relevant.
- **Recall@k** — of all relevant chunks, what fraction appear in the top k.
- **Hit Rate@k** — did at least one relevant chunk make the top k (binary, per query).
- **MRR** (Mean Reciprocal Rank) — 1 / rank of the first relevant chunk, averaged over queries. Rewards getting *one* right answer high up.
- **MAP** (Mean Average Precision) — averages Precision@k at each relevant hit. Rewards getting *all* relevant chunks high up.
- **NDCG@k** (Normalized Discounted Cumulative Gain) — see below. The main one when relevance is graded, not just yes/no.
- (RAG-specific, e.g. RAGAS) **Context Precision / Context Recall** — how much of the retrieved context is relevant / how much of the needed context was retrieved.

---

## NDCG@k explained

**What it solves:** the metrics above are mostly binary (relevant / not) and
either ignore ordering (Precision@k) or only look at the first hit (MRR). NDCG
captures two things at once:
1. **Graded relevance** — a chunk can be perfect (3), good (2), marginal (1), irrelevant (0), not just yes/no.
2. **Position** — a relevant chunk at rank 1 is worth more than the same chunk at rank 5, because the top is seen first (by the user, or by the LLM's context window).

**Built in three steps:**

**1. DCG@k (Discounted Cumulative Gain)** — sum each result's gain, discounted by
a log of its rank so lower positions count for less:

```
DCG@k = Σ (i=1..k)  (2^rel_i − 1) / log2(i + 1)
```

- **Gain** `2^rel_i − 1` (exponential gain): rewards highly-relevant chunks steeply — a grade-3 (gain 7) is worth far more than a grade-1 (gain 1). (A simpler linear variant just uses `rel_i`; the exponential one is the common default and what the example below uses.)
- **Discount** `log2(i + 1)`: rank 1 → /1.00, rank 2 → /1.58, rank 3 → /2.00, … so a relevant chunk slipping down the list keeps less of its value.

**2. IDCG@k (Ideal DCG)** — the DCG of the *best possible* ordering for that query
(take the truly relevant chunks, sort most-relevant-first, compute DCG). This is
the ceiling.

**3. NDCG@k** — normalize so scores are comparable across queries (a query with
many highly-relevant chunks would otherwise post a bigger raw DCG than a sparse one):

```
NDCG@k = DCG@k / IDCG@k        (0..1; 1.0 = your ranking already is the ideal one)
```

### Worked example (k=3, grades 0–3, exponential gain)

  Say the true relevances of your top 3 results are [3, 0, 2] (perfect doc first, junk second, relevant doc third):

  DCG  = (2³−1)/log₂2 + (2⁰−1)/log₂3 + (2²−1)/log₂4
       =   7/1.0      +   0/1.585     +   3/2.0
       = 7 + 0 + 1.5 = 8.5

  Ideal order would be [3, 2, 0]:
  IDCG = 7/1.0 + 3/1.585 + 0/2.0 = 7 + 1.89 = 8.89

  nDCG@3 = 8.5 / 8.89 ≈ 0.956

    Close to 1 — the ranking is nearly ideal; it's only slightly off because the relevant doc (grade 2) sits below the junk doc instead of above it.

**Why it fits this pipeline:** the reranker's whole job is ordering, so a metric
that is both grade- and position-aware is the natural way to tell whether the
BM25 + vector + rerank stack puts the best chunks on top, not just somewhere in
the top k.
