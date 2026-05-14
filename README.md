# Raggit

> Evals for RAG, against the queries your users actually send.

Most RAG evals are written once against synthetic queries, then forgotten. You swap embedding models, ship it, and three weeks later discover retrieval got worse on the queries that actually matter.

Raggit closes that loop:

1. **Monitor** production queries — cluster similar ones in the background.
2. **Inspect** which clusters are popular and worth covering.
3. **Write evals** against the queries that matter.
4. **Re-run** when you change models — compare reports side-by-side.

It's not a benchmark. It's evals on your data, against queries you've seen.

---

## The loop

```python
from raggit import EvalSuite, embedding_eval
from raggit.middleware import Middleware, Monitor

embed = lambda t: model.encode(t).tolist()

# 1. Wrap your retrieval — monitor logs every query in the background
monitor = Monitor(embedder=embed)
mw = Middleware(monitor=monitor, embedder=embed)

@mw.track
def retrieve(query: str) -> str:
    return index.search(query)[0]

# 2. After collecting traffic, see what's popular
for cluster in monitor.popular_queries(top=10):
    print(cluster.count, cluster.representative_query)

# 3. Write evals against the queries that matter
corpus_vecs = [embed(d) for d in docs]
suite = (
    EvalSuite(name="prod_smoke")
    .add("reset password", embedding_eval(
        embed("how do I reset my password"),
        embed("Reset via Settings → Security"),
        corpus_vecs,
    ))
)

# 4. Switch embedding model, re-run, compare
suite.run().show()
```

```
─────────────────────────── Raggit Eval Suite ───────────────────────────
  Suite : prod_smoke
  Date  : 2026-04-30 11:42

  Eval                                        Passed   Rank   Score
 ────────────────────────────────────────────────────────────────────
  reset password                                ✓        1     0.94
  log into account                              ✗        4     0.61
  refund policy                                 ✓        2     0.88

  Total: 3  |  Passed: 2  |  Failed: 1  |  Pass rate: 66.7%
─────────────────────────────────────────────────────────────────────────
```

---

## Who this is for

**Good fit:**
- Small/mid teams running RAG in production who want regression coverage on real traffic.
- Anyone evaluating an embedding-model swap (e.g. `text-embedding-3-small → large`) and wanting empirical proof on their own queries.
- Self-hosted setups — SQLite by default, Postgres + pgvector when you scale.

**Not a good fit:**
- Already on LangSmith / Phoenix / Langfuse and happy — overlap isn't worth the switch.
- You only need observability without eval — use a dedicated APM.
- You only need an eval harness with no traffic loop — Ragas may be lighter.

---

## How this differs

The closest tools eval the *answer* the LLM produces. Raggit evals the *retrieval* that feeds it.

- **LangSmith / Langfuse / Phoenix** — observability platforms with LLM-as-judge evals on final outputs. Raggit's monitor overlaps with their tracing, but its evals operate one layer earlier: did the right doc get retrieved? did the new embedding model still rank it #1?
- **Ragas / OpenAI Evals / DeepEval** — eval harnesses scoring LLM answers (faithfulness, relevance, etc.). Raggit is complementary — you can be confident in your generator and still ship a regression if retrieval silently breaks. Catch it before the LLM sees it.

The wedge: retrieval-layer evals on the queries your users actually send, on infrastructure you own.

---

## Installation

**pip**
```bash
pip install raggit               # SQLite backend
pip install 'raggit[postgres]'   # Postgres + pgvector backend
```

**uv**
```bash
uv add raggit
uv add 'raggit[postgres]'
```

No embedding dependencies — bring your own embedder for any modality (text, audio, image, video). Any callable `str → list[float]` works.

---

## Evals

### embedding_eval

Embedding model retrieval quality against a corpus.

```python
from raggit import EvalSuite, embedding_eval

report = (
    EvalSuite("password_reset")
    .add("reset password", embedding_eval(
        query_vec=embed("How do I reset my password?"),
        expected_vec=embed("Visit Settings → Security to reset."),
        corpus_vecs=corpus_vecs,
        k=3,
    ))
    .run()
)
report.show()
```

### index_eval

Tests any search backend. `search_fn` takes a query vector and returns a ranked list of vectors. Works with Faiss, Chroma, BM25, hybrid retrieval, etc.

```python
import faiss, numpy as np
from raggit import index_eval

index = faiss.IndexFlatL2(dim)
index.add(np.array(corpus_vecs, dtype="float32"))

def faiss_search(query_vec):
    _, idx = index.search(np.array([query_vec], dtype="float32"), k=10)
    return [corpus_vecs[i] for i in idx[0]]

suite.add("faiss", index_eval(query_vec, expected_vec, faiss_search, k=3))
```

### chunk_eval

Tests whether expected content survives a chunking strategy.

```python
from raggit import chunk_eval

def chunker(text: str, overlap: float) -> list[str]:
    size = 512
    step = max(1, int(size * (1 - overlap)))
    return [text[i:i+size] for i in range(0, len(text), step)]

suite.add("overlap=0.25", chunk_eval(doc, expected_vec, chunker, embed, overlap=0.25))
```

### Custom evals

Any `Callable[[], EvalSingleResult]` works.

```python
from raggit import EvalSingleResult

def my_eval() -> EvalSingleResult:
    return EvalSingleResult(passed=score > 0.8, score=score, metric_name="custom")

suite.add("custom", my_eval)
```

---

## Metrics

Similarity (passed as `metric=` to evals):

| | |
|---|---|
| `Metrics.cosine_similarity` | default, best for normalized embeddings |
| `Metrics.dot_product` | fast, good for unit vectors |
| `Metrics.euclidean_similarity` | `1 / (1 + distance)` |

Retrieval aggregations (post-run):

```python
from raggit import RetrievalMetrics

report = (
    suite.run()
    .aggregate(RetrievalMetrics.mrr,         name="avg_mrr")
    .aggregate(RetrievalMetrics.recall_at_k, name="avg_recall")
    .aggregate(RetrievalMetrics.ndcg,        name="avg_ndcg")
)
```

---

## Production layers

Once you're collecting traffic, raggit ships optional layers on top of the eval loop. None of them are required — the eval loop above works on its own.

### Monitor

Clusters similar queries so you can see what production traffic looks like.

```python
from raggit.middleware import Middleware, Monitor, SQLiteMonitorStore

monitor = Monitor(
    embedder=embed,
    store=SQLiteMonitorStore(".raggit/monitor.db"),
    cluster_threshold=0.92,
)
mw = Middleware(monitor=monitor, embedder=embed)

@mw.track
def retrieve(query: str) -> str:
    return index.search(query)[0]

# Optional: pass search-time metadata at the call site
retrieve("my query", _monitor_kwargs={
    "retrieval_score": 0.91,
    "retrieved_doc_ids": ["doc_42"],
})

monitor.popular_queries(top=10)
monitor.popular_queries(min_count=5)
```

Stores:

| Store | Use when |
|---|---|
| `SQLiteMonitorStore` | full per-query history (default) |
| `SQLiteClusterStore` | aggregate counts only, no history |
| `PostgresMonitorStore` | production scale (HNSW-indexed pgvector) |

### Feedback

Record thumbs-up/down or numeric scores per retrieval. Surfaces clusters with low acceptance — your priority queue for new evals.

```python
from raggit.middleware import Monitor, SQLiteMonitorStore, SQLiteEventFeedbackStore

monitor = Monitor(
    store=SQLiteMonitorStore(db),
    feedback_store=SQLiteEventFeedbackStore(db),
    embedder=embed,
)
mw = Middleware(monitor=monitor, embedder=embed)

@mw.track_with_handle
def retrieve(query: str) -> str:
    return index.search(query)[0]

handle = retrieve("how do I reset my password")
monitor.record_feedback(handle, accepted=True, score=0.8, comment="great")

for cluster, rate in monitor.acceptance_rate_per_cluster(top=10):
    print(f"{rate:.0%}  {cluster.count}x  {cluster.representative_query!r}")
```

`SQLiteEventFeedbackStore` keeps per-event rows (comments preserved). `SQLiteClusterFeedbackStore` keeps aggregate counters only (comments dropped). Raggit doesn't ship an HTTP server — wire `record_feedback` into your framework's route handler.

### Semantic cache

Preset responses for high-frequency clusters. Future similar queries skip retrieval.

```python
from raggit.middleware import SemanticCache, SQLiteCacheStore

cache = SemanticCache(SQLiteCacheStore(".raggit/cache.db"), embedder=embed, threshold=0.95)
mw = Middleware(monitor=monitor, cache=cache, embedder=embed)

cache.set("How do I reset my password?", "Settings → Security.")
```

### Automatic cache promotion

Promote answers into the cache once a cluster crosses count + acceptance + score thresholds. Demoted automatically if quality drops. Only promoter-managed entries are eligible for demotion — `cache.set(...)` entries stay forever.

```python
from raggit.middleware import AutoCachePromoter

promoter = AutoCachePromoter(
    cache=cache, store=store, feedback_store=feedback_store,
    min_count=10, min_acceptance=0.8, min_score=None,
)

monitor = Monitor(embedder=embed, store=store, feedback_store=feedback_store, auto_promoter=promoter)
mw      = Middleware(monitor=monitor, cache=cache, embedder=embed, auto_promoter=promoter)
```

Thresholds are AND. `None` means ignore that dimension.

**Privacy:** wiring a promoter means response text is persisted in `clusters.latest_response`. Don't wire it if your responses are sensitive — without a promoter that column stays NULL.

### Postgres + pgvector

SQLite uses an in-memory linear scan — fine for hobby projects, O(n) at scale. The Postgres backend stores vectors in a pgvector column with an HNSW index for O(log n) ANN search.

```bash
pip install raggit[postgres]
```

```python
from raggit.middleware import PostgresMonitorStore, PostgresEventFeedbackStore, PostgresCacheStore

dsn, DIM = "postgresql://user:pass@host/raggit", 1536
store          = PostgresMonitorStore(dsn, dim=DIM)
feedback_store = PostgresEventFeedbackStore(dsn, dim=DIM)
cache          = SemanticCache(PostgresCacheStore(dsn, dim=DIM), embedder=embed, threshold=0.95)
```

`pgvector` must be available on the database — most managed Postgres providers (RDS, Cloud SQL, Supabase, Neon, Railway) ship it. `dim` is required at construction time so the column type is `VECTOR(dim)` and the HNSW index can be built. Each store opens its own internal pool — call `store.close()` on shutdown.

### Custom stores

Implement `MonitorStore` for any backend (DynamoDB, Mongo, etc.):

```python
from raggit.middleware import MonitorStore

class MyStore(MonitorStore):
    def get_schema(self) -> dict[str, type]: ...
    def assign_cluster(self, vec, threshold, query) -> str: ...
    def log(self, query, vec, latency_ms, threshold, **kwargs): ...
```

---

## Roadmap

- [x] `embedding_eval`, `index_eval`, `chunk_eval`
- [x] `EvalSuite` with Rich terminal reports
- [x] Custom similarity metrics + post-run aggregations
- [x] Monitor with query clustering + `popular_queries()`
- [x] Feedback integration with per-cluster acceptance/score
- [x] Auto cache promotion + demotion
- [x] Postgres + pgvector backend
- [ ] Suite history & diff — persist reports, diff across runs (model A vs B over time)
- [ ] Auto-derive embedder dimension for Postgres
- [ ] CI/CD integration — invoke suite from CI, gate merges on regressions

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
