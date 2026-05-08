# Raggit

> You updated your embedding model and your RAG got worse — but you didn't know until users complained.
>
> Raggit lets you see what your production traffic actually looks like, write evals against the queries that matter, and re-run them whenever you change models — so you know on your own data whether the swap helped or hurt.
>
> It's not a benchmark — it's evals against the queries your users actually send.

---

## The loop

Most RAG failures are invisible. The pipeline runs, something gets retrieved, an answer is generated — but nobody checks if the model swap broke retrieval on the queries users actually send.

Raggit closes the feedback loop:

1. **Monitor** production queries — cluster similar ones, log search metadata per event
2. **Inspect** which clusters of queries are most popular and which docs are most retrieved
3. **Write evals** against the queries you care about
4. **Re-run** them when you change models — compare reports side-by-side on the queries you care about

```python
from raggit.middleware import Middleware, Monitor, SQLiteMonitorStore
from raggit import EvalSuite, embedding_eval

embed = lambda t: model.encode(t).tolist()

# 1. Wrap your retrieval function — monitor logs every query in the background
monitor = Monitor(store=SQLiteMonitorStore(".raggit/monitor.db"), embedder=embed)
middleware = Middleware(monitor=monitor, embedder=embed)

@middleware.track
def retrieve(query: str) -> str:
    docs, scores, ids = index.search(query)
    return docs[0]

# Optional: pass search-time metadata at the call site
answer = retrieve(
    "how do I reset my password",
    _monitor_kwargs={
        "retrieval_score": 0.91,         # top-1 confidence the retriever returned
        "retrieved_doc_ids": ["doc_42"], # what came back from the index
    },
)

# 2. After collecting data, see what's popular
for cluster in monitor.popular_queries(top=10):
    print(cluster.count, cluster.representative_query)

# 3. Write evals against the queries you care about
corpus_vecs = [embed(doc) for doc in docs]
suite = (
    EvalSuite(name="prod_smoke")
    .add("reset password", embedding_eval(
        embed("how do I reset my password"),
        embed("Reset your password via Settings."),
        corpus_vecs,
    ))
)

# 4. Switch to a new model, re-run, compare
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

## Installation

**pip**
```bash
pip install raggit
```

**uv**
```bash
uv add raggit
```

Raggit has no embedding dependencies — bring your own embedder for any modality (text, audio, image, video).

---

## Evals

Write evals manually against the queries you want to lock in. Use `monitor.popular_queries()` to see which clusters of queries are worth covering.

### embedding_eval

Tests an embedding model's retrieval quality against a corpus.

```python
from raggit import EvalSuite, embedding_eval

corpus_vecs = [embed(doc) for doc in docs]

report = (
    EvalSuite(name="password_reset")
    .add("reset password", embedding_eval(
        query_vec=embed("How do I reset my password?"),
        expected_vec=embed("Visit the login page to reset your password."),
        corpus_vecs=corpus_vecs,
        k=3,
    ))
    .run()
)
report.show()
```

### index_eval

Tests any search index. `search_fn` receives the query vector and returns a ranked list of vectors. Compatible with Faiss, Chroma, BM25, or any other backend.

```python
index_eval(
    query_vec=embed("How to activate my account?"),
    expected_vec=embed("To activate your account..."),
    search_fn=my_search_fn,   # Callable[[List[float]], List[List[float]]]
    k=3,
)
```

**Faiss example:**
```python
import faiss, numpy as np

index = faiss.IndexFlatL2(dim)
index.add(np.array(corpus_vecs, dtype="float32"))

def faiss_search(query_vec):
    _, indices = index.search(np.array([query_vec], dtype="float32"), k=10)
    return [corpus_vecs[i] for i in indices[0]]

suite.add("faiss", index_eval(query_vec, expected_vec, faiss_search))
```

**Chroma example:**
```python
import chromadb

collection = chromadb.Client().create_collection("docs")
collection.add(embeddings=corpus_vecs, ids=[str(i) for i in range(len(corpus_vecs))])

def chroma_search(query_vec):
    return collection.query(query_embeddings=[query_vec], n_results=10)["embeddings"][0]

suite.add("chroma", index_eval(query_vec, expected_vec, chroma_search))
```

### chunk_eval

Tests a chunking strategy. Checks whether expected content survives chunking by comparing vectors.

```python
def my_chunker(text: str, overlap: float) -> list[str]:
    size = 512
    step = max(1, int(size * (1 - overlap)))
    return [text[i:i + size] for i in range(0, len(text), step)]

report = (
    EvalSuite(name="chunking")
    .add("overlap=0.0",  chunk_eval(document, expected_vec, my_chunker, embed, overlap=0.0))
    .add("overlap=0.25", chunk_eval(document, expected_vec, my_chunker, embed, overlap=0.25))
    .run()
)
```

### Custom evals

Any `Callable[[], EvalSingleResult]` works as an eval.

```python
from raggit import EvalSingleResult

def my_eval() -> EvalSingleResult:
    score = run_my_custom_check()
    return EvalSingleResult(passed=score > 0.8, score=score, metric_name="custom")

EvalSuite().add("custom", my_eval).run()
```

---

## Monitor

The monitor wraps your retrieval function and clusters similar queries so you can see what production traffic actually looks like.

```python
from raggit.middleware import Middleware, Monitor

# Zero-config: defaults to SQLiteMonitorStore(".raggit/middleware.db")
monitor = Monitor(embedder=embed)
middleware = Middleware(monitor=monitor, embedder=embed)
```

Or pass a store explicitly when you want to control the path or threshold:

```python
from raggit.middleware import Middleware, Monitor, SQLiteMonitorStore

monitor = Monitor(
    embedder=embed,
    store=SQLiteMonitorStore(".raggit/monitor.db"),
    cluster_threshold=0.92,
)
middleware = Middleware(monitor=monitor, embedder=embed)

@middleware.track
def retrieve(query: str) -> str:
    docs, scores, ids = index.search(query)
    return docs[0]

# Optional: pass search-time metadata at the call site
result = retrieve("my query", _monitor_kwargs={
    "retrieval_score": 0.91,         # top-1 confidence
    "retrieved_doc_ids": ["doc_42"], # what came back from the index
})

# Inspect what's popular
monitor.stats()
monitor.popular_queries(top=10)                 # top-N by count
monitor.popular_queries(min_count=5)            # everything seen ≥5 times
monitor.popular_queries(top=10, min_count=5)    # top-10 of those seen ≥5 times
```

### Store types

| Store | Use when |
|---|---|
| `SQLiteMonitorStore` | Full per-query history (events + clusters) |
| `SQLiteClusterStore` | Aggregate counts only, no per-query history |
| `SQLiteCacheStore` | Semantic cache |

### Semantic cache

Once the monitor identifies high-frequency clusters, you can preset responses for them. Future similar queries return the cached response without hitting your LLM.

```python
from raggit.middleware import SemanticCache, SQLiteCacheStore

cache = SemanticCache(store=SQLiteCacheStore(".raggit/cache.db"), embedder=embed, threshold=0.95)
middleware = Middleware(monitor=monitor, cache=cache, embedder=embed)

# Manually approve a cached response for a high-frequency cluster
cache.set("How do I reset my password?", "To reset your password, go to Settings → Security.")
```

### Extra fields per event

Add columns to your `events` table, then pass them via `_monitor_kwargs`:

```sql
ALTER TABLE events ADD COLUMN user_id TEXT;
```

```python
retrieve("my query", _monitor_kwargs={"user_id": "abc123"})
```

`Monitor` validates field names and types against the store schema at log time.

### Custom stores

Implement `MonitorStore` to use any backend:

```python
from raggit.middleware import MonitorStore

class MyDynamoStore(MonitorStore):
    def get_schema(self) -> dict[str, type]:
        return {"user_id": str}

    def assign_cluster(self, vec, threshold, query) -> str:
        ...  # find or create cluster, bump count, return cluster_id

    def log(self, query, vec, latency_ms, threshold, cache_hit=False,
            cluster_id=None, event_id=None, **kwargs):
        ...  # write to DynamoDB; if cluster_id is given, skip clustering
```

---

## Feedback

Once you're collecting production queries, the next signal you want is whether
the answers were any good. Feedback lets users (or your own code) record
thumbs-up/down or a numeric score against a specific retrieval, so you can see
which clusters of queries have low acceptance — i.e. which ones are worth
writing evals against next.

Feedback persistence is a separate concern from monitor persistence. You pair
a `MonitorStore` with a matching `FeedbackStore` at the `Monitor` layer:

```python
from raggit.middleware import (
    Middleware, Monitor, SQLiteMonitorStore, SQLiteEventFeedbackStore,
)

# Both stores typically share the same DB file.
db = ".raggit/monitor.db"
monitor = Monitor(
    store=SQLiteMonitorStore(db),
    feedback_store=SQLiteEventFeedbackStore(db),
    embedder=embed,
)
mw = Middleware(monitor=monitor, embedder=embed)

# Use track_with_handle instead of track when you want feedback later.
@mw.track_with_handle
def retrieve(query: str) -> str:
    return my_index.search(query)[0]

handle = retrieve("how do I reset my password")
print(handle.answer)         # the answer your users see
print(handle.cluster_id)     # always set
print(handle.event_id)       # set when the store keeps event history

# Later, when the user clicks 👍 / 👎 / leaves a star rating:
monitor.record_feedback(handle, accepted=True)                  # thumb-only
monitor.record_feedback(handle, score=0.8)                      # rating-only
monitor.record_feedback(handle, accepted=True, score=0.8, comment="great")

# Surface clusters with low acceptance — your priority queue for new evals
for cluster, rate in monitor.acceptance_rate_per_cluster(top=10):
    print(f"{rate:.0%}  {cluster.count}x  {cluster.representative_query!r}")

for cluster, avg in monitor.avg_score_per_cluster(top=10):
    print(f"{avg:.2f}  {cluster.count}x  {cluster.representative_query!r}")
```

At least one of `accepted` or `score` must be provided — calling
`record_feedback(handle)` with no signal raises `ValueError`. Calling it
without a `feedback_store` wired into the `Monitor` also raises a clear
error.

### Pairing tables

The two `FeedbackStore` implementations match the two `MonitorStore` philosophies:

| MonitorStore | Pair with | What gets stored | Comments preserved? |
|---|---|---|---|
| `SQLiteMonitorStore` | `SQLiteEventFeedbackStore` | Per-event rows in a `feedback` table (FK to `events`) | yes |
| `SQLiteClusterStore` | `SQLiteClusterFeedbackStore` | Counter columns appended to the `clusters` table | dropped |

`SQLiteClusterStore` was chosen for "aggregates only, no history" — its
paired feedback store respects that. Comments are silently dropped because
there's no row to attach them to. If you need per-event feedback or comment
retention, use the event-flavored pair.

Both feedback stores are idempotent on init and can be instantiated in any
order relative to their paired `MonitorStore` — they share the same DB and
each ensures its own schema exists.

### Wiring feedback to your own HTTP route

Raggit doesn't ship an HTTP server. Wire `record_feedback` into your existing
framework's route handler:

```python
@app.post("/feedback")
def feedback(req: FeedbackReq):
    handle = handle_cache.pop(req.handle_token)  # however your app stores it
    monitor.record_feedback(
        handle,
        accepted=req.accepted,
        score=req.score,
        comment=req.comment,
    )
```

---

## Automatic cache promotion

Once a cluster has enough volume *and* enough positive signal, the answer
gets promoted into the semantic cache automatically. Future similar queries
are served from cache without re-running your retrieval. If quality later
drops, the cache entry is evicted.

```python
from raggit.middleware import (
    AutoCachePromoter, Middleware, Monitor, SemanticCache,
    SQLiteCacheStore, SQLiteEventFeedbackStore, SQLiteMonitorStore,
)

db = ".raggit/middleware.db"
cache = SemanticCache(SQLiteCacheStore(db), embedder=embed, threshold=0.95)
store = SQLiteMonitorStore(db)
feedback_store = SQLiteEventFeedbackStore(db)

promoter = AutoCachePromoter(
    cache=cache,
    store=store,
    feedback_store=feedback_store,
    min_count=10,           # at least 10 events in the cluster
    min_acceptance=0.8,     # ≥80% thumb-up rate (set to None to ignore)
    min_score=None,         # average score floor (set to a value to gate)
)

monitor = Monitor(
    embedder=embed, store=store, feedback_store=feedback_store,
    auto_promoter=promoter,        # monitor calls promoter on feedback
)
mw = Middleware(
    monitor=monitor, cache=cache, embedder=embed,
    auto_promoter=promoter,        # middleware calls promoter on answer
)

@mw.track_with_handle
def retrieve(query: str) -> str:
    return my_index.search(query)[0]
```

After enough events + feedback meeting the thresholds, `cache.get(query)`
starts returning the auto-promoted answer transparently — your retrieval
function isn't called.

### Trigger semantics

All thresholds are AND. `None` means "ignore this dimension":

```python
# Pure popularity — 20 events, no quality gate.
AutoCachePromoter(cache, store, fb, min_count=20, min_acceptance=None)

# Default — 10 events AND 80%+ thumb-up rate.
AutoCachePromoter(cache, store, fb, min_count=10, min_acceptance=0.8)

# Rating-only UIs (no thumb data) — gate on score.
AutoCachePromoter(cache, store, fb, min_count=5, min_acceptance=None, min_score=0.8)
```

### Demotion

If a cluster's quality drops below the threshold (e.g. acceptance falls
below 80%), the auto-promoted entry is automatically removed from the
cache. Manually-set entries (`cache.set(...)`) are never auto-evicted —
only entries promoted by `AutoCachePromoter` are eligible for demotion.

### Privacy

When an `AutoCachePromoter` is wired in, response text is persisted in the
`clusters.latest_response` column (the candidate for promotion). Without a
promoter, that column stays NULL — no response text is stored. If your data
is sensitive, don't wire the promoter.

---

## Metrics

**`Metrics`** — similarity metrics, passed as `metric=` to `embedding_eval`:

| | Description |
|---|---|
| `Metrics.cosine_similarity` | Default. Angle between vectors — best for normalized embeddings |
| `Metrics.dot_product` | Raw dot product — fast, good for unit vectors |
| `Metrics.euclidean_similarity` | `1 / (1 + distance)` — closer vectors score higher |

**`RetrievalMetrics`** — post-run aggregations, passed to `SuiteReport.aggregate()`:

| | Description |
|---|---|
| `RetrievalMetrics.recall_at_k` | 1.0 if found, 0.0 if not |
| `RetrievalMetrics.mrr` | Mean Reciprocal Rank — `1/rank` |
| `RetrievalMetrics.ndcg` | Normalized Discounted Cumulative Gain (default `k=10`) |

```python
report = (
    EvalSuite()
    .add("cats", embedding_eval(...))
    .run()
    .aggregate(RetrievalMetrics.mrr,         name="avg_mrr")
    .aggregate(RetrievalMetrics.recall_at_k, name="avg_recall")
    .aggregate(RetrievalMetrics.ndcg,        name="avg_ndcg")
    .aggregate(lambda r: RetrievalMetrics.ndcg(r, k=3), name="ndcg@3")
)
```

---

## Embedder backends

All evals operate on pre-computed `list[float]` vectors — bring your own embedder.

| Modality | Example |
|---|---|
| Text (OpenAI) | `lambda t: client.embeddings.create(input=t, model="text-embedding-3-large").data[0].embedding` |
| Text (HuggingFace) | `lambda t: SentenceTransformer("all-MiniLM-L6-v2").encode(t).tolist()` |
| Audio (CLAP) | `lambda audio: clap_model.get_audio_embedding(audio)` |
| Image/Video (CLIP) | `lambda img: clip_model.encode_image(img).tolist()` |

---

## Project structure

```
src/raggit/
├── __init__.py
├── metrics.py           similarity + retrieval metrics
├── models.py            Pydantic data models
├── evaluation/
│   ├── suite.py         EvalSuite orchestrator
│   └── report.py        Rich terminal output
├── fns/
│   ├── chunk.py         chunk_eval factory
│   ├── embedding.py     embedding_eval factory
│   └── index.py         index_eval factory
└── middleware/
    ├── middleware.py     Middleware orchestrator (cache → monitor pipeline)
    ├── models.py         Cluster, Event models
    ├── cache/
    │   └── cache.py      SemanticCache
    ├── monitor/
    │   └── monitor.py    Monitor (clustering, validation, timing)
    └── stores/
        ├── base.py       MonitorStore, CacheStore ABCs
        └── sqlite.py     SQLiteMonitorStore, SQLiteClusterStore, SQLiteCacheStore
```

---

## Roadmap

- [x] `embedding_eval` — embedding model retrieval quality
- [x] `index_eval` — search function retrieval quality (Faiss, Chroma, BM25, ...)
- [x] `chunk_eval` — chunking strategy coverage, with configurable overlap
- [x] `EvalSuite` — orchestrate multiple evals, pass rate, Rich report
- [x] Custom metrics (`cosine_similarity`, `dot_product`, `euclidean_similarity`)
- [x] `RetrievalMetrics` — post-run aggregations (`recall_at_k`, `mrr`, `ndcg`)
- [x] Middleware — semantic cache + query monitor with pluggable stores
- [x] `monitor.popular_queries()` — surface popular query clusters from production
- [ ] Suite history & diff — persist `SuiteReport`s and diff across runs (e.g. model A vs model B over time)
- [x] Feedback integration — `track_with_handle` + `record_feedback` + acceptance/score per-cluster reads
- [x] Automatic cache promotion — `AutoCachePromoter` promotes & demotes based on count + acceptance + score thresholds
- [ ] CI/CD integration

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
