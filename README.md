# Raggit

> You updated your embedding model and your RAG got worse — but you didn't know until users complained.
>
> Raggit lets you monitor retrieval quality in production, automatically surface the queries where your pipeline is failing, and run evals against them — so you always know if a model change made things better or worse on your actual data.
>
> It's not a benchmark. It's version control for your RAG pipeline.

---

## The loop

Most RAG failures are invisible. The pipeline runs, something gets retrieved, an answer is generated — but nobody checks if the right document was at rank 1.

Raggit closes the feedback loop:

1. **Monitor** production queries — cluster similar ones, track retrieval rank and score per cluster
2. **Identify** clusters where retrieval is consistently poor (`avg_rank > 5`, `avg_score < 0.7`)
3. **Generate evals** from those clusters automatically with `EvalSuite.from_monitor()`
4. **Run evals** when you change models — see exactly which queries improved or regressed

```python
from raggit.middleware import Middleware, Monitor, SQLiteMonitorStore
from raggit import EvalSuite

embed = lambda t: model.encode(t).tolist()

# 1. Wrap your retrieval function — monitor logs every query in the background
monitor = Monitor(store=SQLiteMonitorStore(".raggit/monitor.db"), embedder=embed)
middleware = Middleware(monitor=monitor, embedder=embed)

@middleware.track
def retrieve(query: str) -> str:
    docs, scores, ids = index.search(query)
    return docs[0], _monitor_kwargs={
        "retrieval_rank": 1,
        "retrieval_score": scores[0],
        "retrieved_doc_ids": ids,
    }

# ... production traffic flows through retrieve() ...

# 2. After collecting data, generate evals from clusters with poor retrieval
corpus = {"doc1": "Reset your password via Settings.", "doc2": "Contact support at ..."}

suite = EvalSuite.from_monitor(
    monitor=monitor,
    embedder=embed,
    corpus=corpus,
    use_problematic=True,   # only clusters with avg_rank > 5 or avg_score < 0.7
)

# 3. Switch to a new model, run the same suite, compare
report = suite.run()
report.show()
```

```
─────────────────────────── Raggit Eval Suite ───────────────────────────
  Suite : from_monitor
  Date  : 2026-04-30 11:42

  Eval                                        Passed   Rank   Score
 ────────────────────────────────────────────────────────────────────
  how do i reset my password                    ✓        1     0.94
  i cant log into my account                    ✗        4     0.61
  what is the refund policy                     ✓        2     0.88

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

Write evals manually when you know what to test. Use `from_monitor()` when you want production data to tell you what to test.

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

The monitor wraps your retrieval function, clusters similar queries, and tracks retrieval quality per cluster. It's the data source for `from_monitor()`.

```python
from raggit.middleware import Middleware, Monitor, SQLiteMonitorStore

monitor = Monitor(
    store=SQLiteMonitorStore(".raggit/monitor.db"),
    embedder=embed,
    cluster_threshold=0.92,
)
middleware = Middleware(monitor=monitor, embedder=embed)

@middleware.track
def retrieve(query: str) -> str:
    docs, scores, ids = index.search(query)
    return docs[0]

# Pass retrieval data so monitor can track quality per cluster
result = retrieve("my query", _monitor_kwargs={
    "retrieval_rank": 1,
    "retrieval_score": 0.91,
    "retrieved_doc_ids": ["doc_42"],
})

# Inspect clusters
monitor.stats()
monitor.clusters(top=10)
monitor.problematic_clusters(min_rank=5, max_score=0.7)
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

    def log(self, query, vec, latency_ms, threshold, cache_hit=False, **kwargs):
        ...  # write to DynamoDB
```

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
│   ├── suite.py         EvalSuite orchestrator + from_monitor()
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
- [x] `EvalSuite.from_monitor()` — generate evals from production monitoring data
- [ ] Suite aggregator — compare pass rates across multiple suites (e.g. model A vs model B)
- [ ] Human-in-the-loop eval approval
- [ ] CI/CD integration
