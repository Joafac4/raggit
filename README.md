# Raggit

> You updated your embedding model and your RAG got worse — but you didn't know until users complained.
>
> Raggit lets you evaluate and compare embedding models on your own data, track results over time, and always know which model works best for your specific domain.
>
> It's not a benchmark. It's version control for your RAG pipeline.

---

## The Problem

When you use a RAG system in production, you embed documents and queries with a specific model. When that model is updated (silently by the provider) or you want to migrate to a new one, you have no systematic way to know if retrieval quality improved or degraded **for your use case**.

Traditional benchmarks like MTEB evaluate models on generic datasets. Raggit evaluates models on **your data** with **your queries**.

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

## Quickstart

```python
from sentence_transformers import SentenceTransformer
from raggit import EvalSuite, embedding_eval

model = SentenceTransformer("all-MiniLM-L6-v2")
embed = lambda t: model.encode(t).tolist()

docs = [
    "To activate your account, click the link in the email.",
    "Your card expires on the date printed on the front.",
    "Visit the login page to reset your password.",
]
corpus_vecs = [embed(doc) for doc in docs]

report = (
    EvalSuite(name="my_suite")
    .add("activation",  embedding_eval(embed("How to activate my account?"), embed(docs[0]), corpus_vecs))
    .add("card expiry", embedding_eval(embed("When does my card expire?"),   embed(docs[1]), corpus_vecs))
    .add("password",    embedding_eval(embed("How do I reset my password?"), embed(docs[2]), corpus_vecs))
    .run()
)

report.show()
```

Output:
```
─────────────────────────── Raggit Eval Suite ───────────────────────────
  Suite : my_suite
  Date  : 2026-04-20 10:30

  Eval                              Passed   Rank   Score
 ──────────────────────────────────────────────────────
  activation                          ✓        1     0.91
  card expiry                         ✓        1     0.87
  password                            ✓        2     0.83

  Total: 3  |  Passed: 3  |  Failed: 0  |  Pass rate: 100.0%
─────────────────────────────────────────────────────────────────────────
```

---

## How it works

1. **Define your corpus** — the documents your retrieval system should search over.
2. **Wrap your embedder** — pass any callable that converts text to `list[float]`.
3. **Create evals** — `embedding_eval` or `index_eval` returns a thunk for each query/expected-doc pair.
4. **Run a suite** — `EvalSuite` orchestrates all evals and reports pass rate, rank, and score.

---

## Eval types

### embedding_eval
Tests an embedding model's retrieval quality. All inputs are pre-computed vectors — works with any modality (text, audio, image, video, etc.).

```python
corpus_vecs = [embed(doc) for doc in docs]

embedding_eval(
    query_vec=embed("How to activate my account?"),
    expected_vec=embed("To activate your account..."),
    corpus_vecs=corpus_vecs,
    k=3,                               # default
    metric=Metrics.cosine_similarity,  # default
)
```

### chunk_eval
Tests a chunking strategy. Checks whether expected content survives chunking by comparing vectors.
Works with any modality — text, audio, video, image.

`chunk_fn` receives `(document, overlap)` and returns chunks of any type.
`embed_fn` converts each chunk to a vector for similarity comparison.

```python
def my_chunker(text: str, overlap: float) -> list[str]:
    size = 512
    step = max(1, int(size * (1 - overlap)))
    return [text[i:i + size] for i in range(0, len(text), step)]

report = (
    EvalSuite(name="chunking_comparison")
    .add("overlap=0.0",  chunk_eval(document, expected_vec, my_chunker, embed_fn, overlap=0.0))
    .add("overlap=0.25", chunk_eval(document, expected_vec, my_chunker, embed_fn, overlap=0.25))
    .add("overlap=0.5",  chunk_eval(document, expected_vec, my_chunker, embed_fn, overlap=0.5))
    .run()
)
```

- `passed`: True if any chunk has cosine similarity ≥ `threshold` (default `0.9`) with `expected_vec`
- `rank`: index of the first matching chunk (1-indexed)
- `score`: highest cosine similarity across all chunks
- `metric_name`: `"chunk_coverage"`

### index_eval
Tests any search index. `search_fn` receives the query vector and returns a ranked list of vectors.
Compatible with Faiss, Chroma, BM25, or any other backend.

```python
index_eval(
    query_vec=embed("How to activate my account?"),
    expected_vec=embed("To activate your account..."),
    search_fn=my_search_fn,   # Callable[[List[float]], List[List[float]]]
    k=3,
    threshold=0.999,          # cosine similarity to consider a match
)
```

**Faiss example:**
```python
import faiss
import numpy as np

index = faiss.IndexFlatL2(dim)
index.add(np.array(corpus_vecs, dtype="float32"))

def faiss_search(query_vec: list[float]) -> list[list[float]]:
    vec = np.array([query_vec], dtype="float32")
    _, indices = index.search(vec, k=10)
    return [corpus_vecs[i] for i in indices[0]]

suite.add("faiss", index_eval(query_vec, expected_vec, faiss_search, k=3))
```

**Chroma example:**
```python
import chromadb

client = chromadb.Client()
collection = client.create_collection("docs")
collection.add(embeddings=corpus_vecs, ids=[str(i) for i in range(len(corpus_vecs))])

def chroma_search(query_vec: list[float]) -> list[list[float]]:
    results = collection.query(query_embeddings=[query_vec], n_results=10)
    return results["embeddings"][0]

suite.add("chroma", index_eval(query_vec, expected_vec, chroma_search, k=3))
```

**Comparing backends in the same suite:**
```python
report = (
    EvalSuite(name="backend_comparison")
    .add("faiss",  index_eval(query_vec, expected_vec, faiss_search))
    .add("chroma", index_eval(query_vec, expected_vec, chroma_search))
    .run()
)
```

### Custom evals
Any `Callable[[], EvalSingleResult]` works as an eval.

```python
from raggit import EvalSuite, EvalSingleResult

def my_eval() -> EvalSingleResult:
    score = run_my_custom_check()
    return EvalSingleResult(passed=score > 0.8, score=score, metric_name="custom")

report = EvalSuite().add("custom", my_eval).run()
```

---

## Metrics

**`Metrics`** — similarity metrics, passed as `metric=` to `embedding_eval` for ranking:

| | Description |
|---|---|
| `Metrics.cosine_similarity` | Default. Angle between vectors — best for normalized embeddings |
| `Metrics.dot_product` | Raw dot product — fast, good for unit vectors |
| `Metrics.euclidean_similarity` | `1 / (1 + distance)` — closer vectors score higher |

**`RetrievalMetrics`** — post-run aggregation, passed to `SuiteReport.aggregate()`:

| | Description |
|---|---|
| `RetrievalMetrics.recall_at_k` | 1.0 if found, 0.0 if not |
| `RetrievalMetrics.mrr` | Mean Reciprocal Rank — `1/rank` |
| `RetrievalMetrics.ndcg` | Normalized Discounted Cumulative Gain (default `k=10`) |

Aggregations are computed on the report after running — optional and chainable:

```python
report = (
    EvalSuite()
    .add("cats", embedding_eval(..., metric=Metrics.cosine_similarity))
    .run()
    .aggregate(RetrievalMetrics.mrr,         name="avg_mrr")
    .aggregate(RetrievalMetrics.recall_at_k, name="avg_recall")
    .aggregate(RetrievalMetrics.ndcg,        name="avg_ndcg")           # default k=10
    .aggregate(lambda r: RetrievalMetrics.ndcg(r, k=3), name="ndcg@3") # custom k
)
```

Aggregations appear in `report.show()` and `report.aggregations`.

---

## Embedder backends

All evals operate on pre-computed `list[float]` vectors — bring your own embedder for any modality.

| Modality | Example |
|---|---|
| Text (OpenAI) | `lambda t: client.embeddings.create(input=t, model="text-embedding-3-large").data[0].embedding` |
| Text (HuggingFace) | `lambda t: SentenceTransformer("all-MiniLM-L6-v2").encode(t).tolist()` |
| Audio (CLAP) | `lambda audio: clap_model.get_audio_embedding(audio)` |
| Image/Video (CLIP) | `lambda img: clip_model.encode_image(img).tolist()` |
| Any custom | Any `fn(input) -> list[float]` |

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
└── fns/
    ├── chunk.py         chunk_eval factory
    ├── embedding.py     embedding_eval factory
    └── index.py         index_eval factory
```

---

## Roadmap

- [x] `embedding_eval` — embedding model retrieval quality
- [x] `index_eval` — search function retrieval quality (Faiss, Chroma, BM25, ...)
- [x] `chunk_eval` — chunking strategy coverage, with configurable overlap
- [x] `EvalSuite` — orchestrate multiple evals, pass rate, Rich report
- [x] Custom metrics (`cosine_similarity`, `dot_product`, `euclidean_similarity`)
- [x] `RetrievalMetrics` — post-run aggregations (`recall_at_k`, `mrr`, `ndcg`)
- [ ] Suite aggregator — compare pass rates across multiple suites (e.g. model A vs model B)
- [ ] Monitor — track eval results over time, detect regressions
- [ ] Human-in-the-loop mode
- [ ] CI/CD integration
