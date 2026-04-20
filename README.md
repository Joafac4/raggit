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

---

## Quickstart

```python
from sentence_transformers import SentenceTransformer
from raggit import Corpus, EvalSuite, Embedder, embedding_eval

model = SentenceTransformer("all-MiniLM-L6-v2")
embedder = Embedder("all-MiniLM-L6-v2", lambda t: model.encode(t).tolist())

corpus = Corpus(
    docs=[
        "To activate your account, click the link in the email.",
        "Your card expires on the date printed on the front.",
        "Visit the login page to reset your password.",
    ],
    embedder=embedder,
)

report = (
    EvalSuite(name="my_suite")
    .add("activation",  embedding_eval("How to activate my account?", "To activate your account...", corpus))
    .add("card expiry", embedding_eval("When does my card expire?",   "Your card expires on...",     corpus))
    .add("password",    embedding_eval("How do I reset my password?", "Visit the login page...",     corpus))
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
Tests an embedding model's retrieval quality. Embeds the corpus and query, ranks by similarity, checks if the expected document is in the top-k results.

```python
corpus = Corpus(docs=my_docs, embedder=embedder)

embedding_eval(
    query="How to activate my account?",
    expected="To activate your account...",
    corpus=corpus,
    k=3,                               # default
    metric=Metrics.cosine_similarity,  # default
)
```

### index_eval
Tests any search function — keyword, BM25, Faiss, Chroma, hybrid, or external APIs. No embeddings required.
`search_fn` receives a query string and must return a ranked `list[str]` of documents.

```python
index_eval(
    query="How to activate my account?",
    expected="To activate your account...",
    search_fn=my_search_fn,   # Callable[[str], List[str]]
    k=3,
)
```

**Faiss example:**
```python
import faiss
import numpy as np

index = faiss.IndexFlatL2(dim)
index.add(np.array(corpus_vecs, dtype="float32"))

def faiss_search(query: str) -> list[str]:
    vec = np.array([embedder.embed(query)], dtype="float32")
    _, indices = index.search(vec, k=10)
    return [corpus_docs[i] for i in indices[0]]

suite.add("faiss", index_eval("how to activate?", "To activate your account...", faiss_search, k=3))
```

**Chroma example:**
```python
import chromadb

client = chromadb.Client()
collection = client.create_collection("docs")
collection.add(documents=corpus_docs, ids=[str(i) for i in range(len(corpus_docs))])

def chroma_search(query: str) -> list[str]:
    results = collection.query(query_texts=[query], n_results=10)
    return results["documents"][0]

suite.add("chroma", index_eval("how to activate?", "To activate your account...", chroma_search, k=3))
```

**Comparing backends in the same suite:**
```python
report = (
    EvalSuite(name="backend_comparison")
    .add("faiss",  index_eval("how to activate?", expected_doc, faiss_search))
    .add("chroma", index_eval("how to activate?", expected_doc, chroma_search))
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

Built-in metrics available via `Metrics`:

| Metric | Description |
|---|---|
| `cosine_similarity` | Default. Angle between vectors — best for normalized embeddings |
| `dot_product` | Raw dot product — fast, good for unit vectors |
| `euclidean_similarity` | `1 / (1 + distance)` — closer vectors score higher |

Retrieval metrics:

| Metric | Description |
|---|---|
| `recall_at_k(hit)` | 1.0 if hit else 0.0 |
| `mrr(rank)` | Mean Reciprocal Rank — `1/rank`, 0.0 if not found |
| `ndcg(rank, k)` | Normalized Discounted Cumulative Gain |

---

## Supported embedder backends

| Backend | Example |
|---|---|
| OpenAI | `lambda t: client.embeddings.create(input=t, model="text-embedding-3-large").data[0].embedding` |
| HuggingFace | `lambda t: SentenceTransformer("all-MiniLM-L6-v2").encode(t).tolist()` |
| Cohere, Ollama, custom | Any `fn(str) -> list[float]` |

---

## Project structure

```
src/raggit/
├── __init__.py
├── embedder.py       model-agnostic embedding wrapper
├── metrics.py        similarity + retrieval metrics
├── models.py         Pydantic data models
├── suite.py          EvalSuite orchestrator
├── fns/
│   ├── embedding.py  embedding_eval factory
│   └── index.py      index_eval factory
└── evaluation/
    ├── corpus.py     Corpus (pre-computes embeddings once)
    └── report.py     Rich terminal output
```

---

## Roadmap

- [x] `embedding_eval` — embedding model retrieval quality
- [x] `index_eval` — search function retrieval quality (Faiss, Chroma, BM25, ...)
- [x] `EvalSuite` — orchestrate multiple evals, pass rate, Rich report
- [x] Custom metrics (`cosine_similarity`, `dot_product`, `euclidean_similarity`)
- [x] Retrieval metrics (`recall_at_k`, `mrr`, `ndcg`)
- [ ] Monitor — track eval results over time, detect regressions
- [ ] Human-in-the-loop mode
- [ ] CI/CD integration
