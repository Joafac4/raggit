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

# With OpenAI support
pip install raggit[openai]

# With HuggingFace support
pip install raggit[huggingface]
```

**uv**
```bash
uv add raggit

# With OpenAI support
uv add raggit[openai]

# With HuggingFace support
uv add raggit[huggingface]
```

---

## Quickstart

```python
from sentence_transformers import SentenceTransformer
from raggit import EvalSuite, EmbeddingEval, Embedder
from raggit.store import RaggitStore

model = SentenceTransformer("all-MiniLM-L6-v2")
embedder = Embedder("all-MiniLM-L6-v2", lambda t: model.encode(t).tolist())

corpus = [
    "To activate your account, click the link in the email.",
    "Your card expires on the date printed on the front.",
    "Visit the login page to reset your password.",
]

report = (
    EvalSuite(name="my_suite")
    .add(EmbeddingEval("How to activate my account?", corpus[0], corpus, embedder))
    .add(EmbeddingEval("When does my card expire?",   corpus[1], corpus, embedder))
    .add(EmbeddingEval("How do I reset my password?", corpus[2], corpus, embedder))
    .run()
)

report.show()
RaggitStore().save(report)
```

Output:
```
─────────────────────────── Raggit Eval Suite ───────────────────────────
  Suite : my_suite
  Date  : 2026-04-18 10:30

  Eval                              Hit    Rank   Score
 ──────────────────────────────────────────────────────
  all-MiniLM-L6-v2: How to activ…   ✓      1      0.91
  all-MiniLM-L6-v2: When does my…   ✓      1      0.87
  all-MiniLM-L6-v2: How do I res…   ✓      2      0.83

  Total: 3  |  Passed: 3  |  Failed: 0  |  Pass rate: 100.0%
─────────────────────────────────────────────────────────────────────────
```

---

## How it works

1. **Define your corpus** — the documents your retrieval system should search over.
2. **Wrap your embedder** — pass any callable that converts text to `list[float]`.
3. **Create evals** — each `EmbeddingEval` or `SearchEval` tests one query/expected-doc pair.
4. **Run a suite** — `EvalSuite` orchestrates all evals and reports hit rate, rank, and score.
5. **Persist** — `RaggitStore` saves results as JSON under `.raggit/` for tracking over time.

---

## Eval types

### EmbeddingEval
Tests an embedding model's retrieval quality. Embeds the corpus and query, ranks by similarity, checks if the expected document is in the top-k results.

```python
EmbeddingEval(
    query="How to activate my account?",
    expected_doc="To activate your account...",
    corpus=my_corpus,
    embedder=embedder,
    k=3,                           # default
    metric=Metrics.cosine_similarity,  # default
)
```

### SearchEval
Tests any search function — keyword, BM25, hybrid, or external APIs. No embeddings required.

```python
SearchEval(
    query="How to activate my account?",
    expected_doc="To activate your account...",
    search_fn=my_search_fn,   # Callable[[str], List[str]]
    k=3,
)
```

### Custom evals
Extend `BaseEval` to test anything that returns an `EvalSingleResult`.

```python
from raggit import BaseEval, EvalSingleResult

class MyEval(BaseEval):
    name = "my custom eval"

    def run(self) -> EvalSingleResult:
        ...
```

---

## Metrics

Built-in metrics available via `Metrics`:

| Metric | Description |
|---|---|
| `cosine_similarity` | Default. Angle between vectors — best for normalized embeddings |
| `dot_product` | Raw dot product — fast, good for unit vectors |
| `euclidean_similarity` | `1 / (1 + distance)` — closer vectors score higher |

Register a custom metric:

```python
from raggit import Metrics

m = Metrics()
m.register_metric("my_metric", lambda a, b: ...)
```

---

## Supported embedder backends

| Backend | Install extra | Example model |
|---|---|---|
| OpenAI | `raggit[openai]` | `text-embedding-3-large` |
| HuggingFace | `raggit[huggingface]` | `all-MiniLM-L6-v2` |
| Cohere, Ollama, custom | _(none)_ | Any `fn(str) -> list[float]` |

---

## Project structure

```
src/raggit/
├── __init__.py
├── embedder.py          model-agnostic embedding wrapper
├── metrics.py           similarity metrics + custom registry
├── models.py            Pydantic data models
├── store.py             local JSON persistence
├── evaluation/
│   ├── base.py          BaseEval abstract class
│   ├── engine.py        core vector ranking + retrieval metrics
│   ├── suite.py         EvalSuite orchestrator
│   ├── report.py        Rich terminal output
│   └── evals/
│       ├── embedding.py EmbeddingEval
│       └── search.py    SearchEval
└── monitor/             coming soon
```

---

## Roadmap

- [x] `EmbeddingEval` — embedding model retrieval quality
- [x] `SearchEval` — search function retrieval quality
- [x] `EvalSuite` — orchestrate multiple evals, pass rate, Rich report
- [x] Custom metrics registry
- [x] Local persistence with `RaggitStore`
- [ ] Monitor — track eval results over time, detect regressions
- [ ] Human-in-the-loop mode
- [ ] CI/CD integration
