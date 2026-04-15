# Raggit

> You updated your embedding model and your RAG got worse — but you didn't know until users complained.
>
> Raggit lets you compare embedding models on your own data, track results over time, and always know which model works best for your specific domain.
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

## Quickstart (10 lines)

```python
from raggit import RaggitEval, Embedder, EmbeddingPair
from raggit.store import RaggitStore
from sentence_transformers import SentenceTransformer


pairs = [
    EmbeddingPair(query="How to activate my account?", relevant_doc="To activate..."),
    EmbeddingPair(query="Card expiry date", relevant_doc="Your card expires..."),
]

model_a = SentenceTransformer("all-MiniLM-L6-v2")
model_b = SentenceTransformer("all-mpnet-base-v2")

embedder_a = Embedder("all-MiniLM-L6-v2", lambda t: model_a.encode(t).tolist())
embedder_b = Embedder("all-mpnet-base-v2", lambda t: model_b.encode(t).tolist())

run = RaggitEval(pairs=pairs).compare(model_a, model_b)
run.report.show()

RaggitStore().save_run(run)
```

---

## How it works

1. **Define pairs** — each pair is a query and the document that *should* be retrieved.
2. **Wrap your models** — pass any callable that converts text to a list of floats.
3. **Compare** — Raggit embeds every query and document with both models and computes cosine similarity.
4. **Report** — a Rich table shows scores side by side and declares a winner.
5. **Persist** — results are saved as JSON under `.raggit/` so you can track changes over time.

---

## Supported embedder backends

| Backend | Install extra | Example |
|---|---|---|
| OpenAI | `raggit[openai]` | `text-embedding-3-large` |
| HuggingFace | `raggit[huggingface]` | `all-MiniLM-L6-v2` |
| Cohere, Ollama, custom | _(none)_ | Any `fn(str) -> list[float]` |

---

## Project structure

```
src/raggit/
├── __init__.py      public API
├── models.py        Pydantic data models
├── embedder.py      model-agnostic wrapper
├── evaluator.py     comparison logic + cosine similarity
├── store.py         local JSON persistence
└── report.py        Rich terminal output
```

---

## Roadmap

- [x] Auto mode (cosine similarity)
- [ ] Human-in-the-loop mode
- [ ] Drift detection over time
- [ ] `get_best_model()` across runs
- [ ] CI/CD integration
