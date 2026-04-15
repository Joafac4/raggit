"""
basic_comparison.py — Raggit quickstart example

Compares two sentence-transformers models on a small set of Q&A pairs.
No API key required — models are downloaded from HuggingFace on first run.
"""

from sentence_transformers import SentenceTransformer

from raggit import Embedder, EmbeddingPair, RaggitEval
from raggit.store import RaggitStore

# ── 1. Define your query/document pairs ─────────────────────────────────────

pairs = [
    EmbeddingPair(
        query="How do I activate my account?",
        relevant_doc="To activate your account, click the link in the confirmation email.",
    ),
    EmbeddingPair(
        query="When does my card expire?",
        relevant_doc="Your card expiry date is printed on the front of the card.",
    ),
    EmbeddingPair(
        query="How do I reset my password?",
        relevant_doc="Visit the login page and click 'Forgot password' to reset it.",
    ),
]

# ── 2. Load two sentence-transformers models ─────────────────────────────────

print("Loading models...")
st_minilm = SentenceTransformer("all-MiniLM-L6-v2")
st_mpnet  = SentenceTransformer("all-mpnet-base-v2")

model_a = Embedder("all-MiniLM-L6-v2",   embed_fn=lambda t: st_minilm.encode(t).tolist())
model_b = Embedder("all-mpnet-base-v2",   embed_fn=lambda t: st_mpnet.encode(t).tolist())

# ── 3. Run the evaluation ────────────────────────────────────────────────────

run = RaggitEval(pairs=pairs).compare(model_a, model_b, mode="auto")

# ── 4. Show the report ───────────────────────────────────────────────────────

run.report.show()

# ── 5. Persist the result ────────────────────────────────────────────────────

store = RaggitStore()
store.save_run(run)
print(f"\nRun saved: {run.run_id}")

# ── 6. Inspect history ───────────────────────────────────────────────────────

all_runs = store.list_runs()
print(f"Total runs stored: {len(all_runs)}")
