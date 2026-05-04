"""
basic_comparison.py — Raggit quickstart example

Compares two sentence-transformers models on a small set of Q&A pairs.
No API key required — models are downloaded from HuggingFace on first run.
"""

from sentence_transformers import SentenceTransformer

from raggit import EvalSuite, RetrievalMetrics, embedding_eval

# ── 1. Define your query / expected-doc pairs ───────────────────────────────

pairs = [
    ("How do I activate my account?",
     "To activate your account, click the link in the confirmation email."),
    ("When does my card expire?",
     "Your card expiry date is printed on the front of the card."),
    ("How do I reset my password?",
     "Visit the login page and click 'Forgot password' to reset it."),
]

corpus = [doc for _, doc in pairs]

# ── 2. Load two sentence-transformers models ─────────────────────────────────

print("Loading models...")
st_minilm = SentenceTransformer("all-MiniLM-L6-v2")
st_mpnet  = SentenceTransformer("all-mpnet-base-v2")

# ── 3. Build a suite per model and compare ──────────────────────────────────

def run_suite(name, encode):
    embed = lambda t: encode(t).tolist()
    corpus_vecs = [embed(doc) for doc in corpus]

    suite = EvalSuite(name=name)
    for query, expected in pairs:
        suite.add(query, embedding_eval(embed(query), embed(expected), corpus_vecs, k=1))

    (suite.run()
     .aggregate(RetrievalMetrics.mrr,         name="avg_mrr")
     .aggregate(RetrievalMetrics.recall_at_k, name="avg_recall")
     .show())


run_suite("all-MiniLM-L6-v2",  st_minilm.encode)
run_suite("all-mpnet-base-v2", st_mpnet.encode)
