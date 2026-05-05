"""
monitor_to_evals.py — Raggit full loop example

Production loop end-to-end:
1. Wrap retrieval with Middleware → Monitor logs every query, clusters similar ones
2. Simulate production traffic
3. Inspect popular clusters via monitor.popular_queries()
4. Write evals manually against the queries you care about
   (the monitor tells you WHAT to cover; you decide WHAT'S CORRECT)
5. Swap embedding models, re-run the same evals, compare reports
"""

from sentence_transformers import SentenceTransformer

from raggit import EvalSuite, Metrics, RetrievalMetrics, embedding_eval
from raggit.middleware import Middleware, Monitor, SQLiteMonitorStore


# ── Tiny corpus ─────────────────────────────────────────────────────────────

corpus = [
    "To activate your account, click the link in the confirmation email.",
    "Your card expiry date is printed on the front of the card.",
    "Visit the login page and click 'Forgot password' to reset it.",
    "Refunds are issued within 5 business days to your original payment method.",
    "Two-factor authentication can be enabled in Settings → Security.",
]

# ── Pretend production traffic ─────────────────────────────────────────────

prod_queries = [
    "how do I reset my password",
    "reset password please",
    "forgot my password help",
    "how do I activate my account",
    "activate account",
    "where is my refund",
    "card expiry",
    "enable 2fa",
]


# ── 1-2. Wrap retrieval with monitor, simulate traffic ─────────────────────

print("Loading model A: all-MiniLM-L6-v2")
model_a = SentenceTransformer("all-MiniLM-L6-v2")
embed_a = lambda t: model_a.encode(t).tolist()
corpus_vecs_a = [embed_a(d) for d in corpus]


def cosine_top1(query: str, embed, corpus_vecs) -> str:
    q = embed(query)
    scores = [Metrics.cosine_similarity(q, cv) for cv in corpus_vecs]
    return corpus[scores.index(max(scores))]


monitor = Monitor(
    store=SQLiteMonitorStore(".raggit/example_monitor.db"),
    embedder=embed_a,
    cluster_threshold=0.85,
)
mw = Middleware(monitor=monitor, embedder=embed_a)


@mw.track
def retrieve(query: str) -> str:
    return cosine_top1(query, embed_a, corpus_vecs_a)


print("Simulating production traffic...")
for q in prod_queries:
    retrieve(q)
mw.shutdown()  # flush async monitor logs


# ── 3. Inspect what production traffic actually looks like ─────────────────

print("\nPopular query clusters:")
for cluster in monitor.popular_queries(top=5):
    print(f"  {cluster.count}x  {cluster.representative_query!r}")


# ── 4. Write evals manually against the queries you care about ────────────
# Picked by hand from popular_queries above. The monitor surfaces the queries;
# you decide the correct answer for each one.

eval_pairs = [
    ("reset password",
     "how do I reset my password",
     "Visit the login page and click 'Forgot password' to reset it."),
    ("activate account",
     "how do I activate my account",
     "To activate your account, click the link in the confirmation email."),
    ("refund",
     "where is my refund",
     "Refunds are issued within 5 business days to your original payment method."),
]


def run_suite(name, embed):
    corpus_vecs = [embed(d) for d in corpus]
    suite = EvalSuite(name=name)
    for label, query, expected_doc in eval_pairs:
        suite.add(label, embedding_eval(embed(query), embed(expected_doc), corpus_vecs, k=1))
    return (
        suite.run()
        .aggregate(RetrievalMetrics.mrr,         name="avg_mrr")
        .aggregate(RetrievalMetrics.recall_at_k, name="avg_recall")
    )


# ── 5. Run on model A, swap to model B, re-run the same evals ──────────────

print("\n── Model A: all-MiniLM-L6-v2 ──")
run_suite("all-MiniLM-L6-v2", embed_a).show()

print("\nLoading model B: all-mpnet-base-v2")
model_b = SentenceTransformer("all-mpnet-base-v2")
embed_b = lambda t: model_b.encode(t).tolist()

print("\n── Model B: all-mpnet-base-v2 ──")
run_suite("all-mpnet-base-v2", embed_b).show()
