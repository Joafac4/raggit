import math

import pytest

from raggit import Corpus, Embedder, EvalSuite, Metrics, chunk_eval, embedding_eval, index_eval


# ── Fixtures ──────────────────────────────────────────────────────────────────

VECS_GOOD = [
    [1.0, 0.0, 0.0],   # cats doc
    [0.0, 1.0, 0.0],   # dogs doc
    [0.0, 0.0, 1.0],   # birds doc
]

QUERY_CATS    = [0.99, 0.01, 0.0]
QUERY_DOGS    = [0.01, 0.99, 0.0]
EXPECTED_CATS = [1.0, 0.0, 0.0]
EXPECTED_DOGS = [0.0, 1.0, 0.0]

# "bad" corpus: all docs collapse to birds direction — expected cats/dogs not found
VECS_BAD = [
    [0.0, 0.0, 1.0],
    [0.0, 0.0, 1.0],
    [0.0, 0.0, 1.0],
]


# ── EvalSuite ─────────────────────────────────────────────────────────────────

def test_suite_all_pass():
    report = (
        EvalSuite(name="all_pass")
        .add("cats", embedding_eval(QUERY_CATS, EXPECTED_CATS, VECS_GOOD, k=1))
        .add("dogs", embedding_eval(QUERY_DOGS, EXPECTED_DOGS, VECS_GOOD, k=1))
        .run()
    )
    assert report.total == 2
    assert report.passed == 2
    assert report.failed == 0
    assert report.pass_rate == 1.0


def test_suite_all_fail():
    report = (
        EvalSuite(name="all_fail")
        .add("cats", embedding_eval(QUERY_CATS, EXPECTED_CATS, VECS_BAD, k=1))
        .add("dogs", embedding_eval(QUERY_DOGS, EXPECTED_DOGS, VECS_BAD, k=1))
        .run()
    )
    assert report.passed == 0
    assert report.failed == 2
    assert report.pass_rate == 0.0


def test_suite_add_returns_self():
    suite = EvalSuite()
    result = suite.add("cats", embedding_eval(QUERY_CATS, EXPECTED_CATS, VECS_GOOD))
    assert result is suite


def test_suite_empty():
    report = EvalSuite(name="empty").run()
    assert report.total == 0
    assert report.pass_rate == 0.0


# ── Corpus ────────────────────────────────────────────────────────────────────

def test_corpus_pre_computes_vecs():
    embedder = Embedder("identity", lambda v: v)
    corpus = Corpus(VECS_GOOD, embedder)
    assert len(corpus.vecs) == len(VECS_GOOD)


# ── embedding_eval ────────────────────────────────────────────────────────────

def test_embedding_eval_hit():
    result = embedding_eval(QUERY_CATS, EXPECTED_CATS, VECS_GOOD, k=1)()
    assert result.passed is True
    assert result.rank == 1


def test_embedding_eval_miss():
    result = embedding_eval(QUERY_CATS, EXPECTED_CATS, VECS_BAD, k=1)()
    assert result.passed is False


def test_embedding_eval_custom_metric():
    result = embedding_eval(QUERY_CATS, EXPECTED_CATS, VECS_GOOD, k=1, metric=Metrics.dot_product)()
    assert result.metric_name == "dot_product"


def test_embedding_eval_returns_callable():
    fn = embedding_eval(QUERY_CATS, EXPECTED_CATS, VECS_GOOD)
    assert callable(fn)


# ── index_eval ────────────────────────────────────────────────────────────────

def test_index_eval_hit():
    result = index_eval(
        query_vec=QUERY_CATS,
        expected_vec=EXPECTED_CATS,
        search_fn=lambda q: [EXPECTED_CATS, EXPECTED_DOGS],
        k=1,
    )()
    assert result.passed is True
    assert result.rank == 1
    assert result.score == 1.0


def test_index_eval_hit_within_k():
    result = index_eval(
        query_vec=QUERY_DOGS,
        expected_vec=EXPECTED_DOGS,
        search_fn=lambda q: [EXPECTED_CATS, EXPECTED_DOGS, [0.0, 0.0, 1.0]],
        k=2,
    )()
    assert result.passed is True
    assert result.rank == 2


def test_index_eval_miss():
    result = index_eval(
        query_vec=QUERY_CATS,
        expected_vec=[0.0, 0.0, 1.0],
        search_fn=lambda q: [EXPECTED_CATS, EXPECTED_DOGS],
        k=2,
    )()
    assert result.passed is False
    assert result.rank is None
    assert result.score == 0.0


def test_index_eval_returns_callable():
    fn = index_eval(QUERY_CATS, EXPECTED_CATS, search_fn=lambda q: [])
    assert callable(fn)


def test_index_eval_threshold():
    # vector close but below default threshold should miss
    near_but_not_match = [0.9, 0.1, 0.0]  # cos_sim with EXPECTED_CATS < 0.999
    result = index_eval(
        query_vec=QUERY_CATS,
        expected_vec=EXPECTED_CATS,
        search_fn=lambda q: [near_but_not_match],
        k=1,
    )()
    assert result.passed is False


# ── custom eval_fn ────────────────────────────────────────────────────────────

def test_custom_eval_fn():
    from raggit import EvalSingleResult

    def my_eval() -> EvalSingleResult:
        return EvalSingleResult(passed=True, score=0.99, metric_name="custom")

    report = EvalSuite().add("custom", my_eval).run()
    assert report.passed == 1
    assert report.results[0].result.metric_name == "custom"


# ── chunk_eval ────────────────────────────────────────────────────────────────

DOCUMENT = [[float(i), 0.0, 0.0] for i in range(10)]  # 10 "frames" as vecs

def fixed_chunker(doc, overlap):
    size = 3
    step = max(1, int(size * (1 - overlap)))
    return [doc[i:i + size] for i in range(0, len(doc), step)]

def embed_fn(chunk):
    # average of chunk vecs
    n = len(chunk)
    return [sum(v[i] for v in chunk) / n for i in range(3)] if n else [0.0, 0.0, 0.0]

EXPECTED_VEC = [4.0, 0.0, 0.0]   # close to middle of document


def test_chunk_eval_hit():
    result = chunk_eval(DOCUMENT, EXPECTED_VEC, fixed_chunker, embed_fn, overlap=0.0, threshold=0.9)()
    assert result.passed is True
    assert result.rank is not None
    assert result.metric_name == "chunk_coverage"


def test_chunk_eval_miss():
    result = chunk_eval(DOCUMENT, [0.0, 1.0, 0.0], fixed_chunker, embed_fn, overlap=0.0, threshold=0.9)()
    assert result.passed is False
    assert result.rank is None


def test_chunk_eval_returns_callable():
    fn = chunk_eval(DOCUMENT, EXPECTED_VEC, fixed_chunker, embed_fn)
    assert callable(fn)


def test_chunk_eval_default_overlap():
    received = []
    def recording_chunker(doc, overlap):
        received.append(overlap)
        return [doc]
    chunk_eval(DOCUMENT, EXPECTED_VEC, recording_chunker, embed_fn)()
    assert received == [0.0]


# ── Retrieval metrics ─────────────────────────────────────────────────────────

def test_recall_at_k():
    assert Metrics.recall_at_k(True) == 1.0
    assert Metrics.recall_at_k(False) == 0.0


def test_mrr():
    assert Metrics.mrr(1) == 1.0
    assert abs(Metrics.mrr(2) - 0.5) < 1e-9
    assert Metrics.mrr(None) == 0.0


def test_ndcg():
    assert abs(Metrics.ndcg(1, k=3) - 1.0) < 1e-9
    assert abs(Metrics.ndcg(2, k=3) - 1 / math.log2(3)) < 1e-9
    assert Metrics.ndcg(4, k=3) == 0.0
    assert Metrics.ndcg(None, k=3) == 0.0
