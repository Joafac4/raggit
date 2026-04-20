import math

import pytest

from raggit import Corpus, EvalSuite, Embedder, Metrics, embedding_eval, index_eval


# ── Helpers ──────────────────────────────────────────────────────────────────

def make_embedder(name: str, vecs: dict) -> Embedder:
    return Embedder(name, embed_fn=lambda text: vecs[text])


CORPUS_DOCS = ["doc about cats", "doc about dogs", "doc about birds"]

VECS_GOOD = {
    "cats query":      [1.0, 0.0, 0.0],
    "doc about cats":  [0.99, 0.01, 0.0],
    "dogs query":      [0.0, 1.0, 0.0],
    "doc about dogs":  [0.01, 0.99, 0.0],
    "doc about birds": [0.0, 0.0, 1.0],
}

VECS_BAD = {
    "cats query":      [1.0, 0.0, 0.0],
    "doc about cats":  [0.0, 1.0, 0.0],
    "dogs query":      [0.0, 1.0, 0.0],
    "doc about dogs":  [1.0, 0.0, 0.0],
    "doc about birds": [0.0, 0.0, 1.0],
}


# ── EvalSuite ─────────────────────────────────────────────────────────────────

def test_suite_all_pass():
    corpus = Corpus(CORPUS_DOCS, make_embedder("good-model", VECS_GOOD))
    report = (
        EvalSuite(name="all_pass")
        .add("cats", embedding_eval("cats query", "doc about cats", corpus, k=1))
        .add("dogs", embedding_eval("dogs query", "doc about dogs", corpus, k=1))
        .run()
    )
    assert report.total == 2
    assert report.passed == 2
    assert report.failed == 0
    assert report.pass_rate == 1.0


def test_suite_all_fail():
    corpus = Corpus(CORPUS_DOCS, make_embedder("bad-model", VECS_BAD))
    report = (
        EvalSuite(name="all_fail")
        .add("cats", embedding_eval("cats query", "doc about cats", corpus, k=1))
        .add("dogs", embedding_eval("dogs query", "doc about dogs", corpus, k=1))
        .run()
    )
    assert report.passed == 0
    assert report.failed == 2
    assert report.pass_rate == 0.0


def test_suite_add_returns_self():
    corpus = Corpus(CORPUS_DOCS, make_embedder("good-model", VECS_GOOD))
    suite = EvalSuite()
    result = suite.add("cats", embedding_eval("cats query", "doc about cats", corpus))
    assert result is suite


def test_suite_empty():
    report = EvalSuite(name="empty").run()
    assert report.total == 0
    assert report.pass_rate == 0.0


# ── Corpus ────────────────────────────────────────────────────────────────────

def test_corpus_pre_computes_vecs():
    corpus = Corpus(CORPUS_DOCS, make_embedder("good-model", VECS_GOOD))
    assert len(corpus.vecs) == len(CORPUS_DOCS)


# ── embedding_eval ────────────────────────────────────────────────────────────

def test_embedding_eval_hit():
    corpus = Corpus(CORPUS_DOCS, make_embedder("good-model", VECS_GOOD))
    result = embedding_eval("cats query", "doc about cats", corpus, k=1)()
    assert result.passed is True
    assert result.rank == 1


def test_embedding_eval_miss():
    corpus = Corpus(CORPUS_DOCS, make_embedder("bad-model", VECS_BAD))
    result = embedding_eval("cats query", "doc about cats", corpus, k=1)()
    assert result.passed is False


def test_embedding_eval_custom_metric():
    corpus = Corpus(CORPUS_DOCS, make_embedder("good-model", VECS_GOOD))
    result = embedding_eval("cats query", "doc about cats", corpus, k=1, metric=Metrics.dot_product)()
    assert result.metric_name == "dot_product"


def test_embedding_eval_returns_callable():
    corpus = Corpus(CORPUS_DOCS, make_embedder("good-model", VECS_GOOD))
    fn = embedding_eval("cats query", "doc about cats", corpus)
    assert callable(fn)


# ── index_eval ────────────────────────────────────────────────────────────────

def test_index_eval_hit():
    result = index_eval(
        query="cats",
        expected="doc about cats",
        search_fn=lambda q: ["doc about cats", "doc about dogs"],
        k=1,
    )()
    assert result.passed is True
    assert result.rank == 1
    assert result.score == 1.0


def test_index_eval_hit_within_k():
    result = index_eval(
        query="dogs",
        expected="doc about dogs",
        search_fn=lambda q: ["doc about cats", "doc about dogs", "doc about birds"],
        k=2,
    )()
    assert result.passed is True
    assert result.rank == 2


def test_index_eval_miss():
    result = index_eval(
        query="cats",
        expected="doc about whales",
        search_fn=lambda q: ["doc about cats", "doc about dogs"],
        k=2,
    )()
    assert result.passed is False
    assert result.rank is None
    assert result.score == 0.0


def test_index_eval_returns_callable():
    fn = index_eval("q", "doc", search_fn=lambda q: [])
    assert callable(fn)


# ── custom eval_fn ────────────────────────────────────────────────────────────

def test_custom_eval_fn():
    from raggit import EvalSingleResult

    def my_eval() -> EvalSingleResult:
        return EvalSingleResult(passed=True, score=0.99, metric_name="custom")

    report = EvalSuite().add("custom", my_eval).run()
    assert report.passed == 1
    assert report.results[0].result.metric_name == "custom"


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
