import math

import pytest

from raggit import EvalSuite, EmbeddingEval, SearchEval, Embedder, Metrics
from raggit.evaluation.engine import Evaluation


# ── Helpers ──────────────────────────────────────────────────────────────────

def make_embedder(name: str, vecs: dict) -> Embedder:
    return Embedder(name, embed_fn=lambda text: vecs[text])


CORPUS = ["doc about cats", "doc about dogs", "doc about birds"]

# tight clusters — expected doc always lands at rank 1
VECS_GOOD = {
    "cats query":      [1.0, 0.0, 0.0],
    "doc about cats":  [0.99, 0.01, 0.0],
    "dogs query":      [0.0, 1.0, 0.0],
    "doc about dogs":  [0.01, 0.99, 0.0],
    "doc about birds": [0.0, 0.0, 1.0],
}

# orthogonal — expected doc never ranks first
VECS_BAD = {
    "cats query":      [1.0, 0.0, 0.0],
    "doc about cats":  [0.0, 1.0, 0.0],
    "dogs query":      [0.0, 1.0, 0.0],
    "doc about dogs":  [1.0, 0.0, 0.0],
    "doc about birds": [0.0, 0.0, 1.0],
}


# ── EvalSuite ─────────────────────────────────────────────────────────────────

def test_suite_all_pass():
    embedder = make_embedder("good-model", VECS_GOOD)
    report = (
        EvalSuite(name="all_pass")
        .add(EmbeddingEval("cats query", "doc about cats", CORPUS, embedder, k=1))
        .add(EmbeddingEval("dogs query", "doc about dogs", CORPUS, embedder, k=1))
        .run()
    )
    assert report.total == 2
    assert report.passed == 2
    assert report.failed == 0
    assert report.pass_rate == 1.0


def test_suite_all_fail():
    embedder = make_embedder("bad-model", VECS_BAD)
    report = (
        EvalSuite(name="all_fail")
        .add(EmbeddingEval("cats query", "doc about cats", CORPUS, embedder, k=1))
        .add(EmbeddingEval("dogs query", "doc about dogs", CORPUS, embedder, k=1))
        .run()
    )
    assert report.passed == 0
    assert report.failed == 2
    assert report.pass_rate == 0.0


def test_suite_add_returns_self():
    embedder = make_embedder("good-model", VECS_GOOD)
    suite = EvalSuite()
    result = suite.add(EmbeddingEval("cats query", "doc about cats", CORPUS, embedder))
    assert result is suite


def test_suite_empty():
    report = EvalSuite(name="empty").run()
    assert report.total == 0
    assert report.pass_rate == 0.0


# ── EmbeddingEval ─────────────────────────────────────────────────────────────

def test_embedding_eval_hit():
    embedder = make_embedder("good-model", VECS_GOOD)
    result = EmbeddingEval("cats query", "doc about cats", CORPUS, embedder, k=1).run()
    assert result.hit is True
    assert result.rank == 1


def test_embedding_eval_miss():
    embedder = make_embedder("bad-model", VECS_BAD)
    result = EmbeddingEval("cats query", "doc about cats", CORPUS, embedder, k=1).run()
    assert result.hit is False


def test_embedding_eval_custom_metric():
    embedder = make_embedder("good-model", VECS_GOOD)
    result = EmbeddingEval("cats query", "doc about cats", CORPUS, embedder, k=1, metric=Metrics.dot_product).run()
    assert result.metric_name == "dot_product"


def test_embedding_eval_default_name():
    embedder = make_embedder("my-model", VECS_GOOD)
    ev = EmbeddingEval("cats query", "doc about cats", CORPUS, embedder)
    assert "my-model" in ev.name


# ── SearchEval ────────────────────────────────────────────────────────────────

def test_search_eval_hit():
    result = SearchEval(
        query="cats",
        expected_doc="doc about cats",
        search_fn=lambda q: ["doc about cats", "doc about dogs"],
        k=1,
    ).run()
    assert result.hit is True
    assert result.rank == 1
    assert result.score == 1.0


def test_search_eval_hit_within_k():
    result = SearchEval(
        query="dogs",
        expected_doc="doc about dogs",
        search_fn=lambda q: ["doc about cats", "doc about dogs", "doc about birds"],
        k=2,
    ).run()
    assert result.hit is True
    assert result.rank == 2


def test_search_eval_miss():
    result = SearchEval(
        query="cats",
        expected_doc="doc about whales",
        search_fn=lambda q: ["doc about cats", "doc about dogs"],
        k=2,
    ).run()
    assert result.hit is False
    assert result.rank is None
    assert result.score == 0.0


# ── Evaluation engine ─────────────────────────────────────────────────────────

def test_evaluation_raises_without_corpus():
    with pytest.raises(ValueError, match="corpus_vecs required"):
        Evaluation().eval([1.0], [1.0])


def test_evaluation_rank_and_score():
    corpus_vecs = [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]
    result = Evaluation(corpus_vecs=corpus_vecs).eval(
        query_vec=[1.0, 0.0], expected_vec=[1.0, 0.0], k=1
    )
    assert result.hit is True
    assert result.rank == 1
    assert abs(result.score - 1.0) < 1e-9


# ── Retrieval metrics ─────────────────────────────────────────────────────────

def test_recall_at_k():
    assert Evaluation.recall_at_k(True) == 1.0
    assert Evaluation.recall_at_k(False) == 0.0


def test_mrr():
    assert Evaluation.mrr(1) == 1.0
    assert abs(Evaluation.mrr(2) - 0.5) < 1e-9
    assert Evaluation.mrr(None) == 0.0


def test_ndcg():
    assert abs(Evaluation.ndcg(1, k=3) - 1.0) < 1e-9
    assert abs(Evaluation.ndcg(2, k=3) - 1 / math.log2(3)) < 1e-9
    assert Evaluation.ndcg(4, k=3) == 0.0
    assert Evaluation.ndcg(None, k=3) == 0.0
