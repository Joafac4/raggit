import math
import pytest

from raggit import Embedder, EmbeddingPair, RaggitEval


def _make_embedder(name: str, vectors: dict) -> Embedder:
    return Embedder(name, embed_fn=lambda text: vectors[text])


def _cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x**2 for x in a)) * math.sqrt(sum(y**2 for y in b)))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

PAIRS = [
    EmbeddingPair(query="hello", relevant_doc="world"),
    EmbeddingPair(query="foo", relevant_doc="bar"),
]

# model_a: high similarity for both pairs
VECTORS_A = {
    "hello": [1.0, 0.0],
    "world": [0.95, 0.05],
    "foo":   [0.0, 1.0],
    "bar":   [0.05, 0.95],
}

# model_b: low similarity for both pairs (orthogonal vectors)
VECTORS_B = {
    "hello": [1.0, 0.0],
    "world": [0.0, 1.0],
    "foo":   [0.0, 1.0],
    "bar":   [1.0, 0.0],
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_compare_returns_eval_run():
    model_a = _make_embedder("model-a", VECTORS_A)
    model_b = _make_embedder("model-b", VECTORS_B)
    run = RaggitEval(pairs=PAIRS).compare(model_a, model_b)

    assert run.model_a == "model-a"
    assert run.model_b == "model-b"
    assert len(run.results_a) == len(PAIRS)
    assert len(run.results_b) == len(PAIRS)


def test_similarity_scores_are_correct():
    model_a = _make_embedder("model-a", VECTORS_A)
    model_b = _make_embedder("model-b", VECTORS_B)
    run = RaggitEval(pairs=PAIRS).compare(model_a, model_b)

    expected_a0 = _cosine(VECTORS_A["hello"], VECTORS_A["world"])
    assert abs(run.results_a[0].similarity_score - expected_a0) < 1e-9

    expected_b0 = _cosine(VECTORS_B["hello"], VECTORS_B["world"])
    assert abs(run.results_b[0].similarity_score - expected_b0) < 1e-9


def test_winner_is_model_a_when_it_scores_higher_on_all_pairs():
    model_a = _make_embedder("model-a", VECTORS_A)
    model_b = _make_embedder("model-b", VECTORS_B)
    run = RaggitEval(pairs=PAIRS).compare(model_a, model_b)

    assert run.winner == "model-a"


def test_winner_is_model_b_when_it_scores_higher():
    # Swap roles: model_b vectors are closer
    model_a = _make_embedder("model-a", VECTORS_B)
    model_b = _make_embedder("model-b", VECTORS_A)
    run = RaggitEval(pairs=PAIRS).compare(model_a, model_b)

    assert run.winner == "model-b"


def test_tie_sets_winner_to_none():
    # One pair each
    single_pair = [EmbeddingPair(query="hello", relevant_doc="world")]
    tie_vectors_a = {"hello": [1.0, 0.0], "world": [0.9, 0.1]}
    tie_vectors_b = {"hello": [1.0, 0.0], "world": [0.9, 0.1]}

    model_a = _make_embedder("model-a", tie_vectors_a)
    model_b = _make_embedder("model-b", tie_vectors_b)
    run = RaggitEval(pairs=single_pair).compare(model_a, model_b)

    assert run.winner is None


def test_report_is_attached():
    model_a = _make_embedder("model-a", VECTORS_A)
    model_b = _make_embedder("model-b", VECTORS_B)
    run = RaggitEval(pairs=PAIRS).compare(model_a, model_b)

    assert run.report is not None
    assert hasattr(run.report, "show")


def test_unsupported_mode_raises():
    model_a = _make_embedder("model-a", VECTORS_A)
    model_b = _make_embedder("model-b", VECTORS_B)

    with pytest.raises(NotImplementedError):
        RaggitEval(pairs=PAIRS).compare(model_a, model_b, mode="human")
