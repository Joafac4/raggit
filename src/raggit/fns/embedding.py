from __future__ import annotations

from typing import Callable, List

from ..evaluation.measure import evaluate
from ..metrics import Metrics, _MetricFn
from ..models import EvalSingleResult


def embedding_eval(
    query_vec: List[float],
    expected_vec: List[float],
    corpus_vecs: List[List[float]],
    k: int = 3,
    metric: _MetricFn = Metrics.cosine_similarity,
) -> Callable[[], EvalSingleResult]:
    """
    Factory that returns an eval function for embedding retrieval.

    All vectors are pre-computed by the caller — works with any modality
    (text, audio, image, video, etc.).

    score = metric(query_vec, expected_vec) — the embedder's intrinsic relevance
    assignment to the expected doc. Useful for comparing embedders.

    Usage:
        corpus_vecs = [embed(doc) for doc in docs]
        suite.add("activate", embedding_eval(embed("How to activate?"), embed("To activate..."), corpus_vecs))
    """
    def _run() -> EvalSingleResult:
        ranked = sorted(corpus_vecs, key=lambda v: metric(query_vec, v), reverse=True)
        return evaluate(
            ranked,
            expected_vec,
            k=k,
            score=metric(query_vec, expected_vec),
            metric_name=metric.__name__,
        )

    return _run
