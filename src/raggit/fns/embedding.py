from __future__ import annotations

from typing import Callable, List, Optional

from ..metrics import Metrics, _MetricFn
from ..models import EvalSingleResult

_VECTOR_MATCH_THRESHOLD = 0.999


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

    Usage:
        corpus_vecs = [embed(doc) for doc in docs]

        suite.add("activate", embedding_eval(embed("How to activate?"), embed("To activate..."), corpus_vecs))
        suite.add("reset",    embedding_eval(embed("Reset password"),   embed("Visit login..."), corpus_vecs))
    """
    def _run() -> EvalSingleResult:
        scored = [
            (i, metric(query_vec, doc_vec))
            for i, doc_vec in enumerate(corpus_vecs)
        ]
        ranked = sorted(scored, key=lambda x: x[1], reverse=True)

        rank: Optional[int] = None
        for position, (idx, _) in enumerate(ranked, start=1):
            if Metrics.cosine_similarity(expected_vec, corpus_vecs[idx]) >= _VECTOR_MATCH_THRESHOLD:
                rank = position
                break

        return EvalSingleResult(
            passed=rank is not None and rank <= k,
            score=metric(query_vec, expected_vec),
            rank=rank,
            metric_name=metric.__name__,
        )

    return _run
