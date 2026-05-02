from __future__ import annotations

from typing import List, Optional

from ..metrics import Metrics
from ..models import EvalSingleResult


def evaluate(
    ranked_vecs: List[List[float]],
    expected_vec: List[float],
    k: Optional[int] = 3,
    threshold: float = 0.999,
    score: Optional[float] = None,
    metric_name: str = "retrieval",
) -> EvalSingleResult:
    """
    Find the rank of expected_vec in ranked_vecs and produce an EvalSingleResult.

    Eval factories produce a ranked list (specific to the system being tested) and
    delegate measurement to this helper, so rank/score/passed semantics stay consistent.

    If `score` is not provided, defaults to the cosine similarity of the top-ranked
    vector to expected — useful for "how close did the system get?". Factories that
    measure something else (e.g. embedder's intrinsic query→expected relevance) pass
    their own `score`.

    `k=None` means any positive rank passes (used by chunk_eval — any matching chunk
    counts).
    """
    rank: Optional[int] = None
    for i, vec in enumerate(ranked_vecs, start=1):
        if Metrics.cosine_similarity(vec, expected_vec) >= threshold:
            rank = i
            break

    passed = rank is not None and (k is None or rank <= k)
    if score is None:
        score = Metrics.cosine_similarity(ranked_vecs[0], expected_vec) if ranked_vecs else 0.0

    return EvalSingleResult(
        passed=passed,
        rank=rank,
        score=score,
        metric_name=metric_name,
    )
