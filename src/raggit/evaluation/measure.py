from __future__ import annotations

from typing import Any, Callable, List, Optional

from ..metrics import Metrics
from ..models import EvalSingleResult


def evaluate(
    ranked_items: List[Any],
    expected_item: Any,
    k: Optional[int] = 3,
    threshold: float = 0.999,
    match_metric: Callable[[Any, Any], float] = Metrics.cosine_similarity,
    match_fn: Optional[Callable[[Any, Any], bool]] = None,
    score: Optional[float] = None,
    metric_name: str = "retrieval",
) -> EvalSingleResult:
    """
    Find the rank of expected_item in ranked_items and produce an EvalSingleResult.

    Eval factories produce a ranked list (specific to the system being tested) and
    delegate measurement to this helper, so rank/score/passed semantics stay consistent.

    Matching:
        - By default, items are matched against expected via cosine similarity ≥ threshold.
        - Override `match_metric` to use a different similarity metric (dot_product, etc.).
        - Override `match_fn` (item, expected) -> bool to bypass similarity entirely —
          useful for non-vector retrieval (IDs, strings, sparse representations).

    Scoring:
        - If `score` is None and items are vectors, defaults to match_metric(top, expected).
        - If `score` is None and `match_fn` is used, defaults to 1.0 if matched else 0.0.
        - Pass `score` explicitly to override (e.g. embedder's intrinsic query→expected score).

    `k=None` means any positive rank passes (used by chunk_eval — any matching chunk counts).
    """
    rank: Optional[int] = None
    for i, item in enumerate(ranked_items, start=1):
        if match_fn is not None:
            matched = match_fn(item, expected_item)
        else:
            matched = match_metric(item, expected_item) >= threshold
        if matched:
            rank = i
            break

    passed = rank is not None and (k is None or rank <= k)

    if score is None:
        if not ranked_items:
            score = 0.0
        elif match_fn is not None:
            score = 1.0 if rank is not None else 0.0
        else:
            score = match_metric(ranked_items[0], expected_item)

    return EvalSingleResult(
        passed=passed,
        rank=rank,
        score=score,
        metric_name=metric_name,
    )
