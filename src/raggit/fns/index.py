from __future__ import annotations

from typing import Callable, List, Optional

from ..metrics import Metrics
from ..models import EvalSingleResult

_VECTOR_MATCH_THRESHOLD = 0.999


def index_eval(
    query_vec: List[float],
    expected_vec: List[float],
    search_fn: Callable[[List[float]], List[List[float]]],
    k: int = 3,
    threshold: float = _VECTOR_MATCH_THRESHOLD,
) -> Callable[[], EvalSingleResult]:
    """
    Factory that returns an eval function for any search index.

    search_fn receives the query vector and returns a ranked list of vectors.
    Compatible with any backend: Faiss, Chroma, BM25, etc.

    Usage:
        suite.add("faiss",  index_eval(query_vec, expected_vec, faiss_search))
        suite.add("chroma", index_eval(query_vec, expected_vec, chroma_search))
    """
    def _run() -> EvalSingleResult:
        results = search_fn(query_vec)

        rank: Optional[int] = None
        for position, vec in enumerate(results, start=1):
            if Metrics.cosine_similarity(vec, expected_vec) >= threshold:
                rank = position
                break

        return EvalSingleResult(
            passed=rank is not None and rank <= k,
            score=1 / rank if rank else 0.0,
            rank=rank,
            metric_name="search",
        )

    return _run
