from __future__ import annotations

from typing import Callable, List

from ..evaluation.measure import evaluate
from ..models import EvalSingleResult


def index_eval(
    query_vec: List[float],
    expected_vec: List[float],
    search_fn: Callable[[List[float]], List[List[float]]],
    k: int = 3,
    threshold: float = 0.999,
) -> Callable[[], EvalSingleResult]:
    """
    Factory that returns an eval function for any search index.

    search_fn receives the query vector and returns a ranked list of vectors.
    Compatible with any backend: Faiss, Chroma, BM25, etc.

    score = cosine similarity of the top retrieved vector to expected — tells you
    how close the backend's #1 result was to the right answer. Useful for comparing
    backends.

    Usage:
        suite.add("faiss",  index_eval(query_vec, expected_vec, faiss_search))
        suite.add("chroma", index_eval(query_vec, expected_vec, chroma_search))
    """
    def _run() -> EvalSingleResult:
        results = search_fn(query_vec)
        return evaluate(
            results,
            expected_vec,
            k=k,
            threshold=threshold,
            metric_name="search",
        )

    return _run
