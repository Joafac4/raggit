from __future__ import annotations

from typing import Callable, List, Optional

from ..models import EvalSingleResult


def index_eval(
    query: str,
    expected: str,
    search_fn: Callable[[str], List[str]],
    k: int = 3,
) -> Callable[[], EvalSingleResult]:
    """
    Factory that returns an eval function for any search index.

    search_fn receives the raw query string and returns a ranked list of
    documents. Compatible with any backend: Faiss, Chroma, BM25, etc.

    Usage:
        suite.add("faiss",  index_eval("How to activate?", "To activate...", faiss_search))
        suite.add("chroma", index_eval("How to activate?", "To activate...", chroma_search))
    """
    def _run() -> EvalSingleResult:
        results = search_fn(query)

        rank: Optional[int] = None
        for position, doc in enumerate(results, start=1):
            if doc.strip() == expected.strip():
                rank = position
                break

        return EvalSingleResult(
            passed=rank is not None and rank <= k,
            score=1 / rank if rank else 0.0,
            rank=rank,
            metric_name="search",
        )

    return _run
