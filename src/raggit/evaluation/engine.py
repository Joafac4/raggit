from __future__ import annotations

import math
from typing import Callable, List, Optional

from ..metrics import Metrics
from ..models import EvalSingleResult

_VECTOR_MATCH_THRESHOLD = 0.999
_MetricFn = Callable[[List[float], List[float]], float]


class Evaluation:
    def __init__(
        self,
        corpus_vecs: Optional[List[List[float]]] = None,
    ) -> None:
        self.corpus_vecs = corpus_vecs

    def eval(
        self,
        query_vec: List[float],
        expected_vec: List[float],
        k: int = 3,
        metric: _MetricFn = Metrics.cosine_similarity,
    ) -> EvalSingleResult:
        if self.corpus_vecs is None:
            raise ValueError(
                "corpus_vecs required for eval(). Pass them in __init__."
            )

        scored = [
            (i, metric(query_vec, doc_vec))
            for i, doc_vec in enumerate(self.corpus_vecs)
        ]
        ranked = sorted(scored, key=lambda x: x[1], reverse=True)

        rank: Optional[int] = None
        for position, (idx, _) in enumerate(ranked, start=1):
            if Metrics.cosine_similarity(expected_vec, self.corpus_vecs[idx]) >= _VECTOR_MATCH_THRESHOLD:
                rank = position
                break

        return EvalSingleResult(
            hit=rank is not None and rank <= k,
            rank=rank,
            score=metric(query_vec, expected_vec),
            k=k,
            metric_name=metric.__name__,
        )

    def eval_search(
        self,
        query: str,
        expected_doc: str,
        search_fn: Callable[[str], List[str]],
        k: int = 3,
    ) -> EvalSingleResult:
        results = search_fn(query)

        rank: Optional[int] = None
        for position, doc in enumerate(results, start=1):
            if doc.strip() == expected_doc.strip():
                rank = position
                break

        return EvalSingleResult(
            hit=rank is not None and rank <= k,
            rank=rank,
            score=1 / rank if rank else 0.0,
            k=k,
            metric_name="search",
        )

    # ── Retrieval metrics ────────────────────────────────────────────────────

    @staticmethod
    def recall_at_k(hit: bool) -> float:
        return 1.0 if hit else 0.0

    @staticmethod
    def mrr(rank: Optional[int]) -> float:
        return 1 / rank if rank else 0.0

    @staticmethod
    def ndcg(rank: Optional[int], k: int) -> float:
        return 1 / math.log2(rank + 1) if (rank and rank <= k) else 0.0
