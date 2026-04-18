from __future__ import annotations

from typing import Callable, List, Optional

from ..base import BaseEval
from ...models import EvalSingleResult


class SearchEval(BaseEval):
    def __init__(
        self,
        query: str,
        expected_doc: str,
        search_fn: Callable[[str], List[str]],
        k: int = 3,
        name: str = "",
    ):
        self.query = query
        self.expected_doc = expected_doc
        self.search_fn = search_fn
        self.k = k
        self.name = name or f"search: {query[:40]}"

    def run(self) -> EvalSingleResult:
        results = self.search_fn(self.query)

        rank: Optional[int] = None
        for position, doc in enumerate(results, start=1):
            if doc.strip() == self.expected_doc.strip():
                rank = position
                break

        return EvalSingleResult(
            hit=rank is not None and rank <= self.k,
            rank=rank,
            score=1 / rank if rank else 0.0,
            k=self.k,
            metric_name="search",
        )
