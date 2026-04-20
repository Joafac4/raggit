from __future__ import annotations

from datetime import datetime
from typing import Callable, List, Optional

from pydantic import BaseModel, Field


class EvalSingleResult(BaseModel):
    passed: bool
    score: Optional[float] = None
    rank: Optional[int] = None
    metadata: dict = {}
    metric_name: str = ""


class SuiteEvalResult(BaseModel):
    name: str
    result: EvalSingleResult


class AggregationResult(BaseModel):
    name: str
    value: float


class SuiteReport(BaseModel):
    suite_name: str
    results: List[SuiteEvalResult]
    total: int
    passed: int
    failed: int
    pass_rate: float
    created_at: datetime = Field(default_factory=datetime.now)
    aggregations: List[AggregationResult] = []

    def aggregate(
        self,
        fn: Callable[[Optional[int]], float],
        name: str = "",
    ) -> SuiteReport:
        scores = [fn(r.result.rank) for r in self.results]
        avg = sum(scores) / self.total if self.total else 0.0
        new_agg = AggregationResult(name=name or fn.__name__, value=avg)
        return self.model_copy(update={"aggregations": [*self.aggregations, new_agg]})

    def show(self) -> None:
        from .evaluation.report import show as _show  # lazy to avoid circular import
        _show(self)
