from __future__ import annotations

from datetime import datetime
from typing import List, Optional

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


class SuiteReport(BaseModel):
    suite_name: str
    results: List[SuiteEvalResult]
    total: int
    passed: int
    failed: int
    pass_rate: float
    created_at: datetime = Field(default_factory=datetime.now)

    def show(self) -> None:
        from .report import show as _show  # lazy to avoid circular import
        _show(self)
