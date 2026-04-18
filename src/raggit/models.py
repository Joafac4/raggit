from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


class EmbeddingPair(BaseModel):
    query: str
    relevant_doc: str
    metadata: dict = {}


class EvalResult(BaseModel):
    model_name: str
    query: str
    relevant_doc: str
    metric_score: float
    hit: bool
    rank: Optional[int]
    human_preference: Optional[bool] = None
    timestamp: datetime


class EvalRun(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    model_a: str
    model_b: str
    pairs: List[EmbeddingPair]
    results_a: List[EvalResult]
    results_b: List[EvalResult]
    winner: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    # Attached after construction by the evaluator; excluded from serialization
    report: Optional[Any] = Field(default=None, exclude=True)


class EvalSingleResult(BaseModel):
    hit: bool
    rank: Optional[int]
    score: float
    k: int
    metric_name: str


class SuiteEvalResult(BaseModel):
    eval_name: str
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
        from .evaluation.report import show as _show  # lazy to avoid circular import
        _show(self)
