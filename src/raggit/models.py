from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class EmbeddingPair(BaseModel):
    query: str
    relevant_doc: str
    metadata: dict = {}


class EvalResult(BaseModel):
    model_name: str
    query: str
    relevant_doc: str
    similarity_score: float
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
