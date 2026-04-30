from __future__ import annotations

import uuid
from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class Cluster(BaseModel):
    cluster_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    representative_query: str
    representative_vec: List[float]
    count: int = 0
    created_at: datetime = Field(default_factory=datetime.now)
    last_seen: datetime = Field(default_factory=datetime.now)
    avg_retrieval_rank: Optional[float] = None
    avg_retrieval_score: Optional[float] = None


class Event(BaseModel):
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    cluster_id: str
    query_text: str
    latency_ms: Optional[float] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    cache_hit: bool = False
    retrieval_rank: Optional[int] = None
    retrieval_score: Optional[float] = None
    retrieved_doc_ids: Optional[List[str]] = None
    user_feedback: Optional[bool] = None
