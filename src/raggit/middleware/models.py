from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, List, Optional

from pydantic import BaseModel, Field


@dataclass
class RetrievalHandle:
    """Returned by Middleware.track_with_handle. Carries the IDs needed to
    record feedback against the retrieval that produced `answer`.

    `cluster_id` is always populated. `event_id` is only set when the store
    keeps per-event history (e.g. SQLiteMonitorStore); for aggregate-only
    stores like SQLiteClusterStore it is None.
    """
    answer: Any
    cluster_id: str
    event_id: Optional[str] = None


class Cluster(BaseModel):
    cluster_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    representative_query: str
    representative_vec: List[float]
    count: int = 0
    created_at: datetime = Field(default_factory=datetime.now)
    last_seen: datetime = Field(default_factory=datetime.now)


class Event(BaseModel):
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    cluster_id: str
    query_text: str
    latency_ms: Optional[float] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    cache_hit: bool = False
    retrieval_score: Optional[float] = None
    retrieved_doc_ids: Optional[List[str]] = None
