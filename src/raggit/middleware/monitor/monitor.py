from __future__ import annotations

import time
import uuid
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple

from ..models import Cluster, RetrievalHandle
from ..stores.base import FeedbackStore, MonitorStore

_BUILTIN_FIELDS = frozenset({"retrieval_score", "retrieved_doc_ids"})


class Monitor:
    def __init__(
        self,
        store: MonitorStore,
        embedder: Callable[[str], List[float]],
        cluster_threshold: float = 0.92,
        feedback_store: Optional[FeedbackStore] = None,
    ):
        self.store = store
        self.embedder = embedder
        self.cluster_threshold = cluster_threshold
        self.feedback_store = feedback_store
        self._schema: Dict[str, type] = store.get_schema()

    def log(
        self,
        query: str,
        latency_ms: float,
        cache_hit: bool = False,
        vec: Optional[List[float]] = None,
        cluster_id: Optional[str] = None,
        event_id: Optional[str] = None,
        **kwargs,
    ) -> None:
        self._validate(kwargs)
        if vec is None:
            vec = self.embedder(query)
        self.store.log(
            query=query,
            vec=vec,
            latency_ms=latency_ms,
            threshold=self.cluster_threshold,
            cache_hit=cache_hit,
            cluster_id=cluster_id,
            event_id=event_id,
            **kwargs,
        )

    def assign_cluster(
        self,
        query: str,
        vec: Optional[List[float]] = None,
    ) -> Tuple[str, Optional[str], List[float]]:
        """
        Synchronously cluster `query` and return (cluster_id, event_id, vec).

        event_id is generated up-front when the store keeps events; otherwise None.
        Used by Middleware.track_with_handle so the IDs are available before the
        async event write completes.
        """
        if vec is None:
            vec = self.embedder(query)
        cluster_id = self.store.assign_cluster(vec, self.cluster_threshold, query)
        event_id = str(uuid.uuid4()) if self.store.keeps_events else None
        return cluster_id, event_id, vec

    def _validate(self, kwargs: dict) -> None:
        user_kwargs = {k: v for k, v in kwargs.items() if k not in _BUILTIN_FIELDS}
        unknown = set(user_kwargs) - set(self._schema)
        if unknown:
            raise ValueError(
                f"Unknown fields: {unknown}. Store schema accepts: {set(self._schema) or 'no extra fields'}"
            )
        for key, value in user_kwargs.items():
            expected = self._schema[key]
            if not isinstance(value, expected):
                raise TypeError(
                    f"Field '{key}' expected {expected.__name__}, got {type(value).__name__}"
                )

    @staticmethod
    def calculate_timing(start: float) -> float:
        return (time.time() - start) * 1000

    def popular_queries(
        self,
        top: Optional[int] = None,
        since: Optional[datetime] = None,
        last_seen_before: Optional[datetime] = None,
    ) -> List[Cluster]:
        return self.store.get_clusters(top=top, since=since, last_seen_before=last_seen_before)

    def events(
        self,
        cluster_id: Optional[str] = None,
        since: Optional[datetime] = None,
        has_retrieved_docs: Optional[bool] = None,
    ):
        return self.store.get_events(
            cluster_id=cluster_id,
            since=since,
            has_retrieved_docs=has_retrieved_docs,
        )

    def stats(self) -> Dict:
        return self.store.stats()

    # ── Feedback ──────────────────────────────────────────────────────────────
    # Delegated to feedback_store if one was provided. If not, the methods raise
    # a clear error pointing the user at the FeedbackStore wiring.

    def record_feedback(
        self,
        handle: RetrievalHandle,
        accepted: Optional[bool] = None,
        score: Optional[float] = None,
        comment: Optional[str] = None,
    ) -> None:
        """
        Record feedback for a retrieval. At least one of `accepted` or `score`
        must be provided — a feedback row with no signal is rejected.

        Requires a feedback_store to be passed to the Monitor at construction.
        """
        if accepted is None and score is None:
            raise ValueError(
                "record_feedback requires at least one of accepted= or score= "
                "(both were None — no signal to record)."
            )
        self._require_feedback_store("record_feedback")
        self.feedback_store.record_feedback(
            handle, accepted=accepted, score=score, comment=comment
        )

    def acceptance_rate_per_cluster(
        self,
        top: Optional[int] = None,
    ) -> List[Tuple[Cluster, float]]:
        self._require_feedback_store("acceptance_rate_per_cluster")
        return self.feedback_store.acceptance_rate_per_cluster(top=top)

    def avg_score_per_cluster(
        self,
        top: Optional[int] = None,
    ) -> List[Tuple[Cluster, float]]:
        self._require_feedback_store("avg_score_per_cluster")
        return self.feedback_store.avg_score_per_cluster(top=top)

    def _require_feedback_store(self, op: str) -> None:
        if self.feedback_store is None:
            raise ValueError(
                f"Monitor.{op}() requires a feedback_store. "
                f"Pass one when constructing Monitor:\n"
                f"    Monitor(store=..., embedder=..., feedback_store=SQLiteEventFeedbackStore('...'))"
            )
