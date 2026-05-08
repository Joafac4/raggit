from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from ..models import Cluster, RetrievalHandle


class MonitorStore(ABC):
    """
    Contract for monitor persistence.
    Implement get_schema() and log() — everything else is optional.

    Feedback persistence is a separate concern handled by FeedbackStore. A
    MonitorStore knows nothing about feedback; pair it with a FeedbackStore
    when you want feedback support.
    """

    @property
    def keeps_events(self) -> bool:
        """
        Whether this store maintains per-event records (vs aggregates only).
        Stores that override should set this to True. Drives whether
        track_with_handle generates an event_id and whether feedback can be
        attached at event-level granularity.
        """
        return False

    @abstractmethod
    def get_schema(self) -> Dict[str, type]:
        """
        Return accepted extra fields and their Python types.
        Called once by Monitor at instantiation.
        Built-in fields (latency_ms, cache_hit) are excluded — Monitor handles those.
        """
        ...

    def assign_cluster(self, vec: List[float], threshold: float, query: str) -> str:
        """
        Synchronously find or create a cluster for vec, bumping count and
        last_seen. Returns the cluster_id. Used by track_with_handle to make
        cluster_id available before the event row is written.

        `query` is used as the cluster's representative_query when a new
        cluster is created (the very first query that forms the cluster).

        Optional. Stores that don't implement this can still be used with
        @mw.track (the original API), but @mw.track_with_handle will raise.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support assign_cluster() — "
            f"required by Middleware.track_with_handle"
        )

    @abstractmethod
    def log(
        self,
        query: str,
        vec: List[float],
        latency_ms: float,
        threshold: float,
        cache_hit: bool = False,
        cluster_id: Optional[str] = None,
        event_id: Optional[str] = None,
        **kwargs,
    ) -> None:
        """
        Persist the event. kwargs are pre-validated by Monitor.

        If cluster_id is None, the store performs clustering itself (the
        original behavior, used by @mw.track). If cluster_id is provided
        (via @mw.track_with_handle), the store skips clustering and trusts
        the caller — assign_cluster has already run and bumped the count.

        event_id is used by stores that keep per-event history; ignored by
        aggregate-only stores.
        """
        ...

    def get_clusters(
        self,
        top: Optional[int] = None,
        since: Optional[datetime] = None,
        last_seen_before: Optional[datetime] = None,
        min_count: Optional[int] = None,
    ) -> List[Cluster]:
        raise NotImplementedError(f"{type(self).__name__} does not support get_clusters()")

    def get_cluster(self, cluster_id: str) -> Optional[Cluster]:
        """Fetch a single cluster by id. Returns None if not found.

        Used by AutoCachePromoter to evaluate eligibility for one cluster
        without paying for a full get_clusters() scan.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support get_cluster()")

    def update_latest_response(self, cluster_id: str, response: str) -> None:
        """Persist the latest answer produced for this cluster, overwriting
        any previous one. Used by AutoCachePromoter as the candidate
        response that gets promoted to cache when thresholds are met.
        Default: NotImplementedError. Stores must opt in.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support update_latest_response()"
        )

    def stats(self) -> Dict:
        raise NotImplementedError(f"{type(self).__name__} does not support stats()")


class FeedbackStore(ABC):
    """
    Contract for feedback persistence — separate from MonitorStore.

    A FeedbackStore is paired with a MonitorStore at the Monitor layer. Both
    typically share the same DB file; the storage layout is up to each
    implementation. Some FeedbackStore impls add their own table (e.g. a
    `feedback` table FK'd to events); others append columns to a table that
    the paired MonitorStore created (e.g. counter columns on `clusters`).

    Implementations must be idempotent on schema initialization so they can
    coexist with the paired MonitorStore regardless of instantiation order.
    """

    @abstractmethod
    def record_feedback(
        self,
        handle: RetrievalHandle,
        accepted: Optional[bool] = None,
        score: Optional[float] = None,
        comment: Optional[str] = None,
    ) -> None:
        """
        Record feedback for the retrieval that produced `handle`. Validation
        ("at least one of accepted/score") is performed at the Monitor layer
        before this is called.
        """
        ...

    def get_feedback_summary(self, cluster_id: str) -> Optional[Dict]:
        """Return per-cluster feedback aggregates for one cluster, or None
        if the cluster has no feedback at all. Shape:
            {
                "feedback_count": int,
                "accepted_count": int,
                "acceptance_rate": Optional[float],  # None if no thumb feedback
                "score_count": int,
                "avg_score": Optional[float],         # None if no score feedback
            }

        Used by AutoCachePromoter to evaluate eligibility for one cluster
        without paying for an aggregate-over-all-clusters query.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support get_feedback_summary()"
        )

    def acceptance_rate_per_cluster(
        self,
        top: Optional[int] = None,
    ) -> List[Tuple[Cluster, float]]:
        """Return (cluster, acceptance_rate) for clusters that have thumb feedback."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support acceptance_rate_per_cluster()"
        )

    def avg_score_per_cluster(
        self,
        top: Optional[int] = None,
    ) -> List[Tuple[Cluster, float]]:
        """Return (cluster, avg_score) for clusters that have score feedback."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support avg_score_per_cluster()"
        )


class CacheStore(ABC):
    """
    Contract for cache persistence.
    Each implementation manages its own vector index — independent from MonitorStore.
    """

    @abstractmethod
    def get(self, vec: List[float], threshold: float) -> Optional[str]:
        """
        Search own vector store for similar vector above threshold.
        Returns cached response or None.
        """
        ...

    @abstractmethod
    def set(
        self,
        vec: List[float],
        response: str,
        approved_by: str = "llm",
        cluster_id: Optional[str] = None,
    ) -> None:
        """Store cached response for vec.

        If `cluster_id` is provided and an entry with the same cluster_id and
        approved_by='auto' already exists, the call is idempotent (no insert,
        no error). This protects AutoCachePromoter from double-promotion under
        concurrent feedback events.
        """
        ...

    def delete_by_cluster(self, cluster_id: str) -> None:
        """Delete cache entries with this cluster_id and approved_by='auto'.
        Human-set entries are never deleted by this. Used by AutoCachePromoter
        for demotion. Default raises NotImplementedError.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support delete_by_cluster()"
        )

    def has_auto_entry(self, cluster_id: str) -> bool:
        """True iff an auto-promoted cache entry exists for this cluster_id.
        Used by AutoCachePromoter to decide whether to promote/demote/skip.
        Default raises NotImplementedError.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support has_auto_entry()"
        )
