from __future__ import annotations

from typing import Optional

from ..models import Cluster, RetrievalHandle
from ..stores.base import FeedbackStore, MonitorStore
from .cache import SemanticCache


class AutoCachePromoter:
    """
    Promotes cluster answers to the semantic cache once thresholds are met,
    and demotes them if quality later drops.

    Wiring (both Monitor and Middleware hold the same instance):

        promoter = AutoCachePromoter(cache, store, feedback_store, min_count=10)
        monitor  = Monitor(embed, store=store, feedback_store=feedback_store, auto_promoter=promoter)
        mw       = Middleware(monitor=monitor, cache=cache, embedder=embed, auto_promoter=promoter)

    Trigger semantics — all gates are AND. None means "ignore this dimension":
        min_count       — required (default 10)
        min_acceptance  — optional [0..1]; ignored if None
        min_score       — optional [0..1]; ignored if None

    Demotion only touches entries with approved_by='auto'. Human-set entries
    are never auto-evicted.
    """

    def __init__(
        self,
        cache: SemanticCache,
        store: MonitorStore,
        feedback_store: FeedbackStore,
        min_count: int = 10,
        min_acceptance: Optional[float] = 0.8,
        min_score: Optional[float] = None,
    ):
        self.cache = cache
        self.store = store
        self.feedback_store = feedback_store
        self.min_count = min_count
        self.min_acceptance = min_acceptance
        self.min_score = min_score

    # ── Hooks ────────────────────────────────────────────────────────────────

    def on_answer(self, handle: RetrievalHandle, response: str) -> None:
        """Called after a track_with_handle call produces an answer.

        Persists the latest response for the cluster (the candidate for
        promotion) and re-evaluates eligibility — count just went up by one,
        so this can flip a cluster from ineligible to eligible.
        """
        self.store.update_latest_response(handle.cluster_id, response)
        self._reconcile(handle.cluster_id)

    def on_feedback(self, handle: RetrievalHandle) -> None:
        """Called after monitor.record_feedback persists feedback.

        Re-evaluates eligibility — feedback aggregates just changed, which
        can flip a cluster either way (promote or demote).
        """
        self._reconcile(handle.cluster_id)

    # ── Core reconciliation ──────────────────────────────────────────────────

    def _reconcile(self, cluster_id: str) -> None:
        eligible = self._is_eligible(cluster_id)
        cached = self.cache.has_auto_entry(cluster_id)

        if eligible and not cached:
            cluster = self.store.get_cluster(cluster_id)
            if cluster is not None:
                self._promote(cluster)
        elif cached and not eligible:
            self.cache.delete_auto(cluster_id)
        # else: nothing to do.

    def _is_eligible(self, cluster_id: str) -> bool:
        cluster = self.store.get_cluster(cluster_id)
        if cluster is None or cluster.count < self.min_count:
            return False

        if self.min_acceptance is None and self.min_score is None:
            return True  # count-only mode

        summary = self.feedback_store.get_feedback_summary(cluster_id)
        if summary is None:
            return False  # quality gate set but no feedback yet

        if self.min_acceptance is not None:
            rate = summary.get("acceptance_rate")
            if rate is None or rate < self.min_acceptance:
                return False

        if self.min_score is not None:
            avg = summary.get("avg_score")
            if avg is None or avg < self.min_score:
                return False

        return True

    def _promote(self, cluster: Cluster) -> None:
        if cluster.latest_response is None:
            return  # no candidate response yet
        self.cache.set_auto(
            vec=cluster.representative_vec,
            response=cluster.latest_response,
            cluster_id=cluster.cluster_id,
        )
