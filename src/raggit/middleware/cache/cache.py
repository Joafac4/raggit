from __future__ import annotations

from typing import Callable, List, Optional

from ..stores.base import CacheStore


class SemanticCache:
    def __init__(
        self,
        store: CacheStore,
        embedder: Callable[[str], List[float]],
        threshold: float = 0.95,
    ):
        self.store = store
        self.embedder = embedder
        self.threshold = threshold

    def get(self, query: str, vec: Optional[List[float]] = None) -> Optional[str]:
        if vec is None:
            vec = self.embedder(query)
        return self.store.get(vec, self.threshold)

    def set(self, query: str, response: str, approved_by: str = "llm") -> None:
        vec = self.embedder(query)
        self.store.set(vec, response, approved_by)

    # ── Auto-promotion helpers ────────────────────────────────────────────────
    # Used by AutoCachePromoter. Operate on pre-computed vectors and tag
    # entries with cluster_id + approved_by='auto' so demotion can find them.

    def set_auto(self, vec: List[float], response: str, cluster_id: str) -> None:
        """Idempotent: no-op if an auto entry already exists for cluster_id."""
        self.store.set(vec, response, approved_by="auto", cluster_id=cluster_id)

    def delete_auto(self, cluster_id: str) -> None:
        """Demote an auto-promoted entry. Does nothing for human-set entries."""
        self.store.delete_by_cluster(cluster_id)

    def has_auto_entry(self, cluster_id: str) -> bool:
        return self.store.has_auto_entry(cluster_id)
