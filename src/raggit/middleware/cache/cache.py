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

    def set(self, cluster_id: str, response: str, approved_by: str = "llm") -> None:
        self.store.set(cluster_id, response, approved_by)
