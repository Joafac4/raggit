from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional

from ..models import Cluster


class MonitorStore(ABC):
    """
    Minimal contract for monitor persistence.
    Implement get_schema() and log() — everything else is optional.
    """

    @abstractmethod
    def get_schema(self) -> Dict[str, type]:
        """
        Return accepted extra fields and their Python types.
        Called once by Monitor at instantiation.
        Built-in fields (latency_ms, cache_hit) are excluded — Monitor handles those.
        """
        ...

    @abstractmethod
    def log(
        self,
        query: str,
        vec: List[float],
        latency_ms: float,
        threshold: float,
        cache_hit: bool = False,
        **kwargs,
    ) -> None:
        """
        Persist the event. kwargs are pre-validated by Monitor.
        Handles clustering: find similar cluster above threshold, create or update it.
        """
        ...

    def get_clusters(
        self,
        top: Optional[int] = None,
        since: Optional[datetime] = None,
        last_seen_before: Optional[datetime] = None,
    ) -> List[Cluster]:
        raise NotImplementedError(f"{type(self).__name__} does not support get_clusters()")

    def stats(self) -> Dict:
        raise NotImplementedError(f"{type(self).__name__} does not support stats()")


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
        cluster_id: str,
        response: str,
        approved_by: str = "llm",
    ) -> None:
        """Store cached response for cluster_id."""
        ...
