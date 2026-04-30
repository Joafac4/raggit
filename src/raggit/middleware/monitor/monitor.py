from __future__ import annotations

import time
from datetime import datetime
from typing import Callable, Dict, List, Optional

from ..models import Cluster
from ..stores.base import MonitorStore


class Monitor:
    def __init__(
        self,
        store: MonitorStore,
        embedder: Callable[[str], List[float]],
        cluster_threshold: float = 0.92,
    ):
        self.store = store
        self.embedder = embedder
        self.cluster_threshold = cluster_threshold
        self._schema: Dict[str, type] = store.get_schema()

    def log(
        self,
        query: str,
        latency_ms: float,
        cache_hit: bool = False,
        vec: Optional[List[float]] = None,
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
            **kwargs,
        )

    def _validate(self, kwargs: dict) -> None:
        unknown = set(kwargs) - set(self._schema)
        if unknown:
            raise ValueError(
                f"Unknown fields: {unknown}. Store schema accepts: {set(self._schema) or 'no extra fields'}"
            )
        for key, value in kwargs.items():
            expected = self._schema[key]
            if not isinstance(value, expected):
                raise TypeError(
                    f"Field '{key}' expected {expected.__name__}, got {type(value).__name__}"
                )

    @staticmethod
    def calculate_timing(start: float) -> float:
        return (time.time() - start) * 1000

    def clusters(
        self,
        top: Optional[int] = None,
        since: Optional[datetime] = None,
        last_seen_before: Optional[datetime] = None,
    ) -> List[Cluster]:
        return self.store.get_clusters(top=top, since=since, last_seen_before=last_seen_before)

    def stats(self) -> Dict:
        return self.store.stats()
