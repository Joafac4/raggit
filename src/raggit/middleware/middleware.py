from __future__ import annotations

import functools
import time
from typing import Callable, Optional

from .cache.cache import SemanticCache
from .monitor.monitor import Monitor


class Middleware:
    def __init__(
        self,
        monitor: Optional[Monitor] = None,
        cache: Optional[SemanticCache] = None,
    ):
        self.monitor = monitor
        self.cache = cache

    def track(self, fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(query: str, *args, **kwargs):
            monitor_kwargs = kwargs.pop("_monitor_kwargs", {})
            return self._execute(query, fn, *args, monitor_kwargs=monitor_kwargs, **kwargs)
        return wrapper

    def _execute(self, query: str, fn: Callable, *args, monitor_kwargs: dict = None, **kwargs):
        """Pipeline: cache → fn → monitor. Add future steps here."""
        monitor_kwargs = monitor_kwargs or {}

        cached = self.cache.get(query) if self.cache else None
        if cached is not None:
            if self.monitor:
                self.monitor.log(query, latency_ms=0.0, cache_hit=True, **monitor_kwargs)
            return cached

        start = time.time()
        result = fn(query, *args, **kwargs)

        if self.monitor:
            self.monitor.log(query, latency_ms=Monitor.calculate_timing(start), **monitor_kwargs)
        return result
