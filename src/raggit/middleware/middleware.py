from __future__ import annotations

import functools
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional

from .cache.cache import SemanticCache
from .monitor.monitor import Monitor

logger = logging.getLogger(__name__)


class Middleware:
    def __init__(
        self,
        monitor: Optional[Monitor] = None,
        cache: Optional[SemanticCache] = None,
        monitor_workers: int = 2,
    ):
        self.monitor = monitor
        self.cache = cache
        self._executor = ThreadPoolExecutor(max_workers=monitor_workers) if monitor else None

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
            self._log_monitor(query, latency_ms=0.0, cache_hit=True, **monitor_kwargs)
            return cached

        start = time.time()
        result = fn(query, *args, **kwargs)

        self._log_monitor(query, latency_ms=Monitor.calculate_timing(start), **monitor_kwargs)
        return result

    def _log_monitor(self, query: str, **kwargs) -> None:
        if self.monitor is None:
            return
        self._executor.submit(self._safe_log, query, **kwargs)

    def _safe_log(self, query: str, **kwargs) -> None:
        try:
            self.monitor.log(query, **kwargs)
        except Exception:
            logger.exception("Monitor.log() failed for query %r", query)

    def shutdown(self, wait: bool = True) -> None:
        """Shut down the monitor thread pool. Call on app exit to flush pending logs."""
        if self._executor:
            self._executor.shutdown(wait=wait)
