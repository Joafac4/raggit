import time
from typing import Dict, List

import pytest

from raggit.middleware import (
    Middleware,
    Monitor,
    MonitorStore,
    SemanticCache,
    SQLiteCacheStore,
    SQLiteClusterStore,
    SQLiteMonitorStore,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def embed(text: str) -> List[float]:
    if "password" in text.lower():
        return [1.0, 0.0, 0.0]
    if "account" in text.lower():
        return [0.0, 1.0, 0.0]
    return [0.0, 0.0, 1.0]


class FakeMonitorStore(MonitorStore):
    def __init__(self, schema: Dict[str, type] = None):
        self._schema = schema or {}
        self.logged = []

    def get_schema(self) -> Dict[str, type]:
        return self._schema

    def log(self, query, vec, latency_ms, threshold, cache_hit=False, **kwargs):
        self.logged.append({"query": query, "vec": vec, "latency_ms": latency_ms, "cache_hit": cache_hit, **kwargs})


# ── SQLiteCacheStore ──────────────────────────────────────────────────────────

def test_cache_store_miss(tmp_path):
    store = SQLiteCacheStore(str(tmp_path / "cache.db"))
    assert store.get([1.0, 0.0, 0.0], threshold=0.9) is None


def test_cache_store_hit(tmp_path):
    store = SQLiteCacheStore(str(tmp_path / "cache.db"))
    store.set([1.0, 0.0, 0.0], "Reset via settings.", approved_by="human")
    assert store.get([1.0, 0.0, 0.0], threshold=0.9) == "Reset via settings."


def test_cache_store_below_threshold(tmp_path):
    store = SQLiteCacheStore(str(tmp_path / "cache.db"))
    store.set([1.0, 0.0, 0.0], "Reset via settings.")
    assert store.get([0.0, 1.0, 0.0], threshold=0.9) is None  # orthogonal vec


# ── SQLiteMonitorStore ────────────────────────────────────────────────────────

def test_monitor_store_log_creates_cluster(tmp_path):
    store = SQLiteMonitorStore(str(tmp_path / "monitor.db"))
    store.log("reset my password", vec=[1.0, 0.0, 0.0], latency_ms=42.0, threshold=0.9)
    clusters = store.get_clusters()
    assert len(clusters) == 1
    assert clusters[0].representative_query == "reset my password"
    assert clusters[0].count == 1


def test_monitor_store_merges_similar_queries(tmp_path):
    store = SQLiteMonitorStore(str(tmp_path / "monitor.db"))
    store.log("reset my password", vec=[1.0, 0.0, 0.0], latency_ms=10.0, threshold=0.9)
    store.log("how do I reset password", vec=[0.99, 0.1, 0.0], latency_ms=12.0, threshold=0.9)
    clusters = store.get_clusters()
    assert len(clusters) == 1
    assert clusters[0].count == 2


def test_monitor_store_creates_separate_clusters(tmp_path):
    store = SQLiteMonitorStore(str(tmp_path / "monitor.db"))
    store.log("reset my password", vec=[1.0, 0.0, 0.0], latency_ms=10.0, threshold=0.9)
    store.log("activate account", vec=[0.0, 1.0, 0.0], latency_ms=10.0, threshold=0.9)
    assert len(store.get_clusters()) == 2


def test_monitor_store_stats(tmp_path):
    store = SQLiteMonitorStore(str(tmp_path / "monitor.db"))
    store.log("reset my password", vec=[1.0, 0.0, 0.0], latency_ms=10.0, threshold=0.9)
    store.log("activate account", vec=[0.0, 1.0, 0.0], latency_ms=10.0, threshold=0.9)
    stats = store.stats()
    assert stats["total_events"] == 2
    assert stats["unique_clusters"] == 2


# ── SQLiteClusterStore ────────────────────────────────────────────────────────

def test_cluster_store_merges_similar_queries(tmp_path):
    store = SQLiteClusterStore(str(tmp_path / "clusters.db"))
    store.log("reset my password", vec=[1.0, 0.0, 0.0], latency_ms=10.0, threshold=0.9)
    store.log("how do I reset password", vec=[0.99, 0.1, 0.0], latency_ms=10.0, threshold=0.9)
    stats = store.stats()
    assert stats["unique_clusters"] == 1
    assert stats["total_events"] == 2


# ── SemanticCache ─────────────────────────────────────────────────────────────

def test_semantic_cache_miss(tmp_path):
    cache = SemanticCache(SQLiteCacheStore(str(tmp_path / "cache.db")), embed, threshold=0.9)
    assert cache.get("reset my password") is None


def test_semantic_cache_hit(tmp_path):
    cache = SemanticCache(SQLiteCacheStore(str(tmp_path / "cache.db")), embed, threshold=0.9)
    cache.set("reset my password", "Go to settings.")
    assert cache.get("reset my password") == "Go to settings."


def test_semantic_cache_uses_provided_vec(tmp_path):
    calls = []

    def counting_embed(text):
        calls.append(text)
        return embed(text)

    cache = SemanticCache(SQLiteCacheStore(str(tmp_path / "cache.db")), counting_embed, threshold=0.9)
    cache.set("reset my password", "Go to settings.")
    calls.clear()

    cache.get("reset my password", vec=[1.0, 0.0, 0.0])
    assert len(calls) == 0  # embedder not called when vec is provided


# ── Monitor ───────────────────────────────────────────────────────────────────

def test_monitor_unknown_field_raises():
    monitor = Monitor(FakeMonitorStore(), embed)
    with pytest.raises(ValueError, match="Unknown fields"):
        monitor.log("query", latency_ms=10.0, user_id="abc")


def test_monitor_wrong_type_raises():
    monitor = Monitor(FakeMonitorStore({"user_id": str}), embed)
    with pytest.raises(TypeError, match="user_id"):
        monitor.log("query", latency_ms=10.0, user_id=123)


def test_monitor_logs_with_valid_kwargs():
    store = FakeMonitorStore({"user_id": str})
    monitor = Monitor(store, embed)
    monitor.log("reset my password", latency_ms=10.0, user_id="abc")
    assert store.logged[0]["user_id"] == "abc"


def test_monitor_calculate_timing():
    start = time.time()
    time.sleep(0.01)
    assert Monitor.calculate_timing(start) >= 10.0


def test_monitor_provided_vec_skips_embed():
    calls = []

    def counting_embed(text):
        calls.append(text)
        return embed(text)

    store = FakeMonitorStore()
    monitor = Monitor(store, counting_embed)
    monitor.log("query", latency_ms=10.0, vec=[1.0, 0.0, 0.0])
    assert len(calls) == 0


# ── Middleware ────────────────────────────────────────────────────────────────

def test_middleware_calls_fn():
    @Middleware().track
    def retrieve(query):
        return f"result:{query}"

    assert retrieve("hello") == "result:hello"


def test_middleware_cache_miss_calls_fn(tmp_path):
    cache = SemanticCache(SQLiteCacheStore(str(tmp_path / "cache.db")), embed, threshold=0.9)

    @Middleware(cache=cache, embedder=embed).track
    def retrieve(query):
        return "live response"

    assert retrieve("reset my password") == "live response"


def test_middleware_cache_hit_skips_fn(tmp_path):
    cache = SemanticCache(SQLiteCacheStore(str(tmp_path / "cache.db")), embed, threshold=0.9)
    cache.set("reset my password", "Cached response.")

    called = []

    @Middleware(cache=cache, embedder=embed).track
    def retrieve(query):
        called.append(query)
        return "live response"

    assert retrieve("reset my password") == "Cached response."
    assert len(called) == 0


def test_middleware_monitor_kwargs():
    store = FakeMonitorStore({"user_id": str})
    monitor = Monitor(store, embed)
    mw = Middleware(monitor=monitor, embedder=embed)

    @mw.track
    def retrieve(query):
        return "result"

    retrieve("reset my password", _monitor_kwargs={"user_id": "abc"})
    mw.shutdown()  # flush thread pool before asserting
    assert store.logged[0]["user_id"] == "abc"


def test_middleware_no_monitor_no_executor():
    assert Middleware()._executor is None


def test_middleware_shutdown_no_error():
    mw = Middleware(monitor=Monitor(FakeMonitorStore(), embed))
    mw.shutdown()
