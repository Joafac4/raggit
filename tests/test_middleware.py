import time
from typing import Dict, List

import pytest

from raggit.middleware import (
    Middleware,
    Monitor,
    MonitorStore,
    RetrievalHandle,
    SemanticCache,
    SQLiteCacheStore,
    SQLiteClusterFeedbackStore,
    SQLiteClusterStore,
    SQLiteEventFeedbackStore,
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
    keeps_events = True

    def __init__(self, schema: Dict[str, type] = None):
        self._schema = schema or {}
        self.logged = []
        self._cluster_counter = 0

    def get_schema(self) -> Dict[str, type]:
        return self._schema

    def assign_cluster(self, vec, threshold, query):
        self._cluster_counter += 1
        return f"cluster-{self._cluster_counter}"

    def log(self, query, vec, latency_ms, threshold, cache_hit=False,
            cluster_id=None, event_id=None, **kwargs):
        self.logged.append({
            "query": query, "vec": vec, "latency_ms": latency_ms,
            "cache_hit": cache_hit, "cluster_id": cluster_id,
            "event_id": event_id, **kwargs,
        })


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


def test_monitor_store_records_retrieval_metadata(tmp_path):
    store = SQLiteMonitorStore(str(tmp_path / "monitor.db"))
    store.log("reset my password", vec=[1.0, 0.0, 0.0], latency_ms=10.0, threshold=0.9,
              retrieval_score=0.8, retrieved_doc_ids=["doc1"])
    events = store.get_events()
    assert len(events) == 1
    assert events[0].retrieval_score == 0.8
    assert events[0].retrieved_doc_ids == ["doc1"]


def test_monitor_store_get_events(tmp_path):
    store = SQLiteMonitorStore(str(tmp_path / "monitor.db"))
    store.log("reset my password", vec=[1.0, 0.0, 0.0], latency_ms=10.0, threshold=0.9,
              retrieved_doc_ids=["doc1", "doc2"])
    store.log("activate account", vec=[0.0, 1.0, 0.0], latency_ms=10.0, threshold=0.9)
    events = store.get_events()
    assert len(events) == 2


def test_monitor_store_get_events_filter_retrieved_docs(tmp_path):
    store = SQLiteMonitorStore(str(tmp_path / "monitor.db"))
    store.log("reset my password", vec=[1.0, 0.0, 0.0], latency_ms=10.0, threshold=0.9,
              retrieved_doc_ids=["doc1"])
    store.log("activate account", vec=[0.0, 1.0, 0.0], latency_ms=10.0, threshold=0.9)
    events = store.get_events(has_retrieved_docs=True)
    assert len(events) == 1
    assert events[0].retrieved_doc_ids == ["doc1"]


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


# ── track_with_handle ─────────────────────────────────────────────────────────

def test_track_with_handle_returns_handle(tmp_path):
    monitor = Monitor(SQLiteMonitorStore(str(tmp_path / "m.db")), embed)
    mw = Middleware(monitor=monitor, embedder=embed)

    @mw.track_with_handle
    def retrieve(query):
        return f"answer:{query}"

    handle = retrieve("reset my password")
    mw.shutdown()
    assert isinstance(handle, RetrievalHandle)
    assert handle.answer == "answer:reset my password"
    assert handle.cluster_id is not None
    assert handle.event_id is not None  # MonitorStore keeps events


def test_track_with_handle_event_id_none_for_clusterstore(tmp_path):
    monitor = Monitor(SQLiteClusterStore(str(tmp_path / "c.db")), embed)
    mw = Middleware(monitor=monitor, embedder=embed)

    @mw.track_with_handle
    def retrieve(query):
        return "answer"

    handle = retrieve("reset my password")
    mw.shutdown()
    assert handle.cluster_id is not None
    assert handle.event_id is None


def test_track_with_handle_requires_monitor():
    mw = Middleware(embedder=embed)
    with pytest.raises(ValueError, match="requires a monitor"):
        mw.track_with_handle(lambda q: q)


def test_track_with_handle_requires_embedder(tmp_path):
    monitor = Monitor(SQLiteMonitorStore(str(tmp_path / "m.db")), embed)
    mw = Middleware(monitor=monitor)  # no embedder
    with pytest.raises(ValueError, match="requires an embedder"):
        mw.track_with_handle(lambda q: q)


def test_track_with_handle_cache_hit(tmp_path):
    cache = SemanticCache(SQLiteCacheStore(str(tmp_path / "cache.db")), embed, threshold=0.9)
    cache.set("reset my password", "Cached response.")
    monitor = Monitor(SQLiteMonitorStore(str(tmp_path / "m.db")), embed)
    mw = Middleware(monitor=monitor, cache=cache, embedder=embed)

    called = []

    @mw.track_with_handle
    def retrieve(query):
        called.append(query)
        return "live"

    handle = retrieve("reset my password")
    mw.shutdown()
    assert handle.answer == "Cached response."
    assert handle.cluster_id is not None
    assert len(called) == 0


# ── Monitor.record_feedback validation ────────────────────────────────────────

def _make_monitor_with_event_feedback(tmp_path, db="m.db"):
    path = str(tmp_path / db)
    return Monitor(
        SQLiteMonitorStore(path),
        embed,
        feedback_store=SQLiteEventFeedbackStore(path),
    )


def _make_monitor_with_cluster_feedback(tmp_path, db="c.db"):
    path = str(tmp_path / db)
    return Monitor(
        SQLiteClusterStore(path),
        embed,
        feedback_store=SQLiteClusterFeedbackStore(path),
    )


def test_record_feedback_requires_signal(tmp_path):
    monitor = _make_monitor_with_event_feedback(tmp_path)
    handle = RetrievalHandle(answer="x", cluster_id="c1", event_id="e1")
    with pytest.raises(ValueError, match="at least one"):
        monitor.record_feedback(handle)


def test_record_feedback_without_feedback_store_raises(tmp_path):
    monitor = Monitor(SQLiteMonitorStore(str(tmp_path / "m.db")), embed)  # no feedback_store
    handle = RetrievalHandle(answer="x", cluster_id="c1", event_id="e1")
    with pytest.raises(ValueError, match="requires a feedback_store"):
        monitor.record_feedback(handle, accepted=True)


def test_acceptance_rate_without_feedback_store_raises(tmp_path):
    monitor = Monitor(SQLiteMonitorStore(str(tmp_path / "m.db")), embed)
    with pytest.raises(ValueError, match="requires a feedback_store"):
        monitor.acceptance_rate_per_cluster()


def test_record_feedback_accepts_thumb_only(tmp_path):
    monitor = _make_monitor_with_event_feedback(tmp_path)
    mw = Middleware(monitor=monitor, embedder=embed)

    @mw.track_with_handle
    def retrieve(query):
        return "answer"

    handle = retrieve("reset my password")
    mw.shutdown()
    monitor.record_feedback(handle, accepted=True)


def test_record_feedback_accepts_score_only(tmp_path):
    monitor = _make_monitor_with_event_feedback(tmp_path)
    mw = Middleware(monitor=monitor, embedder=embed)

    @mw.track_with_handle
    def retrieve(query):
        return "answer"

    handle = retrieve("reset my password")
    mw.shutdown()
    monitor.record_feedback(handle, score=0.7)


# ── SQLiteEventFeedbackStore (paired with SQLiteMonitorStore) ────────────────

def test_event_feedback_store_persists(tmp_path):
    monitor = _make_monitor_with_event_feedback(tmp_path)
    mw = Middleware(monitor=monitor, embedder=embed)

    @mw.track_with_handle
    def retrieve(query):
        return "answer"

    h1 = retrieve("reset my password")
    h2 = retrieve("activate account")
    mw.shutdown()

    monitor.record_feedback(h1, accepted=True, score=0.9, comment="great")
    monitor.record_feedback(h2, accepted=False, score=0.2)

    rates = monitor.acceptance_rate_per_cluster()
    assert len(rates) == 2
    by_cluster = {c.cluster_id: rate for c, rate in rates}
    assert by_cluster[h1.cluster_id] == 1.0
    assert by_cluster[h2.cluster_id] == 0.0

    avgs = monitor.avg_score_per_cluster()
    by_cluster_score = {c.cluster_id: s for c, s in avgs}
    assert abs(by_cluster_score[h1.cluster_id] - 0.9) < 1e-9
    assert abs(by_cluster_score[h2.cluster_id] - 0.2) < 1e-9


def test_event_feedback_store_rejects_handle_without_event_id(tmp_path):
    store = SQLiteEventFeedbackStore(str(tmp_path / "m.db"))
    fake_handle = RetrievalHandle(answer="x", cluster_id="c1", event_id=None)
    with pytest.raises(ValueError, match="event_id is None"):
        store.record_feedback(fake_handle, accepted=True)


def test_event_feedback_store_init_idempotent(tmp_path):
    """Multiple instantiations on the same path must not error, regardless of order with paired MonitorStore."""
    path = str(tmp_path / "m.db")
    SQLiteEventFeedbackStore(path)            # creates clusters + events + feedback
    SQLiteMonitorStore(path)                  # idempotent on clusters + events
    SQLiteEventFeedbackStore(path)            # idempotent on all three


# ── SQLiteClusterFeedbackStore (paired with SQLiteClusterStore) ───────────────

def test_cluster_feedback_store_counters(tmp_path):
    monitor = _make_monitor_with_cluster_feedback(tmp_path)
    mw = Middleware(monitor=monitor, embedder=embed)

    @mw.track_with_handle
    def retrieve(query):
        return "answer"

    h1 = retrieve("reset my password")
    retrieve("reset my password again")  # same cluster
    mw.shutdown()

    monitor.record_feedback(h1, accepted=True, score=1.0)
    monitor.record_feedback(h1, accepted=True, score=0.5)
    monitor.record_feedback(h1, accepted=False)
    monitor.record_feedback(h1, score=0.3)  # score-only

    rates = monitor.acceptance_rate_per_cluster()
    assert len(rates) == 1
    cluster, rate = rates[0]
    # 2 accepted out of 3 thumb votes (the score-only vote doesn't count)
    assert abs(rate - (2 / 3)) < 1e-9

    avgs = monitor.avg_score_per_cluster()
    cluster, avg = avgs[0]
    # (1.0 + 0.5 + 0.3) / 3
    assert abs(avg - 0.6) < 1e-9


def test_cluster_feedback_store_drops_comment(tmp_path):
    """Comments are silently dropped in ClusterFeedbackStore (no row to attach to)."""
    monitor = _make_monitor_with_cluster_feedback(tmp_path)
    mw = Middleware(monitor=monitor, embedder=embed)

    @mw.track_with_handle
    def retrieve(query):
        return "answer"

    handle = retrieve("reset my password")
    mw.shutdown()
    # Should not raise, just silently ignore the comment
    monitor.record_feedback(handle, accepted=True, comment="ignored")


def test_cluster_feedback_store_init_idempotent(tmp_path):
    """Order-agnostic and idempotent across both stores sharing a path."""
    path = str(tmp_path / "c.db")
    SQLiteClusterFeedbackStore(path)          # creates clusters with counter columns
    SQLiteClusterStore(path)                  # idempotent on clusters table
    SQLiteClusterFeedbackStore(path)          # idempotent on counter columns


# ── Idempotent migrations ─────────────────────────────────────────────────────

def test_cluster_store_migration_idempotent(tmp_path):
    """Re-instantiating SQLiteClusterStore on the same path must not error."""
    path = str(tmp_path / "c.db")
    SQLiteClusterStore(path)
    SQLiteClusterStore(path)  # should be a no-op due to _column_exists check
    SQLiteClusterStore(path)


def test_monitor_store_migration_idempotent(tmp_path):
    """Re-instantiating SQLiteMonitorStore on the same path must not error."""
    path = str(tmp_path / "m.db")
    SQLiteMonitorStore(path)
    SQLiteMonitorStore(path)
    SQLiteMonitorStore(path)


# ── Custom stores without assign_cluster ──────────────────────────────────────

class MinimalCustomStore(MonitorStore):
    """Implements only the original required methods. No assign_cluster, no feedback."""
    def get_schema(self):
        return {}

    def log(self, query, vec, latency_ms, threshold, cache_hit=False,
            cluster_id=None, event_id=None, **kwargs):
        pass


def test_minimal_custom_store_works_with_track():
    """Stores without assign_cluster still work with @mw.track."""
    monitor = Monitor(MinimalCustomStore(), embed)
    mw = Middleware(monitor=monitor, embedder=embed)

    @mw.track
    def retrieve(query):
        return "ok"

    assert retrieve("anything") == "ok"
    mw.shutdown()


def test_minimal_custom_store_track_with_handle_raises():
    """track_with_handle raises a clear error when the store doesn't implement assign_cluster."""
    monitor = Monitor(MinimalCustomStore(), embed)
    mw = Middleware(monitor=monitor, embedder=embed)

    @mw.track_with_handle
    def retrieve(query):
        return "ok"

    with pytest.raises(NotImplementedError, match="assign_cluster"):
        retrieve("anything")
