"""
Postgres + pgvector integration tests.

Skipped by default. To run:

    pip install raggit[postgres]
    export RAGGIT_TEST_POSTGRES_DSN="postgresql://user:pass@localhost:5432/raggit_test"
    pytest tests/test_postgres.py

The test database must be empty (or willing to be wiped) — every test drops
the `clusters`, `events`, `feedback`, `cache` tables before running.
"""
import os
from typing import List

import pytest

POSTGRES_DSN = os.environ.get("RAGGIT_TEST_POSTGRES_DSN")

pytestmark = pytest.mark.skipif(
    POSTGRES_DSN is None,
    reason="set RAGGIT_TEST_POSTGRES_DSN to run Postgres integration tests",
)

# Imports kept inside the skip guard so the module imports cleanly when the
# optional dep isn't installed.
if POSTGRES_DSN is not None:
    from raggit.middleware import (
        AutoCachePromoter,
        Middleware,
        Monitor,
        PostgresCacheStore,
        PostgresClusterFeedbackStore,
        PostgresClusterStore,
        PostgresEventFeedbackStore,
        PostgresMonitorStore,
        RetrievalHandle,
        SemanticCache,
    )

DIM = 3


def embed(text: str) -> List[float]:
    if "password" in text.lower():
        return [1.0, 0.0, 0.0]
    if "account" in text.lower():
        return [0.0, 1.0, 0.0]
    return [0.0, 0.0, 1.0]


@pytest.fixture(autouse=True)
def _clean_db():
    """Drop all raggit tables before each test."""
    import psycopg
    with psycopg.connect(POSTGRES_DSN) as conn:
        conn.autocommit = True
        for tbl in ["feedback", "events", "cache", "clusters"]:
            conn.execute(f"DROP TABLE IF EXISTS {tbl} CASCADE")
    yield


# ── Smoke: stores instantiate ────────────────────────────────────────────────

def test_postgres_monitor_store_instantiates():
    store = PostgresMonitorStore(POSTGRES_DSN, dim=DIM)
    store.close()


def test_postgres_cluster_store_instantiates():
    store = PostgresClusterStore(POSTGRES_DSN, dim=DIM)
    store.close()


def test_paired_stores_idempotent_init():
    a = PostgresMonitorStore(POSTGRES_DSN, dim=DIM)
    b = PostgresEventFeedbackStore(POSTGRES_DSN, dim=DIM)
    c = PostgresMonitorStore(POSTGRES_DSN, dim=DIM)  # re-init should be no-op
    a.close(); b.close(); c.close()


# ── End-to-end: monitor + feedback + auto-cache ──────────────────────────────

def test_full_loop_pgvector():
    store = PostgresMonitorStore(POSTGRES_DSN, dim=DIM)
    fb = PostgresEventFeedbackStore(POSTGRES_DSN, dim=DIM)
    cache = SemanticCache(PostgresCacheStore(POSTGRES_DSN, dim=DIM), embed, threshold=0.9)
    promoter = AutoCachePromoter(cache, store, fb, min_count=2, min_acceptance=0.8)
    monitor = Monitor(embed, store=store, feedback_store=fb, auto_promoter=promoter)
    mw = Middleware(monitor=monitor, cache=cache, embedder=embed, auto_promoter=promoter)

    @mw.track_with_handle
    def retrieve(query):
        return "the answer"

    h1 = retrieve("reset my password")
    h2 = retrieve("reset my password")
    mw.shutdown()

    assert h1.cluster_id == h2.cluster_id  # clustered
    assert h1.event_id is not None         # event-keeping store

    monitor.record_feedback(h1, accepted=True)
    monitor.record_feedback(h2, accepted=True)

    assert cache.has_auto_entry(h1.cluster_id)
    assert cache.get("reset my password") == "the answer"

    store.close(); fb.close(); cache.store.close()


def test_cluster_store_pairing_pgvector():
    store = PostgresClusterStore(POSTGRES_DSN, dim=DIM)
    fb = PostgresClusterFeedbackStore(POSTGRES_DSN, dim=DIM)
    monitor = Monitor(embed, store=store, feedback_store=fb)

    monitor.log("reset my password", latency_ms=5.0)
    monitor.log("reset my password", latency_ms=5.0)

    clusters = monitor.popular_queries()
    assert len(clusters) == 1
    assert clusters[0].count == 2

    store.close(); fb.close()


# ── min_count filter ─────────────────────────────────────────────────────────

def test_min_count_pgvector():
    store = PostgresMonitorStore(POSTGRES_DSN, dim=DIM)
    monitor = Monitor(embed, store=store)
    monitor.log("reset my password", latency_ms=5.0)
    for _ in range(3):
        monitor.log("activate account", latency_ms=5.0)
    popular = monitor.popular_queries(min_count=2)
    assert len(popular) == 1
    assert popular[0].representative_query == "activate account"
    store.close()


# ── Latest response stays NULL without promoter ─────────────────────────────

def test_latest_response_null_without_promoter_pgvector():
    store = PostgresMonitorStore(POSTGRES_DSN, dim=DIM)
    monitor = Monitor(embed, store=store)
    mw = Middleware(monitor=monitor, embedder=embed)

    @mw.track_with_handle
    def retrieve(query):
        return "secret"

    h = retrieve("reset my password")
    mw.shutdown()
    cluster = store.get_cluster(h.cluster_id)
    assert cluster is not None
    assert cluster.latest_response is None
    store.close()
