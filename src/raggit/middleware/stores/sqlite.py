from __future__ import annotations

import json
import os
import sqlite3
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from ...metrics import Metrics
from ..models import Cluster, Event, RetrievalHandle
from .base import CacheStore, FeedbackStore, MonitorStore

_INTERNAL_EVENT_COLS = frozenset({
    "event_id", "cluster_id", "query_text", "latency_ms", "timestamp", "cache_hit",
    "retrieval_score", "retrieved_doc_ids",
})

_SQLITE_TO_PYTHON: Dict[str, type] = {
    "TEXT": str, "VARCHAR": str, "CHAR": str,
    "REAL": float, "FLOAT": float, "DOUBLE": float,
    "INTEGER": int, "INT": int, "BIGINT": int,
    "BOOLEAN": bool,
    "BLOB": bytes,
    "NUMERIC": float,
}

_CLUSTERS_DDL = """
    CREATE TABLE IF NOT EXISTS clusters (
        cluster_id TEXT PRIMARY KEY,
        representative_vec TEXT NOT NULL,
        representative_query TEXT NOT NULL,
        count INTEGER DEFAULT 0,
        created_at TEXT NOT NULL,
        last_seen TEXT NOT NULL
    )
"""

_EVENTS_DDL = """
    CREATE TABLE IF NOT EXISTS events (
        event_id TEXT PRIMARY KEY,
        cluster_id TEXT NOT NULL,
        query_text TEXT NOT NULL,
        latency_ms REAL,
        timestamp TEXT NOT NULL,
        cache_hit INTEGER DEFAULT 0,
        retrieval_score REAL,
        retrieved_doc_ids TEXT,
        FOREIGN KEY (cluster_id) REFERENCES clusters(cluster_id)
    )
"""


def _ensure_dir(path: str) -> None:
    dir_path = os.path.dirname(path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)


def _row_to_cluster(row) -> Cluster:
    cluster_id, rep_vec, rep_query, count, created_at, last_seen = row[:6]
    return Cluster(
        cluster_id=cluster_id,
        representative_vec=json.loads(rep_vec),
        representative_query=rep_query,
        count=count,
        created_at=datetime.fromisoformat(created_at),
        last_seen=datetime.fromisoformat(last_seen),
    )


def _row_to_event(row) -> Event:
    event_id, cluster_id, query_text, latency_ms, timestamp, cache_hit, \
        retrieval_score, retrieved_doc_ids = row
    return Event(
        event_id=event_id,
        cluster_id=cluster_id,
        query_text=query_text,
        latency_ms=latency_ms,
        timestamp=datetime.fromisoformat(timestamp),
        cache_hit=bool(cache_hit),
        retrieval_score=retrieval_score,
        retrieved_doc_ids=json.loads(retrieved_doc_ids) if retrieved_doc_ids else None,
    )


def _best_match(conn, table: str, vec: List[float], threshold: float) -> Optional[str]:
    """Return cluster_id of most similar vector above threshold, or None."""
    rows = conn.execute(f"SELECT cluster_id, representative_vec FROM {table}").fetchall()
    best_id, best_score = None, -1.0
    for cluster_id, vec_json in rows:
        score = Metrics.cosine_similarity(vec, json.loads(vec_json))
        if score > best_score:
            best_score = score
            best_id = cluster_id
    return best_id if best_score >= threshold else None


def _assign_or_create_cluster(
    conn, vec: List[float], threshold: float, query: str, now: str
) -> str:
    """Find or create a cluster, bumping count + last_seen. Returns cluster_id."""
    cluster_id = _best_match(conn, "clusters", vec, threshold)
    if cluster_id:
        conn.execute(
            "UPDATE clusters SET count = count + 1, last_seen = ? WHERE cluster_id = ?",
            (now, cluster_id),
        )
    else:
        cluster_id = str(uuid.uuid4())
        conn.execute(
            """INSERT INTO clusters
                (cluster_id, representative_vec, representative_query, count,
                 created_at, last_seen)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (cluster_id, json.dumps(vec), query, 1, now, now),
        )
    return cluster_id


def _column_exists(conn, table: str, column: str) -> bool:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return any(row[1] == column for row in rows)


# ── MonitorStore implementations ──────────────────────────────────────────────

class SQLiteMonitorStore(MonitorStore):
    """
    SQLite MonitorStore with full per-query history.
    Tables: clusters, events.

    Pair with SQLiteEventFeedbackStore for per-event feedback.

    To store extra fields per event, add columns to the events table first:
        ALTER TABLE events ADD COLUMN user_id TEXT;
    Then pass them as kwargs to monitor.log():
        monitor.log("query", latency_ms=100, user_id="abc")
    """

    keeps_events = True

    def __init__(self, path: str = ".raggit/middleware.db"):
        self.path = path
        _ensure_dir(path)
        with sqlite3.connect(self.path) as conn:
            conn.execute(_CLUSTERS_DDL)
            conn.execute(_EVENTS_DDL)

    def get_schema(self) -> Dict[str, type]:
        with sqlite3.connect(self.path) as conn:
            rows = conn.execute("PRAGMA table_info(events)").fetchall()
        return {
            row[1]: _SQLITE_TO_PYTHON.get(row[2].split("(")[0].upper(), str)
            for row in rows
            if row[1] not in _INTERNAL_EVENT_COLS
        }

    def assign_cluster(self, vec: List[float], threshold: float, query: str) -> str:
        now = datetime.now().isoformat()
        with sqlite3.connect(self.path) as conn:
            return _assign_or_create_cluster(conn, vec, threshold, query, now)

    def log(
        self,
        query: str,
        vec: List[float],
        latency_ms: float,
        threshold: float,
        cache_hit: bool = False,
        cluster_id: Optional[str] = None,
        event_id: Optional[str] = None,
        **kwargs,
    ) -> None:
        retrieval_score = kwargs.pop("retrieval_score", None)
        retrieved_doc_ids = kwargs.pop("retrieved_doc_ids", None)

        now = datetime.now().isoformat()
        with sqlite3.connect(self.path) as conn:
            if cluster_id is None:
                cluster_id = _assign_or_create_cluster(conn, vec, threshold, query, now)
            # else: caller (track_with_handle) already ran assign_cluster and bumped count

            base_cols = [
                "event_id", "cluster_id", "query_text", "latency_ms", "timestamp",
                "cache_hit", "retrieval_score", "retrieved_doc_ids",
            ]
            base_vals = [
                event_id or str(uuid.uuid4()), cluster_id, query, latency_ms, now,
                int(cache_hit),
                retrieval_score,
                json.dumps(retrieved_doc_ids) if retrieved_doc_ids is not None else None,
            ]
            extra_cols = list(kwargs.keys())
            all_cols = base_cols + extra_cols
            placeholders = ",".join(["?"] * len(all_cols))
            conn.execute(
                f"INSERT INTO events ({','.join(all_cols)}) VALUES ({placeholders})",
                base_vals + list(kwargs.values()),
            )

    def get_events(
        self,
        cluster_id: Optional[str] = None,
        since: Optional[datetime] = None,
        has_retrieved_docs: Optional[bool] = None,
    ) -> List[Event]:
        sql = """SELECT event_id, cluster_id, query_text, latency_ms, timestamp,
                        cache_hit, retrieval_score, retrieved_doc_ids
                 FROM events WHERE 1=1"""
        params: list = []
        if cluster_id is not None:
            sql += " AND cluster_id = ?"
            params.append(cluster_id)
        if since is not None:
            sql += " AND timestamp >= ?"
            params.append(since.isoformat())
        if has_retrieved_docs is True:
            sql += " AND retrieved_doc_ids IS NOT NULL"
        elif has_retrieved_docs is False:
            sql += " AND retrieved_doc_ids IS NULL"
        sql += " ORDER BY timestamp DESC"
        with sqlite3.connect(self.path) as conn:
            rows = conn.execute(sql, params).fetchall()
        return [_row_to_event(row) for row in rows]

    def get_clusters(
        self,
        top: Optional[int] = None,
        since: Optional[datetime] = None,
        last_seen_before: Optional[datetime] = None,
    ) -> List[Cluster]:
        sql = """SELECT cluster_id, representative_vec, representative_query,
                        count, created_at, last_seen
                 FROM clusters WHERE 1=1"""
        params: list = []
        if since is not None:
            sql += " AND created_at >= ?"
            params.append(since.isoformat())
        if last_seen_before is not None:
            sql += " AND last_seen <= ?"
            params.append(last_seen_before.isoformat())
        sql += " ORDER BY count DESC"
        if top is not None:
            sql += " LIMIT ?"
            params.append(top)
        with sqlite3.connect(self.path) as conn:
            rows = conn.execute(sql, params).fetchall()
        return [_row_to_cluster(row) for row in rows]

    def stats(self) -> Dict:
        with sqlite3.connect(self.path) as conn:
            total_events = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
            unique_clusters = conn.execute("SELECT COUNT(*) FROM clusters").fetchone()[0]
        return {
            "total_events": total_events,
            "unique_clusters": unique_clusters,
            "top_clusters": self.get_clusters(top=5),
        }


class SQLiteClusterStore(MonitorStore):
    """
    SQLite MonitorStore — aggregate only, no per-query history.
    Table: clusters only (count + last_seen updated in place).

    Pair with SQLiteClusterFeedbackStore for counter-based feedback.
    """

    # keeps_events stays False (default from ABC)

    def __init__(self, path: str = ".raggit/middleware.db"):
        self.path = path
        _ensure_dir(path)
        with sqlite3.connect(self.path) as conn:
            conn.execute(_CLUSTERS_DDL)

    def get_schema(self) -> Dict[str, type]:
        return {}

    def assign_cluster(self, vec: List[float], threshold: float, query: str) -> str:
        now = datetime.now().isoformat()
        with sqlite3.connect(self.path) as conn:
            return _assign_or_create_cluster(conn, vec, threshold, query, now)

    def log(
        self,
        query: str,
        vec: List[float],
        latency_ms: float,
        threshold: float,
        cache_hit: bool = False,
        cluster_id: Optional[str] = None,
        event_id: Optional[str] = None,
        **kwargs,
    ) -> None:
        if cluster_id is not None:
            return  # caller already ran assign_cluster; no event row to write
        now = datetime.now().isoformat()
        with sqlite3.connect(self.path) as conn:
            _assign_or_create_cluster(conn, vec, threshold, query, now)

    def get_clusters(
        self,
        top: Optional[int] = None,
        since: Optional[datetime] = None,
        last_seen_before: Optional[datetime] = None,
    ) -> List[Cluster]:
        sql = """SELECT cluster_id, representative_vec, representative_query,
                        count, created_at, last_seen
                 FROM clusters WHERE 1=1"""
        params: list = []
        if since is not None:
            sql += " AND created_at >= ?"
            params.append(since.isoformat())
        if last_seen_before is not None:
            sql += " AND last_seen <= ?"
            params.append(last_seen_before.isoformat())
        sql += " ORDER BY count DESC"
        if top is not None:
            sql += " LIMIT ?"
            params.append(top)
        with sqlite3.connect(self.path) as conn:
            rows = conn.execute(sql, params).fetchall()
        return [_row_to_cluster(row) for row in rows]

    def stats(self) -> Dict:
        with sqlite3.connect(self.path) as conn:
            rows = conn.execute("SELECT count FROM clusters").fetchall()
        total = sum(row[0] for row in rows)
        return {
            "total_events": total,
            "unique_clusters": len(rows),
            "top_clusters": self.get_clusters(top=5),
        }


# ── FeedbackStore implementations ─────────────────────────────────────────────

class SQLiteEventFeedbackStore(FeedbackStore):
    """
    Per-event feedback. Pairs with SQLiteMonitorStore.
    Same DB file. Creates a separate `feedback` table FK'd to events.

    Idempotent on init — safe to instantiate before or after the paired
    MonitorStore.
    """

    def __init__(self, path: str = ".raggit/middleware.db"):
        self.path = path
        _ensure_dir(path)
        with sqlite3.connect(self.path) as conn:
            # Ensure paired tables exist (no-op if SQLiteMonitorStore got there first).
            conn.execute(_CLUSTERS_DDL)
            conn.execute(_EVENTS_DDL)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    feedback_id TEXT PRIMARY KEY,
                    event_id TEXT NOT NULL,
                    accepted INTEGER,
                    score REAL,
                    comment TEXT,
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY (event_id) REFERENCES events(event_id)
                )
            """)

    def record_feedback(
        self,
        handle: RetrievalHandle,
        accepted: Optional[bool] = None,
        score: Optional[float] = None,
        comment: Optional[str] = None,
    ) -> None:
        if handle.event_id is None:
            raise ValueError(
                "RetrievalHandle.event_id is None — feedback cannot be attached. "
                "This usually means the handle came from a MonitorStore that does "
                "not keep per-event history (e.g. SQLiteClusterStore). Pair that "
                "store with SQLiteClusterFeedbackStore instead."
            )
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                """INSERT INTO feedback
                    (feedback_id, event_id, accepted, score, comment, timestamp)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    str(uuid.uuid4()),
                    handle.event_id,
                    None if accepted is None else int(accepted),
                    score,
                    comment,
                    datetime.now().isoformat(),
                ),
            )

    def acceptance_rate_per_cluster(
        self,
        top: Optional[int] = None,
    ) -> List[Tuple[Cluster, float]]:
        sql = """
            SELECT c.cluster_id, c.representative_vec, c.representative_query,
                   c.count, c.created_at, c.last_seen,
                   SUM(CASE WHEN f.accepted = 1 THEN 1 ELSE 0 END) * 1.0
                       / SUM(CASE WHEN f.accepted IS NOT NULL THEN 1 ELSE 0 END) AS rate
            FROM clusters c
            JOIN events e   ON e.cluster_id = c.cluster_id
            JOIN feedback f ON f.event_id   = e.event_id
            GROUP BY c.cluster_id
            HAVING SUM(CASE WHEN f.accepted IS NOT NULL THEN 1 ELSE 0 END) > 0
            ORDER BY c.count DESC
        """
        if top is not None:
            sql += " LIMIT ?"
        with sqlite3.connect(self.path) as conn:
            rows = conn.execute(sql, (top,) if top is not None else ()).fetchall()
        return [(_row_to_cluster(row[:6]), row[6]) for row in rows]

    def avg_score_per_cluster(
        self,
        top: Optional[int] = None,
    ) -> List[Tuple[Cluster, float]]:
        sql = """
            SELECT c.cluster_id, c.representative_vec, c.representative_query,
                   c.count, c.created_at, c.last_seen,
                   AVG(f.score) AS avg_score
            FROM clusters c
            JOIN events e   ON e.cluster_id = c.cluster_id
            JOIN feedback f ON f.event_id   = e.event_id
            WHERE f.score IS NOT NULL
            GROUP BY c.cluster_id
            ORDER BY c.count DESC
        """
        if top is not None:
            sql += " LIMIT ?"
        with sqlite3.connect(self.path) as conn:
            rows = conn.execute(sql, (top,) if top is not None else ()).fetchall()
        return [(_row_to_cluster(row[:6]), row[6]) for row in rows]


class SQLiteClusterFeedbackStore(FeedbackStore):
    """
    Counter-based feedback. Pairs with SQLiteClusterStore.
    Same DB file. Appends 4 counter columns to the existing `clusters` table —
    no separate feedback table.

    Comments are silently dropped (no row to attach them to). This is
    documented behavior, consistent with the paired store's "aggregates only"
    philosophy.

    Idempotent on init — safe to instantiate before or after the paired
    MonitorStore.
    """

    _FEEDBACK_COLUMNS = {
        "accepted_count": "INTEGER DEFAULT 0",
        "feedback_count": "INTEGER DEFAULT 0",
        "score_sum":      "REAL DEFAULT 0",
        "score_count":    "INTEGER DEFAULT 0",
    }

    def __init__(self, path: str = ".raggit/middleware.db"):
        self.path = path
        _ensure_dir(path)
        with sqlite3.connect(self.path) as conn:
            conn.execute(_CLUSTERS_DDL)
            for col, decl in self._FEEDBACK_COLUMNS.items():
                if not _column_exists(conn, "clusters", col):
                    conn.execute(f"ALTER TABLE clusters ADD COLUMN {col} {decl}")

    def record_feedback(
        self,
        handle: RetrievalHandle,
        accepted: Optional[bool] = None,
        score: Optional[float] = None,
        comment: Optional[str] = None,
    ) -> None:
        # comment is intentionally dropped — no per-event row to attach it to.
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                """UPDATE clusters
                   SET feedback_count = feedback_count
                       + (CASE WHEN ? IS NOT NULL THEN 1 ELSE 0 END),
                       accepted_count = accepted_count
                       + (CASE WHEN ? = 1 THEN 1 ELSE 0 END),
                       score_count    = score_count
                       + (CASE WHEN ? IS NOT NULL THEN 1 ELSE 0 END),
                       score_sum      = score_sum + COALESCE(?, 0)
                   WHERE cluster_id = ?""",
                (
                    None if accepted is None else int(accepted),
                    None if accepted is None else int(accepted),
                    score,
                    score,
                    handle.cluster_id,
                ),
            )

    def acceptance_rate_per_cluster(
        self,
        top: Optional[int] = None,
    ) -> List[Tuple[Cluster, float]]:
        sql = """SELECT cluster_id, representative_vec, representative_query,
                        count, created_at, last_seen,
                        accepted_count * 1.0 / feedback_count AS rate
                 FROM clusters
                 WHERE feedback_count > 0
                 ORDER BY count DESC"""
        if top is not None:
            sql += " LIMIT ?"
        with sqlite3.connect(self.path) as conn:
            rows = conn.execute(sql, (top,) if top is not None else ()).fetchall()
        return [(_row_to_cluster(row[:6]), row[6]) for row in rows]

    def avg_score_per_cluster(
        self,
        top: Optional[int] = None,
    ) -> List[Tuple[Cluster, float]]:
        sql = """SELECT cluster_id, representative_vec, representative_query,
                        count, created_at, last_seen,
                        score_sum / score_count AS avg_score
                 FROM clusters
                 WHERE score_count > 0
                 ORDER BY count DESC"""
        if top is not None:
            sql += " LIMIT ?"
        with sqlite3.connect(self.path) as conn:
            rows = conn.execute(sql, (top,) if top is not None else ()).fetchall()
        return [(_row_to_cluster(row[:6]), row[6]) for row in rows]


# ── CacheStore ────────────────────────────────────────────────────────────────

class SQLiteCacheStore(CacheStore):
    """
    SQLite CacheStore — fully independent from MonitorStore.
    Table: cache (vec + response together). If query matches above threshold, return response.
    """

    def __init__(self, path: str = ".raggit/middleware.db"):
        self.path = path
        _ensure_dir(path)
        with sqlite3.connect(self.path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    cache_id TEXT PRIMARY KEY,
                    vec TEXT NOT NULL,
                    response TEXT NOT NULL,
                    approved_by TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)

    def get(self, vec: List[float], threshold: float) -> Optional[str]:
        with sqlite3.connect(self.path) as conn:
            rows = conn.execute("SELECT cache_id, vec FROM cache").fetchall()
        best_id, best_score = None, -1.0
        for cache_id, vec_json in rows:
            score = Metrics.cosine_similarity(vec, json.loads(vec_json))
            if score > best_score:
                best_score = score
                best_id = cache_id
        if best_score < threshold or best_id is None:
            return None
        with sqlite3.connect(self.path) as conn:
            row = conn.execute(
                "SELECT response FROM cache WHERE cache_id = ?", (best_id,)
            ).fetchone()
        return row[0] if row else None

    def set(self, vec: List[float], response: str, approved_by: str = "llm") -> None:
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                "INSERT INTO cache (cache_id, vec, response, approved_by, created_at) VALUES (?, ?, ?, ?, ?)",
                (str(uuid.uuid4()), json.dumps(vec), response, approved_by, datetime.now().isoformat()),
            )
