from __future__ import annotations

import json
import os
import sqlite3
import uuid
from datetime import datetime
from typing import Dict, List, Optional

from ...metrics import Metrics
from ..models import Cluster
from .base import CacheStore, MonitorStore

_INTERNAL_EVENT_COLS = frozenset(
    {"event_id", "cluster_id", "query_text", "latency_ms", "timestamp", "cache_hit"}
)

_SQLITE_TO_PYTHON: Dict[str, type] = {
    "TEXT": str, "VARCHAR": str, "CHAR": str,
    "REAL": float, "FLOAT": float, "DOUBLE": float,
    "INTEGER": int, "INT": int, "BIGINT": int,
    "BOOLEAN": bool,
    "BLOB": bytes,
    "NUMERIC": float,
}


def _ensure_dir(path: str) -> None:
    dir_path = os.path.dirname(path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)


def _row_to_cluster(row) -> Cluster:
    cluster_id, rep_vec, rep_query, count, created_at, last_seen = row
    return Cluster(
        cluster_id=cluster_id,
        representative_vec=json.loads(rep_vec),
        representative_query=rep_query,
        count=count,
        created_at=datetime.fromisoformat(created_at),
        last_seen=datetime.fromisoformat(last_seen),
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


class SQLiteMonitorStore(MonitorStore):
    """
    SQLite MonitorStore with full per-query history.
    Tables: clusters, events.

    To store extra fields per event, add columns to the events table first:
        ALTER TABLE events ADD COLUMN user_id TEXT;
    Then pass them as kwargs to monitor.log():
        monitor.log("query", latency_ms=100, user_id="abc")
    """

    def __init__(self, path: str = ".raggit/middleware.db"):
        self.path = path
        _ensure_dir(path)
        with sqlite3.connect(self.path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS clusters (
                    cluster_id TEXT PRIMARY KEY,
                    representative_vec TEXT NOT NULL,
                    representative_query TEXT NOT NULL,
                    count INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL,
                    last_seen TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    event_id TEXT PRIMARY KEY,
                    cluster_id TEXT NOT NULL,
                    query_text TEXT NOT NULL,
                    latency_ms REAL,
                    timestamp TEXT NOT NULL,
                    cache_hit INTEGER DEFAULT 0,
                    FOREIGN KEY (cluster_id) REFERENCES clusters(cluster_id)
                )
            """)

    def get_schema(self) -> Dict[str, type]:
        with sqlite3.connect(self.path) as conn:
            rows = conn.execute("PRAGMA table_info(events)").fetchall()
        return {
            row[1]: _SQLITE_TO_PYTHON.get(row[2].split("(")[0].upper(), str)
            for row in rows
            if row[1] not in _INTERNAL_EVENT_COLS
        }

    def log(
        self,
        query: str,
        vec: List[float],
        latency_ms: float,
        threshold: float,
        cache_hit: bool = False,
        **kwargs,
    ) -> None:
        now = datetime.now().isoformat()
        with sqlite3.connect(self.path) as conn:
            cluster_id = _best_match(conn, "clusters", vec, threshold)
            if cluster_id:
                conn.execute(
                    "UPDATE clusters SET count = count + 1, last_seen = ? WHERE cluster_id = ?",
                    (now, cluster_id),
                )
            else:
                cluster_id = str(uuid.uuid4())
                conn.execute(
                    """
                    INSERT INTO clusters
                        (cluster_id, representative_vec, representative_query, count, created_at, last_seen)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (cluster_id, json.dumps(vec), query, 1, now, now),
                )

            base_cols = ["event_id", "cluster_id", "query_text", "latency_ms", "timestamp", "cache_hit"]
            base_vals = [str(uuid.uuid4()), cluster_id, query, latency_ms, now, int(cache_hit)]
            extra_cols = list(kwargs.keys())
            all_cols = base_cols + extra_cols
            placeholders = ",".join(["?"] * len(all_cols))
            conn.execute(
                f"INSERT INTO events ({','.join(all_cols)}) VALUES ({placeholders})",
                base_vals + list(kwargs.values()),
            )

    def get_clusters(
        self,
        top: Optional[int] = None,
        since: Optional[datetime] = None,
        last_seen_before: Optional[datetime] = None,
    ) -> List[Cluster]:
        sql = "SELECT * FROM clusters WHERE 1=1"
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
    """

    def __init__(self, path: str = ".raggit/middleware.db"):
        self.path = path
        _ensure_dir(path)
        with sqlite3.connect(self.path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS clusters (
                    cluster_id TEXT PRIMARY KEY,
                    representative_vec TEXT NOT NULL,
                    representative_query TEXT NOT NULL,
                    count INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL,
                    last_seen TEXT NOT NULL
                )
            """)

    def get_schema(self) -> Dict[str, type]:
        return {}

    def log(
        self,
        query: str,
        vec: List[float],
        latency_ms: float,
        threshold: float,
        cache_hit: bool = False,
        **kwargs,
    ) -> None:
        now = datetime.now().isoformat()
        with sqlite3.connect(self.path) as conn:
            cluster_id = _best_match(conn, "clusters", vec, threshold)
            if cluster_id:
                conn.execute(
                    "UPDATE clusters SET count = count + 1, last_seen = ? WHERE cluster_id = ?",
                    (now, cluster_id),
                )
            else:
                conn.execute(
                    """
                    INSERT INTO clusters
                        (cluster_id, representative_vec, representative_query, count, created_at, last_seen)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (str(uuid.uuid4()), json.dumps(vec), query, 1, now, now),
                )

    def get_clusters(
        self,
        top: Optional[int] = None,
        since: Optional[datetime] = None,
        last_seen_before: Optional[datetime] = None,
    ) -> List[Cluster]:
        sql = "SELECT * FROM clusters WHERE 1=1"
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
