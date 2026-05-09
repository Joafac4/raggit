"""
Postgres + pgvector implementations of MonitorStore, FeedbackStore, CacheStore.

Requires the optional `postgres` extra:
    pip install raggit[postgres]

Each store opens its own internal connection pool (default min=1, max=5).
The pool runs in autocommit mode and registers pgvector's adapters so
list[float] values can be passed directly into VECTOR columns.

Vector similarity uses pgvector's cosine distance operator `<=>`. We convert
distance → similarity (1 - distance) at query time so the user-facing
`cluster_threshold` semantics match SQLite (similarity ≥ threshold).
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from ..models import Cluster, Event, RetrievalHandle
from .base import CacheStore, FeedbackStore, MonitorStore

try:
    import psycopg
    from pgvector.psycopg import register_vector
    from psycopg_pool import ConnectionPool
    _POSTGRES_AVAILABLE = True
except ImportError:
    _POSTGRES_AVAILABLE = False


_POSTGRES_INSTALL_HINT = (
    "Postgres support requires `pip install raggit[postgres]` "
    "(installs psycopg[binary], psycopg_pool, pgvector)."
)


def _require_postgres() -> None:
    if not _POSTGRES_AVAILABLE:
        raise ImportError(_POSTGRES_INSTALL_HINT)


def _configure_conn(conn) -> None:
    """Per-connection setup: autocommit + pgvector type adapters."""
    conn.autocommit = True
    register_vector(conn)


def _make_pool(dsn: str, pool_size: int) -> "ConnectionPool":
    return ConnectionPool(
        dsn,
        min_size=1,
        max_size=pool_size,
        configure=_configure_conn,
        open=True,
    )


def _ensure_extension(pool: "ConnectionPool") -> None:
    """Best-effort CREATE EXTENSION vector. Swallow permission errors —
    on managed Postgres (RDS, Cloud SQL, Supabase, Neon) the extension may
    be pre-installed but unavailable to non-admin users; document in the
    error message that the user may need to run it once as admin.
    """
    try:
        with pool.connection() as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    except psycopg.errors.InsufficientPrivilege:
        pass


def _column_exists_pg(conn, table: str, column: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM information_schema.columns "
        "WHERE table_name = %s AND column_name = %s",
        (table, column),
    ).fetchone()
    return row is not None


def _row_to_cluster(row) -> Cluster:
    """Accept rows of length 6 (no latest_response) or 7+ (latest_response in
    position 6). Postgres returns native datetime objects and pgvector
    returns numpy arrays — coerce vec to list[float]."""
    cluster_id, rep_vec, rep_query, count, created_at, last_seen = row[:6]
    latest_response = row[6] if len(row) > 6 else None
    return Cluster(
        cluster_id=cluster_id,
        representative_vec=[float(x) for x in rep_vec] if rep_vec is not None else [],
        representative_query=rep_query,
        count=count,
        created_at=created_at if isinstance(created_at, datetime) else datetime.fromisoformat(created_at),
        last_seen=last_seen if isinstance(last_seen, datetime) else datetime.fromisoformat(last_seen),
        latest_response=latest_response,
    )


def _row_to_event(row) -> Event:
    event_id, cluster_id, query_text, latency_ms, timestamp, cache_hit, \
        retrieval_score, retrieved_doc_ids = row
    return Event(
        event_id=event_id,
        cluster_id=cluster_id,
        query_text=query_text,
        latency_ms=latency_ms,
        timestamp=timestamp if isinstance(timestamp, datetime) else datetime.fromisoformat(timestamp),
        cache_hit=bool(cache_hit),
        retrieval_score=retrieval_score,
        retrieved_doc_ids=retrieved_doc_ids,  # JSONB → Python list directly
    )


def _clusters_ddl(dim: int) -> str:
    return f"""
        CREATE TABLE IF NOT EXISTS clusters (
            cluster_id TEXT PRIMARY KEY,
            representative_vec VECTOR({dim}) NOT NULL,
            representative_query TEXT NOT NULL,
            count INTEGER DEFAULT 0,
            created_at TIMESTAMPTZ NOT NULL,
            last_seen TIMESTAMPTZ NOT NULL,
            latest_response TEXT
        )
    """


_EVENTS_DDL = """
    CREATE TABLE IF NOT EXISTS events (
        event_id TEXT PRIMARY KEY,
        cluster_id TEXT NOT NULL REFERENCES clusters(cluster_id),
        query_text TEXT NOT NULL,
        latency_ms REAL,
        timestamp TIMESTAMPTZ NOT NULL,
        cache_hit BOOLEAN DEFAULT FALSE,
        retrieval_score REAL,
        retrieved_doc_ids JSONB
    )
"""

_CLUSTERS_SELECT = """SELECT cluster_id, representative_vec, representative_query,
                            count, created_at, last_seen, latest_response
                     FROM clusters"""


def _ensure_clusters_schema(conn, dim: int) -> None:
    conn.execute(_clusters_ddl(dim))
    if not _column_exists_pg(conn, "clusters", "latest_response"):
        conn.execute("ALTER TABLE clusters ADD COLUMN latest_response TEXT")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_clusters_vec "
        "ON clusters USING hnsw (representative_vec vector_cosine_ops)"
    )


def _assign_or_create_cluster(conn, vec: List[float], threshold: float, query: str, now: datetime) -> str:
    """Find best cluster via pgvector HNSW, or create a new one. Returns cluster_id."""
    row = conn.execute(
        "SELECT cluster_id, 1 - (representative_vec <=> %s::vector) AS similarity "
        "FROM clusters "
        "ORDER BY representative_vec <=> %s::vector "
        "LIMIT 1",
        (vec, vec),
    ).fetchone()

    if row is not None and row[1] >= threshold:
        cluster_id = row[0]
        conn.execute(
            "UPDATE clusters SET count = count + 1, last_seen = %s WHERE cluster_id = %s",
            (now, cluster_id),
        )
    else:
        cluster_id = str(uuid.uuid4())
        conn.execute(
            "INSERT INTO clusters "
            "(cluster_id, representative_vec, representative_query, count, created_at, last_seen) "
            "VALUES (%s, %s, %s, %s, %s, %s)",
            (cluster_id, vec, query, 1, now, now),
        )
    return cluster_id


_PG_TO_PYTHON: Dict[str, type] = {
    "text": str, "varchar": str, "character varying": str, "char": str,
    "real": float, "double precision": float, "numeric": float,
    "integer": int, "bigint": int, "smallint": int,
    "boolean": bool,
    "bytea": bytes,
    "timestamp with time zone": datetime, "timestamp without time zone": datetime,
    "jsonb": dict, "json": dict,
}

_INTERNAL_EVENT_COLS = frozenset({
    "event_id", "cluster_id", "query_text", "latency_ms", "timestamp", "cache_hit",
    "retrieval_score", "retrieved_doc_ids",
})


# ── MonitorStore implementations ──────────────────────────────────────────────

class PostgresMonitorStore(MonitorStore):
    """
    Postgres + pgvector MonitorStore with full per-query history.
    Tables: clusters (with HNSW-indexed VECTOR column), events.

    Pair with PostgresEventFeedbackStore for per-event feedback.

    `dim` is required at construction so the VECTOR(dim) column is typed —
    typed columns get HNSW indexes for O(log n) ANN search instead of the
    O(n) scan SQLite does.

    Roadmap: derive `dim` from the embedder automatically. For now it's
    explicit.
    """

    keeps_events = True

    def __init__(
        self,
        dsn: str,
        dim: int,
        pool_size: int = 5,
    ):
        _require_postgres()
        self.dsn = dsn
        self.dim = dim
        self.pool = _make_pool(dsn, pool_size)
        _ensure_extension(self.pool)
        with self.pool.connection() as conn:
            _ensure_clusters_schema(conn, dim)
            conn.execute(_EVENTS_DDL)

    def close(self) -> None:
        """Close the connection pool. Long-running apps can call this on exit."""
        self.pool.close()

    def get_schema(self) -> Dict[str, type]:
        with self.pool.connection() as conn:
            rows = conn.execute(
                "SELECT column_name, data_type FROM information_schema.columns "
                "WHERE table_name = 'events'"
            ).fetchall()
        return {
            row[0]: _PG_TO_PYTHON.get(row[1].lower(), str)
            for row in rows
            if row[0] not in _INTERNAL_EVENT_COLS
        }

    def assign_cluster(self, vec: List[float], threshold: float, query: str) -> str:
        now = datetime.now()
        with self.pool.connection() as conn:
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

        now = datetime.now()
        with self.pool.connection() as conn:
            if cluster_id is None:
                cluster_id = _assign_or_create_cluster(conn, vec, threshold, query, now)

            base_cols = [
                "event_id", "cluster_id", "query_text", "latency_ms", "timestamp",
                "cache_hit", "retrieval_score", "retrieved_doc_ids",
            ]
            base_vals = [
                event_id or str(uuid.uuid4()), cluster_id, query, latency_ms, now,
                cache_hit, retrieval_score,
                psycopg.types.json.Jsonb(retrieved_doc_ids) if retrieved_doc_ids is not None else None,
            ]
            extra_cols = list(kwargs.keys())
            all_cols = base_cols + extra_cols
            placeholders = ",".join(["%s"] * len(all_cols))
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
        sql = ("SELECT event_id, cluster_id, query_text, latency_ms, timestamp, "
               "cache_hit, retrieval_score, retrieved_doc_ids "
               "FROM events WHERE TRUE")
        params: list = []
        if cluster_id is not None:
            sql += " AND cluster_id = %s"
            params.append(cluster_id)
        if since is not None:
            sql += " AND timestamp >= %s"
            params.append(since)
        if has_retrieved_docs is True:
            sql += " AND retrieved_doc_ids IS NOT NULL"
        elif has_retrieved_docs is False:
            sql += " AND retrieved_doc_ids IS NULL"
        sql += " ORDER BY timestamp DESC"
        with self.pool.connection() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [_row_to_event(row) for row in rows]

    def get_clusters(
        self,
        top: Optional[int] = None,
        since: Optional[datetime] = None,
        last_seen_before: Optional[datetime] = None,
        min_count: Optional[int] = None,
    ) -> List[Cluster]:
        sql = _CLUSTERS_SELECT + " WHERE TRUE"
        params: list = []
        if since is not None:
            sql += " AND created_at >= %s"
            params.append(since)
        if last_seen_before is not None:
            sql += " AND last_seen <= %s"
            params.append(last_seen_before)
        if min_count is not None:
            sql += " AND count >= %s"
            params.append(min_count)
        sql += " ORDER BY count DESC"
        if top is not None:
            sql += " LIMIT %s"
            params.append(top)
        with self.pool.connection() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [_row_to_cluster(row) for row in rows]

    def get_cluster(self, cluster_id: str) -> Optional[Cluster]:
        with self.pool.connection() as conn:
            row = conn.execute(
                _CLUSTERS_SELECT + " WHERE cluster_id = %s", (cluster_id,)
            ).fetchone()
        return _row_to_cluster(row) if row else None

    def update_latest_response(self, cluster_id: str, response: str) -> None:
        with self.pool.connection() as conn:
            conn.execute(
                "UPDATE clusters SET latest_response = %s WHERE cluster_id = %s",
                (response, cluster_id),
            )

    def stats(self) -> Dict:
        with self.pool.connection() as conn:
            total_events = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
            unique_clusters = conn.execute("SELECT COUNT(*) FROM clusters").fetchone()[0]
        return {
            "total_events": total_events,
            "unique_clusters": unique_clusters,
            "top_clusters": self.get_clusters(top=5),
        }


class PostgresClusterStore(MonitorStore):
    """
    Postgres + pgvector MonitorStore — aggregate only, no per-query history.
    Pair with PostgresClusterFeedbackStore for counter-based feedback.
    """

    # keeps_events stays False (default)

    def __init__(self, dsn: str, dim: int, pool_size: int = 5):
        _require_postgres()
        self.dsn = dsn
        self.dim = dim
        self.pool = _make_pool(dsn, pool_size)
        _ensure_extension(self.pool)
        with self.pool.connection() as conn:
            _ensure_clusters_schema(conn, dim)

    def close(self) -> None:
        self.pool.close()

    def get_schema(self) -> Dict[str, type]:
        return {}

    def assign_cluster(self, vec: List[float], threshold: float, query: str) -> str:
        now = datetime.now()
        with self.pool.connection() as conn:
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
        now = datetime.now()
        with self.pool.connection() as conn:
            _assign_or_create_cluster(conn, vec, threshold, query, now)

    def get_clusters(
        self,
        top: Optional[int] = None,
        since: Optional[datetime] = None,
        last_seen_before: Optional[datetime] = None,
        min_count: Optional[int] = None,
    ) -> List[Cluster]:
        sql = _CLUSTERS_SELECT + " WHERE TRUE"
        params: list = []
        if since is not None:
            sql += " AND created_at >= %s"
            params.append(since)
        if last_seen_before is not None:
            sql += " AND last_seen <= %s"
            params.append(last_seen_before)
        if min_count is not None:
            sql += " AND count >= %s"
            params.append(min_count)
        sql += " ORDER BY count DESC"
        if top is not None:
            sql += " LIMIT %s"
            params.append(top)
        with self.pool.connection() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [_row_to_cluster(row) for row in rows]

    def get_cluster(self, cluster_id: str) -> Optional[Cluster]:
        with self.pool.connection() as conn:
            row = conn.execute(
                _CLUSTERS_SELECT + " WHERE cluster_id = %s", (cluster_id,)
            ).fetchone()
        return _row_to_cluster(row) if row else None

    def update_latest_response(self, cluster_id: str, response: str) -> None:
        with self.pool.connection() as conn:
            conn.execute(
                "UPDATE clusters SET latest_response = %s WHERE cluster_id = %s",
                (response, cluster_id),
            )

    def stats(self) -> Dict:
        with self.pool.connection() as conn:
            rows = conn.execute("SELECT count FROM clusters").fetchall()
        total = sum(row[0] for row in rows)
        return {
            "total_events": total,
            "unique_clusters": len(rows),
            "top_clusters": self.get_clusters(top=5),
        }


# ── FeedbackStore implementations ─────────────────────────────────────────────

class PostgresEventFeedbackStore(FeedbackStore):
    """
    Per-event feedback. Pairs with PostgresMonitorStore.
    Same DB. Creates a separate `feedback` table FK'd to events.
    """

    def __init__(self, dsn: str, dim: int, pool_size: int = 5):
        _require_postgres()
        self.dsn = dsn
        self.dim = dim
        self.pool = _make_pool(dsn, pool_size)
        _ensure_extension(self.pool)
        with self.pool.connection() as conn:
            _ensure_clusters_schema(conn, dim)
            conn.execute(_EVENTS_DDL)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    feedback_id TEXT PRIMARY KEY,
                    event_id TEXT NOT NULL REFERENCES events(event_id),
                    accepted BOOLEAN,
                    score REAL,
                    comment TEXT,
                    timestamp TIMESTAMPTZ NOT NULL
                )
            """)

    def close(self) -> None:
        self.pool.close()

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
                "Pair PostgresClusterStore with PostgresClusterFeedbackStore instead."
            )
        with self.pool.connection() as conn:
            conn.execute(
                "INSERT INTO feedback (feedback_id, event_id, accepted, score, comment, timestamp) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                (str(uuid.uuid4()), handle.event_id, accepted, score, comment, datetime.now()),
            )

    def get_feedback_summary(self, cluster_id: str) -> Optional[Dict]:
        sql = """
            SELECT
                COUNT(*),
                COUNT(*) FILTER (WHERE f.accepted IS NOT NULL),
                COUNT(*) FILTER (WHERE f.accepted = TRUE),
                COUNT(*) FILTER (WHERE f.score IS NOT NULL),
                COALESCE(SUM(f.score), 0)
            FROM feedback f
            JOIN events e ON e.event_id = f.event_id
            WHERE e.cluster_id = %s
        """
        with self.pool.connection() as conn:
            row = conn.execute(sql, (cluster_id,)).fetchone()
        if row is None or (row[0] or 0) == 0:
            return None
        total, thumb_total, accepted, score_count, score_sum = row
        thumb_total = thumb_total or 0
        accepted = accepted or 0
        score_count = score_count or 0
        score_sum = score_sum or 0.0
        return {
            "feedback_count": thumb_total,
            "accepted_count": accepted,
            "acceptance_rate": (accepted / thumb_total) if thumb_total > 0 else None,
            "score_count": score_count,
            "avg_score": (score_sum / score_count) if score_count > 0 else None,
        }

    def acceptance_rate_per_cluster(
        self,
        top: Optional[int] = None,
    ) -> List[Tuple[Cluster, float]]:
        sql = """
            SELECT c.cluster_id, c.representative_vec, c.representative_query,
                   c.count, c.created_at, c.last_seen,
                   SUM(CASE WHEN f.accepted = TRUE THEN 1 ELSE 0 END) * 1.0
                       / NULLIF(SUM(CASE WHEN f.accepted IS NOT NULL THEN 1 ELSE 0 END), 0) AS rate
            FROM clusters c
            JOIN events e   ON e.cluster_id = c.cluster_id
            JOIN feedback f ON f.event_id   = e.event_id
            GROUP BY c.cluster_id, c.representative_vec, c.representative_query,
                     c.count, c.created_at, c.last_seen
            HAVING SUM(CASE WHEN f.accepted IS NOT NULL THEN 1 ELSE 0 END) > 0
            ORDER BY c.count DESC
        """
        if top is not None:
            sql += " LIMIT %s"
        with self.pool.connection() as conn:
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
            GROUP BY c.cluster_id, c.representative_vec, c.representative_query,
                     c.count, c.created_at, c.last_seen
            ORDER BY c.count DESC
        """
        if top is not None:
            sql += " LIMIT %s"
        with self.pool.connection() as conn:
            rows = conn.execute(sql, (top,) if top is not None else ()).fetchall()
        return [(_row_to_cluster(row[:6]), row[6]) for row in rows]


class PostgresClusterFeedbackStore(FeedbackStore):
    """
    Counter-based feedback. Pairs with PostgresClusterStore.
    Appends 4 counter columns to the existing `clusters` table — no separate
    feedback table. Comments are silently dropped.
    """

    _FEEDBACK_COLUMNS = {
        "accepted_count": "INTEGER DEFAULT 0",
        "feedback_count": "INTEGER DEFAULT 0",
        "score_sum":      "REAL DEFAULT 0",
        "score_count":    "INTEGER DEFAULT 0",
    }

    def __init__(self, dsn: str, dim: int, pool_size: int = 5):
        _require_postgres()
        self.dsn = dsn
        self.dim = dim
        self.pool = _make_pool(dsn, pool_size)
        _ensure_extension(self.pool)
        with self.pool.connection() as conn:
            _ensure_clusters_schema(conn, dim)
            for col, decl in self._FEEDBACK_COLUMNS.items():
                if not _column_exists_pg(conn, "clusters", col):
                    conn.execute(f"ALTER TABLE clusters ADD COLUMN {col} {decl}")

    def close(self) -> None:
        self.pool.close()

    def record_feedback(
        self,
        handle: RetrievalHandle,
        accepted: Optional[bool] = None,
        score: Optional[float] = None,
        comment: Optional[str] = None,
    ) -> None:
        # comment is intentionally dropped — no per-event row to attach it to.
        accepted_int = None if accepted is None else (1 if accepted else 0)
        with self.pool.connection() as conn:
            conn.execute(
                """UPDATE clusters
                   SET feedback_count = feedback_count
                       + (CASE WHEN %s::INTEGER IS NOT NULL THEN 1 ELSE 0 END),
                       accepted_count = accepted_count
                       + (CASE WHEN %s::INTEGER = 1 THEN 1 ELSE 0 END),
                       score_count    = score_count
                       + (CASE WHEN %s::REAL IS NOT NULL THEN 1 ELSE 0 END),
                       score_sum      = score_sum + COALESCE(%s::REAL, 0)
                   WHERE cluster_id = %s""",
                (accepted_int, accepted_int, score, score, handle.cluster_id),
            )

    def get_feedback_summary(self, cluster_id: str) -> Optional[Dict]:
        with self.pool.connection() as conn:
            row = conn.execute(
                "SELECT accepted_count, feedback_count, score_sum, score_count "
                "FROM clusters WHERE cluster_id = %s",
                (cluster_id,),
            ).fetchone()
        if row is None:
            return None
        accepted_count, feedback_count, score_sum, score_count = row
        if (feedback_count or 0) == 0 and (score_count or 0) == 0:
            return None
        return {
            "feedback_count": feedback_count or 0,
            "accepted_count": accepted_count or 0,
            "acceptance_rate": (accepted_count / feedback_count) if feedback_count else None,
            "score_count":    score_count or 0,
            "avg_score":      (score_sum / score_count) if score_count else None,
        }

    def acceptance_rate_per_cluster(
        self,
        top: Optional[int] = None,
    ) -> List[Tuple[Cluster, float]]:
        sql = (
            "SELECT cluster_id, representative_vec, representative_query, "
            "       count, created_at, last_seen, "
            "       accepted_count * 1.0 / feedback_count AS rate "
            "FROM clusters "
            "WHERE feedback_count > 0 "
            "ORDER BY count DESC"
        )
        if top is not None:
            sql += " LIMIT %s"
        with self.pool.connection() as conn:
            rows = conn.execute(sql, (top,) if top is not None else ()).fetchall()
        return [(_row_to_cluster(row[:6]), row[6]) for row in rows]

    def avg_score_per_cluster(
        self,
        top: Optional[int] = None,
    ) -> List[Tuple[Cluster, float]]:
        sql = (
            "SELECT cluster_id, representative_vec, representative_query, "
            "       count, created_at, last_seen, "
            "       score_sum / score_count AS avg_score "
            "FROM clusters "
            "WHERE score_count > 0 "
            "ORDER BY count DESC"
        )
        if top is not None:
            sql += " LIMIT %s"
        with self.pool.connection() as conn:
            rows = conn.execute(sql, (top,) if top is not None else ()).fetchall()
        return [(_row_to_cluster(row[:6]), row[6]) for row in rows]


# ── CacheStore ────────────────────────────────────────────────────────────────

class PostgresCacheStore(CacheStore):
    """
    Postgres + pgvector CacheStore. HNSW-indexed VECTOR column for fast
    similarity lookups. The optional `cluster_id` column links auto-promoted
    entries back to their source cluster.
    """

    def __init__(self, dsn: str, dim: int, pool_size: int = 5):
        _require_postgres()
        self.dsn = dsn
        self.dim = dim
        self.pool = _make_pool(dsn, pool_size)
        _ensure_extension(self.pool)
        with self.pool.connection() as conn:
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS cache (
                    cache_id TEXT PRIMARY KEY,
                    vec VECTOR({dim}) NOT NULL,
                    response TEXT NOT NULL,
                    approved_by TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL,
                    cluster_id TEXT
                )
            """)
            if not _column_exists_pg(conn, "cache", "cluster_id"):
                conn.execute("ALTER TABLE cache ADD COLUMN cluster_id TEXT")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_cache_vec "
                "ON cache USING hnsw (vec vector_cosine_ops)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_cache_cluster_id ON cache(cluster_id)"
            )

    def close(self) -> None:
        self.pool.close()

    def get(self, vec: List[float], threshold: float) -> Optional[str]:
        with self.pool.connection() as conn:
            row = conn.execute(
                "SELECT response, 1 - (vec <=> %s::vector) AS similarity "
                "FROM cache "
                "ORDER BY vec <=> %s::vector "
                "LIMIT 1",
                (vec, vec),
            ).fetchone()
        if row is None or row[1] < threshold:
            return None
        return row[0]

    def set(
        self,
        vec: List[float],
        response: str,
        approved_by: str = "llm",
        cluster_id: Optional[str] = None,
    ) -> None:
        # Idempotency guard for auto-promotion
        if cluster_id is not None and approved_by == "auto":
            with self.pool.connection() as conn:
                exists = conn.execute(
                    "SELECT 1 FROM cache WHERE cluster_id = %s AND approved_by = 'auto' LIMIT 1",
                    (cluster_id,),
                ).fetchone()
            if exists is not None:
                return

        with self.pool.connection() as conn:
            conn.execute(
                "INSERT INTO cache (cache_id, vec, response, approved_by, created_at, cluster_id) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                (str(uuid.uuid4()), vec, response, approved_by, datetime.now(), cluster_id),
            )

    def delete_by_cluster(self, cluster_id: str) -> None:
        with self.pool.connection() as conn:
            conn.execute(
                "DELETE FROM cache WHERE cluster_id = %s AND approved_by = 'auto'",
                (cluster_id,),
            )

    def has_auto_entry(self, cluster_id: str) -> bool:
        with self.pool.connection() as conn:
            row = conn.execute(
                "SELECT 1 FROM cache WHERE cluster_id = %s AND approved_by = 'auto' LIMIT 1",
                (cluster_id,),
            ).fetchone()
        return row is not None
