# Feedback — v1 Design

Feedback lets users (or downstream services) tell Raggit whether a retrieved
answer was good or bad. The signal feeds back into the monitor so devs can see
which clustered queries have low acceptance — i.e. which ones to write evals
against next.

## Design philosophy

- **Opt-in.** Users who don't want feedback never see any of it. No flag, no
  hidden state — feedback is opt-in by virtue of being a separate API.
- **Library, not server.** Raggit stays HTTP-free. We provide the function;
  the user wires it to their own framework's route.
- **Manual, not auto.** Feedback surfaces problematic clusters; the human still
  decides what to do (write an eval, swap a model, fix the corpus). No
  auto-suite generation.
- **YAGNI, with one pragmatic exception.** Default feedback fields are
  `accepted`, `score`, `comment` — fixed schema. No `**kwargs`. `score` is
  included even though pure YAGNI would skip it, because rating UIs (1–5
  stars, NPS, etc.) are common enough that forcing those users to fork is
  more painful than carrying one nullable column.
- **Separate concern, separate ABC.** `FeedbackStore` is a distinct ABC from
  `MonitorStore`. A `Monitor` wires one of each. Custom `MonitorStore`
  implementations don't have to know feedback exists; users opt in by passing
  a paired `FeedbackStore`.
- **Storage layout matches the paired store's philosophy.** The event-flavored
  feedback store creates its own `feedback` table FK'd to events. The
  cluster-flavored feedback store appends counter columns to the `clusters`
  table that the paired `MonitorStore` already created. No new feedback table
  for ClusterStore — that would contradict its "aggregates only" philosophy.

---

## Public API

### Decorator: `@mw.track_with_handle`

Parallel to `@mw.track`. Returns a `RetrievalHandle` instead of the plain
answer, so the caller has the IDs needed to attach feedback later.

```python
from raggit.middleware import (
    Middleware, Monitor, SQLiteMonitorStore, SQLiteEventFeedbackStore,
)

db = ".raggit/monitor.db"
monitor = Monitor(
    store=SQLiteMonitorStore(db),
    feedback_store=SQLiteEventFeedbackStore(db),
    embedder=embed,
)
mw = Middleware(monitor=monitor, embedder=embed)

@mw.track_with_handle
def retrieve(query: str) -> str:
    return my_index.search(query)[0]

handle = retrieve("how do I reset my password")
print(handle.answer)         # the actual response
print(handle.event_id)       # for record_feedback (None on ClusterStore pairings)
print(handle.cluster_id)     # always populated
```

`@mw.track` is **unchanged** — existing users keep the simple
`retrieve(query) -> str` contract.

### Recording feedback: `monitor.record_feedback(handle, ...)`

```python
monitor.record_feedback(handle, accepted=True)                       # thumb-only
monitor.record_feedback(handle, score=0.8)                           # rating-only
monitor.record_feedback(handle, accepted=True, score=0.8, comment="great")
```

**Validation (Monitor layer):**
- At least one of `accepted` / `score` must be provided. A bare
  `record_feedback(handle)` raises `ValueError`.
- `Monitor` must have a `feedback_store` wired in. Calling `record_feedback`
  on a Monitor without one raises a `ValueError` pointing at the wiring.

The user wires this into their own HTTP route:

```python
@app.post("/feedback")
def feedback(req: FeedbackReq):
    handle = handle_cache.pop(req.handle_token)  # however the app stores it
    monitor.record_feedback(
        handle,
        accepted=req.accepted,
        score=req.score,
        comment=req.comment,
    )
```

### Reading feedback

Two methods. Same shape, different SQL underneath depending on which
`FeedbackStore` is paired in.

```python
for cluster, rate in monitor.acceptance_rate_per_cluster(top=10):
    print(f"{rate:.0%}  {cluster.count}x  {cluster.representative_query!r}")

for cluster, avg in monitor.avg_score_per_cluster(top=10):
    print(f"{avg:.2f}  {cluster.count}x  {cluster.representative_query!r}")
```

---

## Architecture

Two separate ABCs. Pair one of each at the `Monitor` layer.

```
MonitorStore         FeedbackStore
─────────────         ─────────────
get_schema()          record_feedback(handle, accepted, score, comment)
log()                 acceptance_rate_per_cluster(top)   # optional
assign_cluster()      avg_score_per_cluster(top)         # optional
get_clusters()
stats()
keeps_events
```

A `Monitor` takes both:

```python
Monitor(store=<MonitorStore>, feedback_store=<FeedbackStore>, embedder=...)
```

`feedback_store` is `Optional[FeedbackStore] = None`. If absent, calling any
feedback method on `Monitor` raises a clear error. Existing users without
feedback are unaffected.

---

## Internal: `RetrievalHandle`

```python
@dataclass
class RetrievalHandle:
    answer: Any
    cluster_id: str          # always populated (clustering happens synchronously)
    event_id: str | None = None   # only set when the MonitorStore keeps event-level history
```

- `cluster_id` is **always** present because clustering is moved to be
  synchronous (the store write can stay async; only the embed+cluster lookup
  becomes sync). The few-ms cost buys a non-None contract.
- `event_id` is `None` for `SQLiteClusterStore` (no events table), populated
  for `SQLiteMonitorStore`.

---

## Concrete implementations

The two pairings ship in `raggit.middleware.stores.sqlite`. Both stores in a
pair share the same DB file. Each is idempotent on init, so order of
instantiation doesn't matter.

### Event pairing — `SQLiteMonitorStore` + `SQLiteEventFeedbackStore`

`SQLiteMonitorStore` creates `clusters` and `events`.
`SQLiteEventFeedbackStore` creates a `feedback` table FK'd to events:

```sql
CREATE TABLE IF NOT EXISTS feedback (
    feedback_id TEXT PRIMARY KEY,
    event_id    TEXT NOT NULL,
    accepted    INTEGER,                 -- 1 / 0 / NULL (rating-only UIs)
    score       REAL,                    -- 0..1 normalized rating, or NULL
    comment     TEXT,
    timestamp   TEXT NOT NULL,
    FOREIGN KEY (event_id) REFERENCES events(event_id)
);
```

`record_feedback(handle, ...)` INSERTs one row keyed on `handle.event_id`.
Cluster-level aggregates are a JOIN away.

### Cluster pairing — `SQLiteClusterStore` + `SQLiteClusterFeedbackStore`

`SQLiteClusterStore` creates `clusters` (and nothing else).
`SQLiteClusterFeedbackStore` **appends 4 counter columns to the existing
`clusters` table** — no separate feedback table:

```sql
ALTER TABLE clusters ADD COLUMN accepted_count INTEGER DEFAULT 0;  -- WHERE accepted=1
ALTER TABLE clusters ADD COLUMN feedback_count INTEGER DEFAULT 0;  -- WHERE accepted IS NOT NULL
ALTER TABLE clusters ADD COLUMN score_sum      REAL    DEFAULT 0;
ALTER TABLE clusters ADD COLUMN score_count    INTEGER DEFAULT 0;
```

Migrations are idempotent — `__init__` checks `PRAGMA table_info(clusters)`
before each `ALTER TABLE`.

`record_feedback(handle, accepted=True, score=0.8)` runs one UPDATE on the
counters keyed by `handle.cluster_id`. Comments are silently dropped — there's
no row to attach them to. This is documented behavior, not a bug.

### Read API SQL

| Method | Event pairing | Cluster pairing |
|---|---|---|
| `acceptance_rate_per_cluster` | `SELECT cluster_id, SUM(accepted=1)*1.0 / SUM(accepted IS NOT NULL) FROM feedback JOIN events USING(event_id) GROUP BY cluster_id HAVING SUM(accepted IS NOT NULL) > 0 ORDER BY count DESC LIMIT ?` | `SELECT cluster_id, accepted_count*1.0 / feedback_count FROM clusters WHERE feedback_count > 0 ORDER BY count DESC LIMIT ?` |
| `avg_score_per_cluster` | `SELECT cluster_id, AVG(score) FROM feedback JOIN events USING(event_id) WHERE score IS NOT NULL GROUP BY cluster_id ORDER BY count DESC LIMIT ?` | `SELECT cluster_id, score_sum/score_count FROM clusters WHERE score_count > 0 ORDER BY count DESC LIMIT ?` |

Result shape is identical: `List[tuple[Cluster, float]]`.

---

## What does NOT change

- `@mw.track` return type. Still `str` (or whatever the wrapped function
  returns). Existing users unaffected.
- `Monitor.get_schema()`. Still describes the `events` table only. No
  `get_feedback_schema()` until somebody actually wants `**kwargs` validation
  on feedback.
- The cache layer. Feedback is monitor-side only.
- `MonitorStore` ABC's pre-feedback contract — feedback methods now live on
  `FeedbackStore`, not `MonitorStore`. Custom `MonitorStore` implementations
  remain feedback-agnostic.

---

## Out of scope for v1 (deferred)

| Feature | Why deferred |
|---|---|
| HTTP `/feedback` endpoint | Raggit is a library, not a server. Users wire `record_feedback` into their own framework in 3 lines. Could add `raggit[fastapi]` extras later. |
| `**kwargs` on feedback (`user_id`, `helpfulness_axis`, etc.) | The 80/20 case is `accepted` + `score` + `comment`. Adding `get_feedback_schema()` is a clean upgrade path when someone asks. |
| `popular_with_negative_feedback(top=N)` | `acceptance_rate_per_cluster` already surfaces this — sort by rate ascending, filter by count. A dedicated method is sugar; add if requested. |
| Auto eval-suite generation from low-acceptance clusters | Explicitly rejected per the design memory. Monitor surfaces; human decides correctness. |
| Time-windowed acceptance (rolling 7d, etc.) | Possible from event-pairing data via filters; cluster-pairing loses this by design. Not v1. |
| Embedder retraining on feedback labels | Out of scope entirely — Raggit is BYO embedder. |
| Mix-and-match: e.g. `RedisFeedbackStore` paired with `SQLiteMonitorStore` | The ABC split makes this possible later; no concrete impls in v1 beyond the two SQLite pairings. |

---

## Implementation status

All implemented in this branch:

- [x] `RetrievalHandle` dataclass in `middleware/models.py`
- [x] `Middleware.track_with_handle` decorator (sync clustering, async write)
- [x] `MonitorStore.assign_cluster` (optional, raises `NotImplementedError` by default)
- [x] `FeedbackStore` ABC with `record_feedback` (abstract) and optional reads
- [x] `SQLiteEventFeedbackStore` — feedback table, FK to events
- [x] `SQLiteClusterFeedbackStore` — counter columns on clusters
- [x] `Monitor.feedback_store` parameter + delegated feedback methods
- [x] Idempotent migrations, order-agnostic instantiation
- [x] Tests: handle shape, validation, both pairings, idempotent migrations,
      Monitor-without-feedback-store error path
- [x] README "Feedback" section
- [x] Roadmap checkbox flipped
