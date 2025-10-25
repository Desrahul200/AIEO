# src/audit/audit_service.py
import os, json
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException
from sqlalchemy import create_engine, text
import time, requests

ROUTER_HEALTH_URL = os.getenv("ROUTER_HEALTH_URL", "http://router:7000/health")
MODEL_PING_URL    = os.getenv("MODEL_PING_URL", "http://model_server:6000/ping")
PG_DSN = os.getenv("PG_DSN", "postgresql+psycopg2://user:pass@postgres:5432/mlflow_db")
engine = create_engine(PG_DSN, pool_pre_ping=True)
app = FastAPI(title="Audit API", version="0.1.0")


def _ping(url: str, timeout: float = 2.5) -> tuple[bool, float, str | None]:
    t0 = time.perf_counter()
    try:
        r = requests.get(url, timeout=timeout)
        ok = r.ok
        detail = None
        # MLflow model server usually exposes /ping -> "pong"
        if not ok:
            detail = f"status={r.status_code}"
        return ok, (time.perf_counter() - t0) * 1000.0, detail
    except Exception as e:
        return False, (time.perf_counter() - t0) * 1000.0, str(e)

@app.get("/system/health")
def system_health():
    # 1) DB ping + quick counts
    db_ok, db_ms, total_5m, routed_5m = False, None, 0, 0
    t0 = time.perf_counter()
    try:
        with engine.connect() as c:
            c.execute(text("SELECT 1"))
            db_ok = True
            # quick situational awareness
            total_5m  = c.execute(text("SELECT COUNT(*) FROM events_scored  WHERE created_at >= NOW() - INTERVAL '5 minutes'")).scalar_one()
            routed_5m = c.execute(text("SELECT COUNT(*) FROM routing_events WHERE updated_at >= NOW() - INTERVAL '5 minutes'")).scalar_one()
        db_ms = (time.perf_counter() - t0) * 1000.0
    except Exception as e:
        db_ms = (time.perf_counter() - t0) * 1000.0
        db_err = str(e)

    # 2) Router + Model pings
    r_ok, r_ms, r_detail = _ping(ROUTER_HEALTH_URL, 2.0)
    m_ok, m_ms, m_detail = _ping(MODEL_PING_URL, 2.0)

    overall = bool(db_ok and r_ok and m_ok)
    return {
        "ok": overall,
        "postgres": {"ok": db_ok, "ms": round(db_ms or 0, 1), "counts_last_5m": {"events": int(total_5m), "routed": int(routed_5m)}},
        "router":   {"ok": r_ok,  "ms": round(r_ms, 1), "detail": r_detail},
        "model":    {"ok": m_ok,  "ms": round(m_ms, 1), "detail": m_detail},
        "ts": datetime.utcnow().isoformat() + "Z",
    }
@app.get("/routing/summary")
def routing_summary(minutes: int = 60):
    q = text("""
      SELECT decision, COUNT(*) AS n, AVG(score) AS avg_score
      FROM v_event_audit
      WHERE decision_created_at >= NOW() - INTERVAL ':m minutes'
      GROUP BY decision
      ORDER BY n DESC
    """.replace(":m", str(minutes)))
    with engine.connect() as c:
        rows = c.execute(q).mappings().all()
    return {"window_min": minutes, "by_decision": [dict(r) for r in rows]}

@app.get("/routing/top-features")
def top_features(limit: int = 10, minutes: int = 60):
    # Unnest JSON topk -> aggregate feature hits + avg shap
    q = text(f"""
      WITH exploded AS (
        SELECT
          (elem->>'feature') AS feature,
          (elem->>'shap')::double precision AS shap
        FROM v_event_audit va,
        LATERAL jsonb_array_elements(va.topk) AS elem
        WHERE va.decision_created_at >= NOW() - INTERVAL '{minutes} minutes'
      )
      SELECT feature, COUNT(*) AS hits, AVG(shap) AS mean_shap, AVG(ABS(shap)) AS mean_abs_shap
      FROM exploded
      GROUP BY feature
      ORDER BY mean_abs_shap DESC
      LIMIT {limit}
    """)
    with engine.connect() as c:
        rows = c.execute(q).mappings().all()
    return {"window_min": minutes, "features": [dict(r) for r in rows]}

@app.get("/events/recent")
def events_recent(limit: int = 50):
    q = text(f"""
      SELECT event_id, event, created_at
      FROM events_scored
      ORDER BY created_at DESC
      LIMIT {limit}
    """)
    with engine.connect() as c:
        rows = c.execute(q).mappings().all()
    out = []
    for r in rows:
        e = dict(r)
        e["event"] = e["event"]  # already JSONB; SQLA returns dict
        out.append(e)
    return {"events": out}

@app.get("/routing/by-id")
def routing_by_id(event_id: str):
    q = text("""
        SELECT r.event_id, r.score, r.decision, r.topk, e.event, e.created_at AS event_created_at
        FROM routing_events r
        LEFT JOIN events_scored e ON e.event_id = r.event_id
        WHERE r.event_id = :eid
        LIMIT 1
    """)
    with engine.connect() as conn:
        row = conn.execute(q, {"eid": event_id}).mappings().first()
    if not row:
        raise HTTPException(status_code=404, detail="not found")
    # ensure JSON fields are dicts
    topk = row["topk"] if isinstance(row["topk"], (dict, list)) else (json.loads(row["topk"]) if row["topk"] else [])
    ev   = row["event"] if isinstance(row["event"], dict) else (json.loads(row["event"]) if row["event"] else {})
    return {
        "event_id": row["event_id"],
        "score": row["score"],
        "decision": row["decision"],
        "topk": topk,
        "event": ev,
        "event_created_at": row["event_created_at"].isoformat() if row["event_created_at"] else None
    }