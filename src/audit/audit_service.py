# src/audit/audit_service.py
import os, json
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException
from sqlalchemy import create_engine, text

PG_DSN = os.getenv("PG_DSN", "postgresql+psycopg2://user:pass@postgres:5432/mlflow_db")
engine = create_engine(PG_DSN, pool_pre_ping=True)
app = FastAPI(title="Audit API", version="0.1.0")

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