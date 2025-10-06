import os
import json
import logging
from typing import List, Dict, Any, Optional

import requests
import pandas as pd
import numpy as np
import shap
import xgboost as xgb
import mlflow
from mlflow.tracking import MlflowClient
from fastapi import FastAPI, HTTPException

# ----------- Config -----------
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow_server:5000")
MLFLOW_REGISTRY_URI = os.getenv("MLFLOW_REGISTRY_URI", MLFLOW_TRACKING_URI)
MODEL_NAME = os.getenv("MODEL_NAME", "event_classifier")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "staging")
SERVE_URL = os.getenv("SERVE_URL", "http://model_server:6000/invocations")  # proba endpoint
THRESHOLD = float(os.getenv("ROUTING_THRESHOLD", "0.6"))
TOP_K = int(os.getenv("TOP_K", "5"))

# Optional sinks
USE_REDIS = os.getenv("USE_REDIS", "0") == "1"
USE_PG = os.getenv("USE_PG", "0") == "1"
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
PG_DSN = os.getenv("PG_DSN", "postgresql+psycopg2://user:pass@postgres:5432/mlflow_db")

logging.basicConfig(level=logging.INFO, format="[Router] %(message)s")

# ----------- Init MLflow ----------
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_registry_uri(MLFLOW_REGISTRY_URI)
client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI, registry_uri=MLFLOW_REGISTRY_URI)

# Resolve alias -> version -> run_id
mv = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
RUN_ID = mv.run_id
logging.info(f"Resolved {MODEL_NAME}@{MODEL_ALIAS} -> v{mv.version}, run_id={RUN_ID}")

# Fetch feature columns (top-level artifact)
FEATURE_COLS_PATH = client.download_artifacts(RUN_ID, "feature_columns.json")
FEATURE_COLS = json.load(open(FEATURE_COLS_PATH))
logging.info(f"Loaded {len(FEATURE_COLS)} feature columns")

# Load raw XGBoost model (for local SHAP) from model_xgb
# Note: we logged raw xgb to artifact_path="model_xgb" in train.py
XGB_URI = f"runs:/{RUN_ID}/model_xgb"
xgb_model: xgb.XGBClassifier = mlflow.xgboost.load_model(XGB_URI)
logging.info("Loaded XGBoost model for local SHAP")

# SHAP explainer (tree explainer is fast for XGB)
explainer = shap.TreeExplainer(xgb_model)

# Optional sinks
redis_client = None
if USE_REDIS:
    import redis
    redis_client = redis.from_url(REDIS_URL)
    logging.info("Connected to Redis")

pg_engine = None
if USE_PG:
    from sqlalchemy import create_engine, text
    pg_engine = create_engine(PG_DSN, pool_pre_ping=True)
    # Simple table for routing decisions (idempotent)
    with pg_engine.begin() as conn:
        conn.execute(
            text("""
            CREATE TABLE IF NOT EXISTS routing_events (
                event_id VARCHAR(64) PRIMARY KEY,
                score DOUBLE PRECISION,
                decision VARCHAR(32),
                topk JSONB,
                created_at TIMESTAMP DEFAULT NOW()
            )
            """)
        )
    logging.info("Postgres sink ready")

# ----------- FastAPI app ----------
app = FastAPI(title="AIEO Router", version="0.1.0")

def df_from_events(events: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Events are dicts that include at least:
      - event_id (string)
      - event_type (string)
    We construct the features with the same one-hot columns as training.
    """
    df = pd.DataFrame(events)
    if "event_type" not in df.columns:
        raise ValueError("event_type field is required in each event")

    # Build the same one-hot set and reindex to FEATURE_COLS
    X = pd.get_dummies(df[["event_type"]], drop_first=True)
    # Ensure all expected columns exist, in the same order
    X = X.reindex(columns=FEATURE_COLS, fill_value=0)
    return df, X

def topk_shap(shap_row: np.ndarray, feature_names: List[str], k: int) -> List[Dict[str, float]]:
    # rank by absolute impact
    idx = np.argsort(-np.abs(shap_row))[:k]
    return [{ "feature": feature_names[i], "shap": float(shap_row[i]) } for i in idx]

def route_decision(score: float, threshold: float = THRESHOLD) -> str:
    return "low_latency" if score >= threshold else "batch"

def persist(event_id: str, score: float, decision: str, topk: List[Dict[str, float]]):
    if USE_REDIS and redis_client:
        key = f"route:{event_id}"
        redis_client.hset(key, mapping={
            "score": score, "decision": decision, "topk": json.dumps(topk)
        })
    if USE_PG and pg_engine:
        from sqlalchemy import text
        with pg_engine.begin() as conn:
            conn.execute(
                text("INSERT INTO routing_events (event_id, score, decision, topk) "
                     "VALUES (:event_id, :score, :decision, CAST(:topk AS JSONB)) "
                     "ON CONFLICT (event_id) DO UPDATE SET score = EXCLUDED.score, "
                     "decision = EXCLUDED.decision, topk = EXCLUDED.topk"),
                {"event_id": event_id, "score": score, "decision": decision, "topk": json.dumps(topk)}
            )

@app.post("/score")
def score_events(payload: Dict[str, Any]):
    """
    Request body:
    {
      "events": [
        {"event_id":"e1", "event_type":"click"},
        {"event_id":"e2", "event_type":"purchase"}
      ]
    }
    Response:
    { "results": [ {event_id, score, decision, top_k: [...] }, ... ] }
    """
    try:
        events = payload.get("events", [])
        if not events:
            raise ValueError("No events provided")

        df, X = df_from_events(events)

        # 1) call served model for probabilities
        resp = requests.post(
            SERVE_URL,
            headers={"Content-Type": "application/json"},
            json={"inputs": X.to_dict(orient="records")},
            timeout=10,
        )
        if not resp.ok:
            raise HTTPException(status_code=502, detail=f"Model server error: {resp.text}")
        probs = resp.json()
        # mlflow pyfunc returns {"predictions": [..]} or just list
        y = probs["predictions"] if isinstance(probs, dict) and "predictions" in probs else probs
        y = np.array(y, dtype=float)

        # 2) local SHAP per-event using raw XGB
        shap_vals = explainer.shap_values(X)  # shape: [n_rows, n_features]

        results = []
        for i, row in enumerate(events):
            event_id = str(row.get("event_id", f"evt_{i}"))
            score = float(y[i])
            decision = route_decision(score, THRESHOLD)
            topk = topk_shap(shap_vals[i], FEATURE_COLS, TOP_K)

            # persist optional
            persist(event_id, score, decision, topk)

            results.append({
                "event_id": event_id,
                "score": score,
                "decision": decision,
                "top_k": topk
            })

        return {"results": results, "threshold": THRESHOLD}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
def health():
    return {"ok": True, "model_alias": MODEL_ALIAS, "features": len(FEATURE_COLS)}
