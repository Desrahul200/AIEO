import os
import json
import logging
from typing import List, Dict, Any
from pathlib import Path
from threading import Lock

import requests
import pandas as pd
import numpy as np
import shap
import xgboost as xgb
import mlflow
from mlflow.tracking import MlflowClient
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ==================== Config ====================
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow_server:5000")
MLFLOW_REGISTRY_URI = os.getenv("MLFLOW_REGISTRY_URI", MLFLOW_TRACKING_URI)
MODEL_NAME = os.getenv("MODEL_NAME", "event_classifier")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "staging")
SERVE_URL = os.getenv("SERVE_URL", "http://model_server:6000/invocations")  # proba endpoint

# RL decider (sidecar) toggle + URL
USE_RL = os.getenv("USE_RL", "0") == "1"
RL_DECIDER_URL = os.getenv("RL_DECIDER_URL")  # e.g. http://rl_decider:8080/route
RL_TIMEOUT = float(os.getenv("RL_TIMEOUT", "2.0"))

# Runtime config (mutable)
_CONFIG_LOCK = Lock()
_CONFIG = {
    "threshold": float(os.getenv("ROUTING_THRESHOLD", "0.6")),
    "top_k": int(os.getenv("TOP_K", "5")),
}

def get_threshold() -> float:
    with _CONFIG_LOCK:
        return _CONFIG["threshold"]

def get_top_k() -> int:
    with _CONFIG_LOCK:
        return _CONFIG["top_k"]

def set_config(threshold: float | None = None, top_k: int | None = None):
    with _CONFIG_LOCK:
        if threshold is not None:
            _CONFIG["threshold"] = threshold
        if top_k is not None:
            _CONFIG["top_k"] = top_k

# Optional sinks
USE_REDIS = os.getenv("USE_REDIS", "0") == "1"
USE_PG = os.getenv("USE_PG", "0") == "1"
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
PG_DSN = os.getenv("PG_DSN", "postgresql+psycopg2://user:pass@postgres:5432/mlflow_db")

# When mounted (compose): D:/Project/AIEO/mlruns:/mlflow/mlruns
ARTIFACT_LOCAL_ROOT = os.getenv("ARTIFACT_LOCAL_ROOT")  # e.g. "/mlflow/mlruns"

logging.basicConfig(level=logging.INFO, format="[Router] %(message)s")

# ==================== MLflow init ====================
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_registry_uri(MLFLOW_REGISTRY_URI)
client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI, registry_uri=MLFLOW_REGISTRY_URI)

# Resolve alias -> version -> run_id
mv = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
RUN_ID = mv.run_id
logging.info(f"Resolved {MODEL_NAME}@{MODEL_ALIAS} -> v{mv.version}, run_id={RUN_ID}")

# Feature columns (top-level artifact)
FEATURE_COLS_PATH = client.download_artifacts(RUN_ID, "feature_columns.json")
FEATURE_COLS = json.load(open(FEATURE_COLS_PATH))
logging.info(f"Loaded {len(FEATURE_COLS)} feature columns")

# ==================== Load XGBoost booster ====================
def _make_classifier_from_file(fp: str) -> xgb.XGBClassifier:
    booster = xgb.Booster()
    booster.load_model(fp)
    clf = xgb.XGBClassifier()
    clf._Booster = booster  # give SHAP a booster-backed classifier
    return clf

xgb_model = None

# Prefer the flat booster.json at run root (what you see in MLflow UI)
try:
    root_infos = client.list_artifacts(RUN_ID, "")
    root_names = [i.path for i in root_infos]
    logging.info(f"[Router] root artifacts: {root_names}")

    if "booster.json" in root_names:
        local_file = client.download_artifacts(RUN_ID, "booster.json")
        xgb_model = _make_classifier_from_file(local_file)
        logging.info("[Router] Loaded XGBoost booster from booster.json (root)")
except Exception as e:
    logging.warning(f"[Router] booster.json download failed: {e}")

# Final filesystem fallback (mounted mlruns)
if xgb_model is None and ARTIFACT_LOCAL_ROOT:
    try:
        run = client.get_run(RUN_ID)
        exp_id = run.info.experiment_id
        candidate = Path(ARTIFACT_LOCAL_ROOT) / str(exp_id) / RUN_ID / "artifacts" / "booster.json"
        if candidate.exists():
            xgb_model = _make_classifier_from_file(str(candidate))
            logging.info(f"[Router] Loaded XGBoost booster from local fs: {candidate}")
    except Exception as e:
        logging.warning(f"[Router] local fs fallback failed: {e}")

if xgb_model is None:
    raise RuntimeError("booster.json not found via API or local fs")

# SHAP explainer (tree explainer is fast for XGB)
explainer = shap.TreeExplainer(xgb_model)

# ==================== Optional sinks ====================
redis_client = None
if USE_REDIS:
    import redis
    redis_client = redis.from_url(REDIS_URL)
    logging.info("Connected to Redis")

pg_engine = None
if USE_PG:
    from sqlalchemy import create_engine, text
    pg_engine = create_engine(PG_DSN, pool_pre_ping=True)
    with pg_engine.begin() as conn:
        conn.execute(
            text("""
                CREATE TABLE IF NOT EXISTS routing_events (
                  event_id   VARCHAR(64) PRIMARY KEY,
                  score      DOUBLE PRECISION,
                  decision   VARCHAR(32),
                  topk       JSONB,
                  created_at TIMESTAMP DEFAULT NOW(),
                  updated_at TIMESTAMP DEFAULT NOW()
                )
            """)
        )
    logging.info("Postgres sink ready")

# ==================== FastAPI app ====================
app = FastAPI(title="AIEO Router", version="0.1.0")

def df_from_events(events: List[Dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.DataFrame(events)
    if "event_type" not in df.columns:
        raise ValueError("event_type field is required in each event")

    # lock category space to training cols so one-off batches don't collapse dummies
    cats = [c.replace("event_type_", "") for c in FEATURE_COLS]
    cat_type = pd.api.types.CategoricalDtype(categories=cats)
    df["event_type"] = df["event_type"].astype(cat_type)

    X = pd.get_dummies(df[["event_type"]], drop_first=False)
    X = X.reindex(columns=FEATURE_COLS, fill_value=0)
    return df, X

def topk_shap(shap_row: np.ndarray, feature_names: List[str], k: int) -> List[Dict[str, float]]:
    idx = np.argsort(-np.abs(shap_row))[:k]
    return [{"feature": feature_names[i], "shap": float(shap_row[i])} for i in idx]

def route_decision(score: float, threshold: float | None = None) -> str:
    if threshold is None:
        threshold = get_threshold()
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
                text("""
                    INSERT INTO routing_events (event_id, score, decision, topk)
                    VALUES (:event_id, :score, :decision, CAST(:topk AS JSONB))
                    ON CONFLICT (event_id) DO UPDATE
                    SET score      = EXCLUDED.score,
                        decision   = EXCLUDED.decision,
                        topk       = EXCLUDED.topk,
                        updated_at = NOW()
                """),
                {"event_id": event_id, "score": score, "decision": decision, "topk": json.dumps(topk)}
            )

# ---------- RL decider integration ----------
def decide_with_decider(score: float, topk_names: List[str]) -> str:
    """
    Ask the RL decider sidecar for a routing action.
    Falls back to threshold routing if request fails or USE_RL is off.
    """
    if not (USE_RL and RL_DECIDER_URL):
        return route_decision(score)

    try:
        payload = {
            "score": float(score),
            "topk_names": topk_names,
            # Optional fields are omitted; decider will infer hour, etc.
        }
        r = requests.post(RL_DECIDER_URL, json=payload, timeout=RL_TIMEOUT)
        r.raise_for_status()
        body = r.json()
        decision = body.get("decision")
        if decision in ("low_latency", "batch", "skip"):
            return decision
        logging.warning(f"[Router] RL decider returned unknown decision: {body}; falling back")
        return route_decision(score)
    except Exception as e:
        logging.warning(f"[Router] RL decider call failed: {e}; falling back to threshold")
        return route_decision(score)

def _load_model_from_mlflow():
    global xgb_model, explainer, FEATURE_COLS, RUN_ID
    mv = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
    RUN_ID = mv.run_id
    # feature columns
    fpath = client.download_artifacts(RUN_ID, "feature_columns.json")
    FEATURE_COLS[:] = json.load(open(fpath))
    # booster
    local_booster = client.download_artifacts(RUN_ID, "booster.json")
    booster = xgb.Booster()
    booster.load_model(local_booster)
    new_clf = xgb.XGBClassifier(); new_clf._Booster = booster
    xgb_model = new_clf
    explainer = shap.TreeExplainer(xgb_model)
    logging.info(f"[Router] Reloaded model {MODEL_NAME}@{MODEL_ALIAS} => run_id={RUN_ID}")



# ---------- endpoints ----------
@app.get("/health")
def health():
    return {
        "ok": True,
        "model_alias": MODEL_ALIAS,
        "features": len(FEATURE_COLS),
        "use_rl": USE_RL,
        "rl_url": RL_DECIDER_URL or "",
    }

class ConfigIn(BaseModel):
    threshold: float | None = Field(None, ge=0.0, le=1.0, description="Routing threshold in [0,1]")
    top_k: int | None = Field(None, ge=1, le=len(FEATURE_COLS), description="Top-K SHAP features")

@app.get("/config")
def get_config():
    return {"threshold": get_threshold(), "top_k": get_top_k()}

@app.post("/config")
def set_config_endpoint(cfg: ConfigIn):
    set_config(threshold=cfg.threshold, top_k=cfg.top_k)
    return {"ok": True, "threshold": get_threshold(), "top_k": get_top_k()}

@app.post("/reload")
def reload_endpoint():
    try:
        _load_model_from_mlflow()
        return {"ok": True, "reloaded_run_id": RUN_ID}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/score")
def score_events(payload: Dict[str, Any]):
    """
    Request:
      {"events":[{"event_id":"e1","event_type":"click"}, ...]}
    Response:
      {"results":[{"event_id": "...", "score": 0.83, "decision": "...", "top_k":[...]}, ...],
       "threshold": 0.6}
    """
    try:
        events = payload.get("events", [])
        if not events:
            raise ValueError("No events provided")

        df, X = df_from_events(events)

        # 1) Call served proba model
        resp = requests.post(
            SERVE_URL,
            headers={"Content-Type": "application/json"},
            json={"inputs": X.to_dict(orient="records")},
            timeout=10,
        )
        if not resp.ok:
            raise HTTPException(status_code=502, detail=f"Model server error: {resp.text}")

        body = resp.json()
        y = body["predictions"] if isinstance(body, dict) and "predictions" in body else body
        y = np.asarray(y, dtype=float)

        # 2) Local SHAP with the raw booster
        shap_vals = explainer.shap_values(X)  # [n_rows, n_features]

        results = []
        thr = get_threshold()
        k = get_top_k()
        for i, row in enumerate(events):
            event_id = str(row.get("event_id", f"evt_{i}"))
            score = float(y[i])

            # compute top-k first (the RL policy may use feature names)
            topk = topk_shap(shap_vals[i], FEATURE_COLS, k)
            topk_names = [t["feature"] for t in topk]

            # RL decision (with fallback)
            decision = decide_with_decider(score, topk_names)
            if decision == "skip":
                # Optional: treat "skip" as batch (or drop). Here we store & route as batch.
                decision = "batch"

            persist(event_id, score, decision, topk)

            results.append({
                "event_id": event_id,
                "score": score,
                "decision": decision,
                "top_k": topk
            })

        return {"results": results, "threshold": thr}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
