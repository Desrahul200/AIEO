import os, time, json, logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import numpy as np
from mlflow.tracking import MlflowClient
from stable_baselines3 import PPO

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow_server:5000")
RL_POLICY_URI = os.getenv("RL_POLICY_URI")  # e.g. runs:/<run_id>/rl_policy/policy.zip

logging.basicConfig(level=logging.INFO, format='[RLDecider] %(message)s')
app = FastAPI(title="RL Decision Service", version="0.1.0")

client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
policy = None

def _download(artifact_uri: str) -> str:
    # supports runs:/<run_id>/<path>
    if artifact_uri.startswith("runs:/"):
        _, tail = artifact_uri.split("runs:/", 1)
        run_id, rel = tail.split("/", 1)
        return client.download_artifacts(run_id, rel)
    # you can add handling for models:/ if you later register it
    raise ValueError(f"Unsupported RL_POLICY_URI: {artifact_uri}")

def load_policy():
    global policy
    if not RL_POLICY_URI:
        raise RuntimeError("RL_POLICY_URI not set")
    local = _download(RL_POLICY_URI)
    policy = PPO.load(local)
    logging.info(f"Loaded RL policy from {RL_POLICY_URI}")

load_policy()

class RouteReq(BaseModel):
    score: float
    topk_names: List[str] = []
    hour: int | None = None
    q_ll: float = 0.0
    q_batch: float = 0.0
    ma_latency: float = 0.0

@app.get("/health")
def health():
    return {"ok": True, "policy": bool(policy)}

@app.post("/route")
def route(r: RouteReq):
    try:
        hour = r.hour if r.hour is not None else int((time.time()/3600) % 24)
        has_purchase = 1.0 if "event_type_purchase" in r.topk_names else 0.0
        has_signup   = 1.0 if "event_type_signup" in r.topk_names else 0.0
        obs = np.array([[float(r.score),
                         float(hour),
                         1.0 if hour in (0,6,12,18) else 0.0,
                         has_purchase,
                         has_signup,
                         float(r.q_ll),
                         float(r.q_batch),
                         float(r.ma_latency)]], dtype=np.float32)
        act, _ = policy.predict(obs, deterministic=True)
        decision = "low_latency" if int(act)==1 else ("batch" if int(act)==0 else "skip")
        return {"decision": decision, "action": int(act)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
