# src/ml_training/scripts/smoke_test.py
import os, json, pandas as pd, requests, mlflow
from mlflow.tracking import MlflowClient

MODEL_NAME = "event_classifier"
ALIAS = "staging"
SERVE_URL = "http://localhost:6000/invocations"

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
REGISTRY_URI = os.getenv("MLFLOW_REGISTRY_URI", TRACKING_URI)

print(f"Using MLflow tracking: {TRACKING_URI}")
print(f"Using MLflow registry: {REGISTRY_URI}")
mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_registry_uri(REGISTRY_URI)

client = MlflowClient(tracking_uri=TRACKING_URI, registry_uri=REGISTRY_URI)

# 1) Resolve alias -> version -> run_id
mv = client.get_model_version_by_alias(MODEL_NAME, ALIAS)
print(f"Resolved alias '{ALIAS}' -> version {mv.version}, run_id {mv.run_id}")

# 2) Download feature_columns.json (top-level in your run)
local_path = client.download_artifacts(mv.run_id, "feature_columns.json")
cols = json.load(open(local_path))

# 3) Build a minimal valid payload
row = {c: 0 for c in cols}
for c in cols:
    if "event_type_" in c:
        row[c] = 1
        break
X = pd.DataFrame([row])

# 4) Call the served model
resp = requests.post(
    SERVE_URL,
    headers={"Content-Type": "application/json"},
    json={"inputs": X.to_dict(orient="records")}
)
print("Status:", resp.status_code)
print("Body:", resp.text[:300])
