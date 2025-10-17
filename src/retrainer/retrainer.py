# src/retrainer/retrainer.py
import os, json, time, logging
import pandas as pd
import numpy as np
import shap
import xgboost as xgb
from sqlalchemy import create_engine, text
import mlflow
from mlflow.tracking import MlflowClient
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

logging.basicConfig(level=logging.INFO, format='[Retrainer] %(message)s')

# --- Config ---
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow_server:5000")
MLFLOW_REGISTRY_URI = os.getenv("MLFLOW_REGISTRY_URI", MLFLOW_TRACKING_URI)
EXPERIMENT_NAME     = os.getenv("RETRAIN_EXPERIMENT", "OnlineRetraining")
MODEL_NAME          = os.getenv("MODEL_NAME", "event_classifier")
PROMOTE_ALIAS       = os.getenv("PROMOTE_ALIAS", "staging")
PROMOTE_IF_BETTER   = os.getenv("PROMOTE_IF_BETTER", "1") == "1"

PG_DSN              = os.getenv("PG_DSN", "postgresql+psycopg2://user:pass@postgres:5432/mlflow_db")
TRAIN_WINDOW_MIN    = int(os.getenv("TRAIN_WINDOW_MIN", "5"))         # lookback (minutes)
MIN_ROWS_TO_TRAIN   = int(os.getenv("MIN_ROWS_TO_TRAIN", "200"))
TEST_SIZE           = float(os.getenv("TEST_SIZE", "0.2"))
RANDOM_STATE        = int(os.getenv("RANDOM_STATE", "42"))

ROUTER_RELOAD_URL   = os.getenv("ROUTER_RELOAD_URL", "http://router:7000/reload")
RETRAIN_INTERVAL_S  = int(os.getenv("RETRAIN_INTERVAL_S", "30"))     # schedule loop

# --- Init MLflow ---
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_registry_uri(MLFLOW_REGISTRY_URI)
client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI, registry_uri=MLFLOW_REGISTRY_URI)
try:
    mlflow.set_experiment(EXPERIMENT_NAME)
except Exception:
    pass

engine = create_engine(PG_DSN, pool_pre_ping=True)

def load_recent_events(minutes: int) -> pd.DataFrame:
    q = text(f"""
        SELECT event_id, event, created_at
        FROM events_scored
        WHERE created_at >= NOW() - INTERVAL '{int(minutes)} minutes'
          AND event IS NOT NULL
        ORDER BY created_at DESC
    """)
    with engine.connect() as conn:
        df = pd.read_sql(q, conn)

    if df.empty:
        return pd.DataFrame(columns=["event_id", "event_type", "created_at"])

    # event may already be a dict (JSONB) or a JSON string; normalize safely
    def _to_dict(x):
        if isinstance(x, dict):
            return x
        if isinstance(x, str) and x.strip():
            try:
                return json.loads(x)
            except Exception:
                return None
        return None

    recs = [ _to_dict(x) for x in df["event"].tolist() ]
    mask = [ isinstance(r, dict) and ("event_type" in r) for r in recs ]
    if not any(mask):
        return pd.DataFrame(columns=["event_id", "event_type", "created_at"])

    ev = pd.DataFrame([recs[i] for i, ok in enumerate(mask) if ok])
    base = df.loc[[i for i, ok in enumerate(mask) if ok], ["event_id", "created_at"]].reset_index(drop=True)
    ev = pd.concat([base, ev], axis=1)

    # keep only what we need; drop rows without event_type
    ev = ev.loc[ev["event_type"].notna(), ["event_id", "event_type", "created_at"]]
    return ev.reset_index(drop=True)


def build_dataset(ev: pd.DataFrame):
    y = (ev["event_type"] == "purchase").astype(int).values
    cats = ["click", "view", "purchase", "signup"]
    ev["event_type"] = pd.Categorical(ev["event_type"], categories=cats)
    X = pd.get_dummies(ev[["event_type"]], drop_first=False)
    # Ensure all expected columns exist (even if absent in the window)
    for c in [f"event_type_{c}" for c in cats]:
        if c not in X.columns: X[c] = 0
    X = X[[f"event_type_{c}" for c in cats]]
    feature_columns = list(X.columns)
    return X, y, feature_columns


def compute_global_shap(model: xgb.XGBClassifier, X: pd.DataFrame, feat_cols):
    explainer = shap.TreeExplainer(model)
    # sample to speed up
    S = X.sample(min(2000, len(X)), random_state=RANDOM_STATE)
    vals = explainer.shap_values(S)  # [n, f]
    mean_abs = np.mean(np.abs(vals), axis=0)
    return [{"feature": feat_cols[i], "mean_abs_shap": float(mean_abs[i])} for i in range(len(feat_cols))]

def current_alias_auc(model_name: str, alias: str) -> float | None:
    try:
        mv = client.get_model_version_by_alias(model_name, alias)
        run = client.get_run(mv.run_id)
        auc = run.data.metrics.get("auc")
        return float(auc) if auc is not None else None
    except Exception:
        return None

def maybe_promote(model_name: str, new_version: str, new_auc: float, alias: str):
    old_auc = current_alias_auc(model_name, alias)
    better = (old_auc is None) or (new_auc > old_auc)
    if PROMOTE_IF_BETTER and better:
        client.set_registered_model_alias(model_name, alias, new_version)
        logging.info(f"Promoted {model_name}@{alias} -> v{new_version} (auc {new_auc:.4f} > {old_auc})")
        return True
    logging.info(f"Not promoting (new_auc={new_auc}, old_auc={old_auc}); alias unchanged.")
    return False

def notify_router_reload():
    import requests
    try:
        r = requests.post(ROUTER_RELOAD_URL, timeout=5)
        if r.ok:
            logging.info("Router reloaded successfully.")
        else:
            logging.warning(f"Router reload status={r.status_code}: {r.text}")
    except Exception as e:
        logging.warning(f"Router reload failed: {e}")

def one_retrain_cycle():
    ev = load_recent_events(TRAIN_WINDOW_MIN)
    logging.info(f"Loaded {len(ev)} recent events for training window={TRAIN_WINDOW_MIN}m")
    if len(ev) < MIN_ROWS_TO_TRAIN:
        logging.info(f"Not enough rows (<{MIN_ROWS_TO_TRAIN}). Skipping.")
        return

    X, y, feature_columns = build_dataset(ev)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

    clf = xgb.XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1, subsample=0.9, colsample_bytree=0.9,
        reg_lambda=1.0, n_jobs=4, random_state=RANDOM_STATE, tree_method="hist"
    )
    clf.fit(Xtr, ytr)
    proba = clf.predict_proba(Xte)[:, 1]
    auc = roc_auc_score(yte, proba)

    # Log to MLflow
    with mlflow.start_run(run_name="online_retrain") as run:
        mlflow.log_param("train_window_min", TRAIN_WINDOW_MIN)
        mlflow.log_param("min_rows_to_train", MIN_ROWS_TO_TRAIN)
        mlflow.log_metric("auc", float(auc))

        # Artifacts
        booster = clf.get_booster()
        booster.save_model("booster.json")
        with open("feature_columns.json", "w") as f:
            json.dump(feature_columns, f)

        # Global SHAP
        shap_global = compute_global_shap(clf, X, feature_columns)
        with open("shap_global.json", "w") as f:
            json.dump(shap_global, f)

        mlflow.log_artifact("booster.json")
        mlflow.log_artifact("feature_columns.json")
        mlflow.log_artifact("shap_global.json")

        # Register
        res = client.create_model_version(
            name=MODEL_NAME,
            source=f"{run.info.artifact_uri}",
            run_id=run.info.run_id
        )
        new_version = res.version
        logging.info(f"Registered {MODEL_NAME} v{new_version} (AUC={auc:.4f})")

        promoted = maybe_promote(MODEL_NAME, new_version, float(auc), PROMOTE_ALIAS)
        if promoted:
            notify_router_reload()

def main_loop():
    while True:
        try:
            one_retrain_cycle()
        except Exception as e:
            logging.exception(f"Retrain cycle failed: {e}")
        time.sleep(RETRAIN_INTERVAL_S)

if __name__ == "__main__":
    main_loop()
