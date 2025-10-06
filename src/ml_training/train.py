# src/ml_training/train.py
import os
import json
import time
import logging
import warnings

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
import mlflow
import mlflow.xgboost
import mlflow.pyfunc

from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from mlflow.models.signature import infer_signature

from shap_utils import compute_shap  # expects summary plot + global dict

# ---------------------------
# Logging & seeds
# ---------------------------
logging.basicConfig(level=logging.INFO, format='[MLTraining] %(message)s')
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Silence some noisy warnings
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["PYTHONWARNINGS"] = "ignore"

# ---------------------------
# MLflow setup
# ---------------------------
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow_server:5000")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "Phase2_Experiment_v2")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

# ---------------------------
# Postgres connection info
# ---------------------------
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "postgres")
POSTGRES_DB = os.getenv("POSTGRES_DB", "mlflow_db")
POSTGRES_USER = os.getenv("POSTGRES_USER", "user")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "pass")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")

# ---------------------------
# Artifacts
# ---------------------------
ART_DIR = "artifacts"
os.makedirs(ART_DIR, exist_ok=True)
SHAP_PLOT_PATH = os.path.join(ART_DIR, "shap_summary_plot.png")
SHAP_JSON_PATH = os.path.join(ART_DIR, "global_shap.json")
FEATURE_COLS_PATH = os.path.join(ART_DIR, "feature_columns.json")
CONF_MAT_PATH = os.path.join(ART_DIR, "confusion_matrix.csv")

DATA_QUERY = "SELECT user_id, event_type, timestamp FROM events LIMIT 100000"


def fetch_data():
    """Fetch a sample of events from Postgres with retry."""
    max_retries = 10
    retry_delay = 5
    for attempt in range(1, max_retries + 1):
        try:
            engine = create_engine(
                f"postgresql+psycopg2://{POSTGRES_USER}:{POSTGRES_PASSWORD}"
                f"@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
            )
            df = pd.read_sql(DATA_QUERY, engine)
            logging.info(f"Successfully fetched {len(df)} rows from Postgres")
            return df
        except Exception as e:
            logging.warning(f"Attempt {attempt}: Postgres connection failed: {e}")
            if attempt == max_retries:
                logging.error("Max retries reached. Exiting.")
                raise
            time.sleep(retry_delay)


def make_features(df: pd.DataFrame):
    """
    Minimal, safe features for first pass:
    - one-hot encode event_type only (avoid user_id leakage)
    """
    if "event_type" not in df.columns:
        raise ValueError("Expected column 'event_type' not found in dataframe.")

    X = pd.get_dummies(df[["event_type"]], drop_first=True)
    y = (df["event_type"] == "purchase").astype(int)

    # Guard: ensure both classes exist for stratify
    stratify = y if y.nunique() > 1 else None
    if stratify is None:
        logging.warning("Only one class present in target; splitting without stratify.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=stratify
    )
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    model = xgb.XGBClassifier(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
        n_jobs=0,            # let XGBoost auto-decide
        tree_method="hist",
    )
    model.fit(X_train, y_train)
    return model


# ---------------------------
# PyFunc wrapper to serve probabilities
# ---------------------------
class XGBProbaWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, booster):
        self.booster = booster

    def predict(self, context, model_input):
        # return probability of class 1
        proba = self.booster.predict_proba(model_input)
        # ensure 1D list/array for MLflow pyfunc REST
        return proba[:, 1]


def main():
    # ---------------------------
    # Data
    # ---------------------------
    df = fetch_data()
    X_train, X_test, y_train, y_test = make_features(df)

    # ---------------------------
    # Train & metrics
    # ---------------------------
    model = train_model(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    accuracy = float(accuracy_score(y_test, y_pred))
    pos_rate = float(y_test.mean())
    try:
        auc = float(roc_auc_score(y_test, y_proba))
    except Exception:
        auc = float("nan")

    cm = confusion_matrix(y_test, y_pred)
    np.savetxt(CONF_MAT_PATH, cm, delimiter=",", fmt="%d")
    logging.info(f"Metrics → accuracy: {accuracy:.4f} | auc: {auc:.4f} | pos_rate: {pos_rate:.4f}")

    # ---------------------------
    # SHAP (global)
    # ---------------------------
    shap_values, explainer, shap_dict = compute_shap(model, X_test, output_path=SHAP_PLOT_PATH)
    with open(SHAP_JSON_PATH, "w") as f:
        json.dump(shap_dict, f)

    # Feature columns (schema for inference)
    feature_cols = X_train.columns.tolist()
    with open(FEATURE_COLS_PATH, "w") as f:
        json.dump(feature_cols, f)

    # Sanity: files exist
    for p in (SHAP_PLOT_PATH, SHAP_JSON_PATH, FEATURE_COLS_PATH):
        assert os.path.exists(p), f"Missing artifact: {p}"
    # Save raw booster for easy retrieval (top-level)
    booster_path = os.path.join(ART_DIR, "booster.json")
    model.get_booster().save_model(booster_path)
    

    # ---------------------------
    # MLflow Logging + Registry
    # ---------------------------
    with mlflow.start_run(run_name="xgb_event_classifier"):
        # Params & metrics
        mlflow.log_params(model.get_params())
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("auc", auc)
        mlflow.log_metric("pos_rate", pos_rate)
        mlflow.log_artifact(booster_path)
        # Artifacts
        mlflow.log_artifact(SHAP_PLOT_PATH)
        mlflow.log_artifact(SHAP_JSON_PATH)
        mlflow.log_artifact(FEATURE_COLS_PATH)
        mlflow.log_artifact(CONF_MAT_PATH)

        # Signature + input example for probability output
        signature = infer_signature(X_test, y_proba)
        input_example = X_test.iloc[:3]

        # Log the raw XGBoost model (optional; returns class labels via pyfunc)
        mlflow.xgboost.log_model(
            model,
            artifact_path="model_xgb",
            signature=signature,
            input_example=input_example,
        )

        # Log a PyFunc wrapper that **returns probabilities** and register it
        wrapped = XGBProbaWrapper(model)
        mlflow.pyfunc.log_model(
            artifact_path="model",                        # keep path "model" for serving
            python_model=wrapped,
            signature=signature,
            input_example=input_example,
            registered_model_name="event_classifier",     # this creates a new version
        )

        # Tags
        mlflow.set_tags(
            {
                "phase": "2",
                "featureset": "basic_onehot_v1",
                "label_def": "event_type==purchase",
                "data_query": DATA_QUERY,
                "random_state": str(RANDOM_STATE),
                "serving_output": "probability_class_1",
            }
        )

        logging.info("✅ Logged artifacts + registered proba PyFunc model as 'event_classifier'")

    logging.info("Done.")


if __name__ == "__main__":
    main()
