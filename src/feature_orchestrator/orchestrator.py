# src/feature_orchestrator/orchestrator.py
import os, json, time, logging, math
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

from kafka import KafkaConsumer, KafkaProducer
import redis
from sqlalchemy import create_engine, text
import mlflow

# ---------- Config ----------
KAFKA_BROKERS        = os.getenv("KAFKA_BROKERS", "kafka:9092")
IN_TOPIC             = os.getenv("IN_TOPIC", "events.low_latency")
OUT_TOPIC            = os.getenv("OUT_TOPIC", "features.computed")
GROUP_ID             = os.getenv("GROUP_ID", "feature_orchestrator_v1")

USE_REDIS            = os.getenv("USE_REDIS", "1") == "1"
REDIS_URL            = os.getenv("REDIS_URL", "redis://redis:6379/0")

USE_PG               = os.getenv("USE_PG", "1") == "1"
PG_DSN               = os.getenv("PG_DSN", "postgresql+psycopg2://user:pass@postgres:5432/mlflow_db")

# Gate for conditional compute
FEATURE_THRESHOLD    = float(os.getenv("FEATURE_THRESHOLD", "0.8"))
IMPORTANT_FEATURES   = [s.strip() for s in os.getenv(
    "IMPORTANT_FEATURES", "event_type_purchase,event_type_signup"
).split(",") if s.strip()]

# Metrics
USE_PROMETHEUS       = os.getenv("USE_PROMETHEUS", "1") == "1"
PROM_PORT            = int(os.getenv("PROM_PORT", "9100"))

# MLflow (optional)
MLFLOW_TRACKING_URI  = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow_server:5000")
MLFLOW_EXPERIMENT    = os.getenv("MLFLOW_EXPERIMENT_NAME", "Phase4_FeatureOrchestrator")

logging.basicConfig(level=logging.INFO, format='[Orchestrator] %(message)s')


# ---------- Helpers ----------
def safe_json(v: bytes):
    try:
        if not v:
            return None
        s = v.decode("utf-8").strip()
        if not s:
            return None
        return json.loads(s)
    except Exception:
        return None

def should_compute(score: float, top_k: List[Dict[str, Any]]) -> bool:
    if score >= FEATURE_THRESHOLD:
        return True
    # If any important feature appears in top_k by name, compute
    feats = {d.get("feature") for d in top_k if isinstance(d, dict)}
    return any(f in feats for f in IMPORTANT_FEATURES)

def to_epoch_ms() -> int:
    return int(time.time() * 1000)

def extract_raw_event(conn, event_id: str) -> Optional[Dict[str, Any]]:
    """
    Router worker stored raw event into events_scored(event_id, event JSONB).
    """
    try:
        row = conn.execute(
            text("SELECT event FROM events_scored WHERE event_id = :eid"),
            {"eid": event_id}
        ).mappings().first()
        return row["event"] if row else None
    except Exception as e:
        logging.warning(f"fetch raw event failed for {event_id}: {e}")
        return None

def compute_features(raw_ev: Dict[str, Any], score: float, top_k: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Keep it simple & deterministic for demo:
      - passthrough: user_id, event_type
      - temporal: hour_of_day (if timestamp present)
      - model: score
      - explainability summary: important_topk (names only)
    """
    features: Dict[str, Any] = {}

    if raw_ev:
        features["user_id"] = raw_ev.get("user_id")
        features["event_type"] = raw_ev.get("event_type")
        ts = raw_ev.get("timestamp")
        if isinstance(ts, (int, float)) and ts > 0:
            dt = datetime.fromtimestamp(ts, tz=timezone.utc)
            features["hour_of_day"] = dt.hour
        else:
            features["hour_of_day"] = None
    else:
        # fallback if raw missing
        features["user_id"] = None
        features["event_type"] = None
        features["hour_of_day"] = None

    features["score"] = float(score)
    # Names of top-K features
    features["important_topk"] = [d["feature"] for d in top_k if "feature" in d][:5]

    return features


# ---------- Clients ----------
consumer = KafkaConsumer(
    IN_TOPIC,
    bootstrap_servers=KAFKA_BROKERS,
    value_deserializer=safe_json,
    key_deserializer=lambda v: v.decode("utf-8") if v else None,
    auto_offset_reset="earliest",
    enable_auto_commit=True,
    group_id=GROUP_ID,
    fetch_min_bytes=1024,
    fetch_max_wait_ms=50,
)

producer = KafkaProducer(
    bootstrap_servers=KAFKA_BROKERS,
    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    key_serializer=lambda v: v.encode("utf-8") if v else None,
    linger_ms=20,
    batch_size=64 * 1024,
)

rds = redis.from_url(REDIS_URL) if USE_REDIS else None

pg = create_engine(PG_DSN, pool_pre_ping=True) if USE_PG else None
if pg:
    with pg.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS features_computed (
              event_id    VARCHAR(64) PRIMARY KEY,
              features    JSONB,
              latency_ms  DOUBLE PRECISION,
              created_at  TIMESTAMP DEFAULT NOW(),
              updated_at  TIMESTAMP DEFAULT NOW()
            )
        """))


# ---------- Metrics ----------
if USE_PROMETHEUS:
    from prometheus_client import Counter, Histogram, start_http_server
    start_http_server(PROM_PORT)
    MET_COMPUTED = Counter("features_computed_total", "Computed features count")
    MET_SKIPPED  = Counter("features_skipped_total", "Skipped feature computation count")
    MET_LAT_MS   = Histogram("feature_latency_ms", "Feature computation latency (ms)")
else:
    MET_COMPUTED = MET_SKIPPED = MET_LAT_MS = None

# MLflow init (log heartbeats)
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT)
mlflow_run = mlflow.start_run(run_name="feature_orchestrator", tags={"phase": "4"}, nested=False)
mlflow.log_param("feature_threshold", FEATURE_THRESHOLD)
mlflow.log_param("important_features", ",".join(IMPORTANT_FEATURES))


# ---------- Main Loop ----------
def handle_message(msg_val: Dict[str, Any]):
    """
    Input message from events.low_latency:
      { "event_id", "score", "decision", "top_k": [...] }
    """
    t0 = time.time()
    event_id = msg_val.get("event_id")
    score    = float(msg_val.get("score", 0.0))
    top_k    = msg_val.get("top_k", []) or []

    if not event_id:
        logging.debug("skipping: no event_id")
        return "skip"

    if not should_compute(score, top_k):
        if MET_SKIPPED: MET_SKIPPED.inc()
        return "skip"

    raw_ev = None
    if pg:
        with pg.begin() as conn:
            raw_ev = extract_raw_event(conn, event_id)

    features = compute_features(raw_ev, score, top_k)

    # sinks
    latency_ms = (time.time() - t0) * 1000.0
    if MET_LAT_MS: MET_LAT_MS.observe(latency_ms)

    if rds:
        rds.set(f"feat:{event_id}", json.dumps(features), ex=3600)

    if pg:
        with pg.begin() as conn:
            conn.execute(text("""
                INSERT INTO features_computed (event_id, features, latency_ms)
                VALUES (:eid, CAST(:feat AS JSONB), :lat)
                ON CONFLICT (event_id) DO UPDATE
                SET features = EXCLUDED.features,
                    latency_ms = EXCLUDED.latency_ms,
                    updated_at = NOW()
            """), {"eid": event_id, "feat": json.dumps(features), "lat": float(latency_ms)})

    # emit downstream
    producer.send(OUT_TOPIC, key=event_id, value={
        "event_id": event_id,
        "features": features,
        "computed": True,
        "latency_ms": latency_ms
    })
    return "computed"


def main():
    logging.info(f"Orchestrating from {IN_TOPIC} → {OUT_TOPIC} | thr={FEATURE_THRESHOLD} | important={IMPORTANT_FEATURES}")
    last_log = time.time()
    computed, skipped = 0, 0

    try:
        for msg in consumer:
            val = msg.value
            if not isinstance(val, dict):
                logging.debug("skip: non-json")
                continue

            result = handle_message(val)
            if result == "computed":
                computed += 1
                if MET_COMPUTED: MET_COMPUTED.inc()
            elif result == "skip":
                skipped += 1

            # periodic MLflow metrics
            now = time.time()
            if now - last_log >= 10:
                mlflow.log_metric("computed_count", computed, step=to_epoch_ms())
                mlflow.log_metric("skipped_count", skipped, step=to_epoch_ms())
                last_log = now

    except KeyboardInterrupt:
        pass
    finally:
        producer.flush()
        mlflow.end_run()

if __name__ == "__main__":
    main()
