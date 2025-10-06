import os, json, time, logging, requests
from typing import List, Dict, Any
from kafka import KafkaConsumer, KafkaProducer
import redis
from sqlalchemy import create_engine, text

# ------------ Config ------------
KAFKA_BROKERS = os.getenv("KAFKA_BROKERS", "kafka:9092")
IN_TOPIC      = os.getenv("IN_TOPIC", "user_events")
FAST_TOPIC    = os.getenv("FAST_TOPIC", "events.low_latency")
BATCH_TOPIC   = os.getenv("BATCH_TOPIC", "events.batch")
GROUP_ID      = os.getenv("GROUP_ID", "router_worker_v1")

ROUTER_URL    = os.getenv("ROUTER_URL", "http://router:7000/score")
BATCH_SIZE    = int(os.getenv("BATCH_SIZE", "64"))
BATCH_TIMEOUT = float(os.getenv("BATCH_TIMEOUT", "0.8"))  # seconds

USE_REDIS     = os.getenv("USE_REDIS", "1") == "1"
REDIS_URL     = os.getenv("REDIS_URL", "redis://redis:6379/0")

USE_PG        = os.getenv("USE_PG", "1") == "1"
PG_DSN        = os.getenv("PG_DSN", "postgresql+psycopg2://user:pass@postgres:5432/mlflow_db")

THRESHOLD     = float(os.getenv("ROUTING_THRESHOLD", "0.6"))  # router also returns its threshold

logging.basicConfig(level=logging.INFO, format='[RouterWorker] %(message)s')

def safe_json(v: bytes):
    try:
        if v is None:
            return None
        s = v.decode("utf-8").strip()
        if not s:
            return None
        return json.loads(s)
    except Exception:
        return None

# ------------ Clients ------------
consumer = KafkaConsumer(
    IN_TOPIC,
    bootstrap_servers=KAFKA_BROKERS,
    value_deserializer=safe_json,
    key_deserializer=lambda v: v.decode("utf-8") if v else None,
    auto_offset_reset="earliest",
    enable_auto_commit=True,
    group_id=GROUP_ID,
)
producer = KafkaProducer(
    bootstrap_servers=KAFKA_BROKERS,
    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    key_serializer=lambda v: v.encode("utf-8") if v else None,
)

rds = None
if USE_REDIS:
    rds = redis.from_url(REDIS_URL)

pg = None
if USE_PG:
    pg = create_engine(PG_DSN, pool_pre_ping=True)
    with pg.begin() as conn:
        # routing decisions table
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS routing_events (
              event_id   VARCHAR(64) PRIMARY KEY,
              score      DOUBLE PRECISION,
              decision   VARCHAR(32),
              topk       JSONB,
              created_at TIMESTAMP DEFAULT NOW(),
              updated_at TIMESTAMP DEFAULT NOW()
            )
        """))
        # optional: store the raw event alongside decision (for audit)
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS events_scored (
              event_id   VARCHAR(64) PRIMARY KEY,
              event      JSONB,
              created_at TIMESTAMP DEFAULT NOW()
            )
        """))

def flush_batch(batch: List[Dict[str, Any]]):
    if not batch:
        return
    payload = {"events": batch}
    try:
        resp = requests.post(
            ROUTER_URL,
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=10,
        )
        if not resp.ok:
            logging.error(f"Router error {resp.status_code}: {resp.text}")
            return
        out = resp.json()
        results = out.get("results", [])
        threshold = out.get("threshold", THRESHOLD)

        for res in results:
            event_id = res["event_id"]
            score    = float(res["score"])
            decision = res["decision"]  # "low_latency" or "batch"
            topk     = res["top_k"]

            # sinks
            if rds:
                rds.hset(f"route:{event_id}", mapping={
                    "score": score, "decision": decision, "topk": json.dumps(topk)
                })

            if pg:
                with pg.begin() as conn:
                    conn.execute(text("""
                        INSERT INTO routing_events (event_id, score, decision, topk)
                        VALUES (:event_id, :score, :decision, CAST(:topk AS JSONB))
                        ON CONFLICT (event_id) DO UPDATE
                        SET score      = EXCLUDED.score,
                            decision   = EXCLUDED.decision,
                            topk       = EXCLUDED.topk,
                            updated_at = NOW()
                    """), {"event_id": event_id, "score": score, "decision": decision, "topk": json.dumps(topk)})

            # route to next topics
            out_msg = {"event_id": event_id, "score": score, "decision": decision, "top_k": topk}
            if decision == "low_latency":
                producer.send(FAST_TOPIC,  key=event_id, value=out_msg)
            else:
                producer.send(BATCH_TOPIC, key=event_id, value=out_msg)

        producer.flush()
        logging.info(f"Flushed {len(results)} events → routed (thr={threshold})")
    except Exception as e:
        logging.exception(f"flush_batch failed: {e}")

def main():
    buf: List[Dict[str, Any]] = []
    last = time.time()

    logging.info(f"Consuming from {IN_TOPIC}, routing via {ROUTER_URL}")
    for msg in consumer:
        ev = msg.value
        if not isinstance(ev, dict) or "event_type" not in ev:
            logging.warning(f"Skipping invalid message at {msg.topic}:{msg.partition}@{msg.offset}")
            continue
        if "event_id" not in ev:
            ev["event_id"] = f"{int(time.time()*1000)}_{msg.offset}"

        buf.append({"event_id": ev["event_id"], "event_type": ev["event_type"]})

        # store raw event (optional audit)
        if pg:
            try:
                with pg.begin() as conn:
                    conn.execute(text("""
                        INSERT INTO events_scored (event_id, event)
                        VALUES (:event_id, CAST(:event AS JSONB))
                        ON CONFLICT (event_id) DO NOTHING
                    """), {"event_id": ev["event_id"], "event": json.dumps(ev)})
            except Exception:
                pass

        now = time.time()
        if len(buf) >= BATCH_SIZE or (now - last) >= BATCH_TIMEOUT:
            flush_batch(buf)
            buf.clear()
            last = now

if __name__ == "__main__":
    main()
