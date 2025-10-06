# src/event_ingestor/ingestor.py
import json, time, logging, redis, psycopg2
from kafka import KafkaConsumer

logging.basicConfig(level=logging.INFO, format='[Ingestor] %(message)s')

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

consumer = KafkaConsumer(
    "user_events",
    bootstrap_servers="kafka:9092",
    value_deserializer=safe_json,   # <- robust
    auto_offset_reset="earliest",
    enable_auto_commit=True,
    group_id="event_ingestor_group",
    # a little batching helps
    fetch_max_wait_ms=50,
    fetch_min_bytes=1_024,
)

r = redis.Redis(host="redis", port=6379, db=0)

conn = psycopg2.connect(host="postgres", port=5432, database="mlflow_db", user="user", password="pass")
conn.autocommit = False
cur = conn.cursor()
cur.execute("""
CREATE TABLE IF NOT EXISTS events (
    user_id INT,
    event_type TEXT,
    timestamp DOUBLE PRECISION
);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events (timestamp);
CREATE INDEX IF NOT EXISTS idx_events_type ON events (event_type);
""")
conn.commit()

batch = []
BATCH_SZ = 500
last = time.time()

for msg in consumer:
    ev = msg.value
    if not isinstance(ev, dict) or "user_id" not in ev or "event_type" not in ev or "timestamp" not in ev:
        logging.warning(f"Skipping invalid event at {msg.topic}:{msg.partition}@{msg.offset}")
        continue

    # Redis (ephemeral key)
    r.set(f"user:{ev['user_id']}:{ev['timestamp']}", json.dumps(ev), ex=3600)

    batch.append((ev["user_id"], ev["event_type"], ev["timestamp"]))

    now = time.time()
    if len(batch) >= BATCH_SZ or (now - last) >= 0.5:
        try:
            cur.executemany(
                "INSERT INTO events (user_id, event_type, timestamp) VALUES (%s, %s, %s)",
                batch
            )
            conn.commit()
            logging.info(f"Inserted {len(batch)} rows into Postgres")
        except Exception as e:
            logging.error(f"Postgres batch insert error: {e}")
            conn.rollback()
        batch.clear()
        last = now
