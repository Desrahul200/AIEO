import json
from kafka import KafkaConsumer
import redis
import psycopg2
import time
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='[Ingestor] %(message)s')

consumer = KafkaConsumer(
    "user_events",
    bootstrap_servers='kafka:9092',
    value_deserializer=lambda v: json.loads(v.decode('utf-8')),
    auto_offset_reset='earliest',
    enable_auto_commit=True,
    group_id='event_ingestor_group'
)

r = redis.Redis(host='redis', port=6379, db=0)

conn = psycopg2.connect(
    host="postgres",
    port=5432,
    database="mlflow_db",
    user="user",
    password="pass"
)
cur = conn.cursor()
cur.execute("""
CREATE TABLE IF NOT EXISTS events (
    user_id INT,
    event_type TEXT,
    timestamp DOUBLE PRECISION
)
""")
conn.commit()

for message in consumer:
    event = message.value
    # Store in Redis
    r.set(f"user:{event['user_id']}:{event['timestamp']}", json.dumps(event))
    logging.info(f"Stored in Redis: {event}")  
    # Store in Postgres
    try:
        cur.execute(
            "INSERT INTO events (user_id, event_type, timestamp) VALUES (%s, %s, %s)",
            (event['user_id'], event['event_type'], event['timestamp'])
        )
        conn.commit()
        logging.info(f"Inserted into Postgres: {event}") 
    except Exception as e:
        logging.error(f"Postgres insert error: {e}")
        conn.rollback()
        time.sleep(1)
