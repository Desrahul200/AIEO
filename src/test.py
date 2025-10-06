# test.py
import psycopg2
import redis
from kafka import KafkaProducer
import mlflow

# -------------------------------
# PostgreSQL Test
# -------------------------------
try:
    conn = psycopg2.connect(
        host="localhost",
        port=5432,
        database="mlflow_db",
        user="user",
        password="pass"
    )
    cur = conn.cursor()
    cur.execute("SELECT 1;")
    print("Postgres test:", cur.fetchone())
    cur.close()
    conn.close()
except Exception as e:
    print("Postgres error:", e)

# -------------------------------
# Redis Test
# -------------------------------
try:
    r = redis.Redis(host='localhost', port=6379, db=0)
    r.set('test_key', 'hello')
    value = r.get('test_key').decode('utf-8')
    print("Redis test:", value)
except Exception as e:
    print("Redis error:", e)

# -------------------------------
# Kafka Test
# -------------------------------
try:
    producer = KafkaProducer(
        bootstrap_servers='localhost:9092',
        value_serializer=lambda v: v if isinstance(v, bytes) else str(v).encode('utf-8')
    )
    producer.send('test_topic', b'hello kafka')
    producer.flush()
    print("Kafka test: message sent")
except Exception as e:
    print("Kafka error:", e)

# -------------------------------
# MLflow Test
# -------------------------------
try:
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("Test_Experiment")
    with mlflow.start_run():
        mlflow.log_param("param1", 5)
        mlflow.log_metric("metric1", 0.99)
    print("MLflow test: run logged")
except Exception as e:
    print("MLflow error:", e)
