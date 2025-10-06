# src/event_generator/generator.py
import json, time, random, os, logging, uuid
from kafka import KafkaProducer

logging.basicConfig(level=logging.INFO, format='[Generator] %(message)s')

EVENT_RATE = float(os.getenv("EVENT_RATE", "1000"))
SLEEP_TIME = 1.0 / EVENT_RATE

producer = KafkaProducer(
    bootstrap_servers="kafka:9092",
    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    linger_ms=20,           # batch a bit
    batch_size=64 * 1024,   # 64KB
)

EVENT_TYPES = ["click", "view", "purchase", "signup"]

def generate_event():
    return {
        "event_id": str(uuid.uuid4()),            # <- helpful across the pipeline
        "user_id": random.randint(1, 1000),
        "event_type": random.choice(EVENT_TYPES),
        "timestamp": time.time(),
    }

if __name__ == "__main__":
    while True:
        ev = generate_event()
        producer.send("user_events", ev)  # no per-message flush
        logging.info(f"Sent event: {ev}")
        time.sleep(SLEEP_TIME)
