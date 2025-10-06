import json
import time
import random
from kafka import KafkaProducer
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='[Generator] %(message)s')

EVENT_RATE = float(os.getenv('EVENT_RATE', '1000'))
SLEEP_TIME = 1.0 / EVENT_RATE

producer = KafkaProducer(
    bootstrap_servers='kafka:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

events = ["click", "view", "purchase", "signup"]

def generate_event():
    return {
        "user_id": random.randint(1, 1000),
        "event_type": random.choice(events),
        "timestamp": time.time()
    }

if __name__ == "__main__":
    while True:
        event = generate_event()
        producer.send("user_events", event)
        producer.flush()
        logging.info(f"Sent event: {event}")
        time.sleep(SLEEP_TIME)
