# src/feature_orchestrator/Dockerfile
FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1

# System deps for psycopg2-binary may not be needed, but add curl for debug
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl tini && \
    rm -rf /var/lib/apt/lists/*

COPY orchestrator.py /app/orchestrator.py

RUN pip install --no-cache-dir \
    kafka-python redis psycopg2-binary sqlalchemy mlflow prometheus-client

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "orchestrator.py"]
