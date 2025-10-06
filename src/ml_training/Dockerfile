# src/ml_training/Dockerfile
FROM python:3.11-slim

# system deps
RUN apt-get update && apt-get install -y build-essential libpq-dev && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

# envs injected by docker-compose
ENV PYTHONUNBUFFERED=1

CMD ["python", "train.py"]
