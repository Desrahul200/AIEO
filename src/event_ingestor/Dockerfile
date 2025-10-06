FROM python:3.11-slim
WORKDIR /app
COPY event_ingestor.py .
RUN pip install kafka-python psycopg2-binary redis
CMD ["python", "event_ingestor.py"]
