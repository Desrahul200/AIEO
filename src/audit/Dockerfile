# src/audit/Dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY audit_service.py .
RUN pip install --no-cache-dir fastapi==0.118.0 uvicorn[standard]==0.37.0 sqlalchemy==2.0.43 psycopg2-binary==2.9.9
ENV PORT=8080
CMD ["uvicorn", "audit_service:app", "--host", "0.0.0.0", "--port", "8080"]
