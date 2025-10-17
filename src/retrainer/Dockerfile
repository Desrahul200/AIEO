# src/retrainer/Dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY retrainer.py .
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir xgboost==2.1.1 scikit-learn==1.5.2 pandas==2.2.3 numpy==1.26.4 shap==0.45.1 mlflow==3.4.0 sqlalchemy==2.0.43 psycopg2-binary==2.9.9 requests==2.32.3
ENV PYTHONUNBUFFERED=1
CMD ["python", "retrainer.py"]
