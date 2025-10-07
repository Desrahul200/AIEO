FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PYTHONDONTWRITEBYTECODE=1

WORKDIR /app
COPY decision_service.py .

# small runtime dep for numpy/torch
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip

# Match your working local versions:
# - SB3 2.7.0
# - gymnasium 1.2.1
# - shimmy 2.0.0
# Use CPU Torch wheel from PyTorch index to keep size reasonable.
RUN pip install \
    "fastapi>=0.110,<1.0" \
    "uvicorn[standard]>=0.22,<1.0" \
    "stable-baselines3==2.7.0" \
    "gymnasium==1.2.1" \
    "shimmy==2.0.0" \
    "numpy==2.3.3" \
    "mlflow-skinny==3.4.0" \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    "torch==2.3.1"

ENV PORT=8080
CMD ["uvicorn", "decision_service:app", "--host", "0.0.0.0", "--port", "8080"]
