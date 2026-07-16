FROM python:3.11-slim

WORKDIR /app

# Install dependencies first for better layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

EXPOSE 8000

# The autoscaler runs in-process; horizontal scaling is driven by the /metrics
# endpoint (e.g. a Kubernetes HPA or KEDA scaler) in deploy/.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
