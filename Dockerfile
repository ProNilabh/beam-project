# ── BEAM — Building Energy Assessment with ML ───────────────────────────────
# Single Dockerfile for the entire pipeline:
#   - Training (python -m src.prefect_train)
#   - API serving (uvicorn src.app:app)
#   - Monitoring (python -m monitoring.monitor)
#
# The CMD is overridden per service in docker-compose.yml

FROM python:3.11-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc curl \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies (cached layer — only rebuilds if requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire project
COPY . .

# Expose ports: 8000=FastAPI, 5000=MLflow
EXPOSE 8000 5000

# Default: run the FastAPI web service
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
