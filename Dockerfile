FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Source
COPY src/ /app/src/
COPY monitoring/ /app/monitoring/

# Create artifact dirs (mounted as volumes at runtime)
RUN mkdir -p /app/models /app/mlruns /app/data

# Default — overridden by docker-compose `command:`
CMD ["python", "-m", "src.prefect_train"]
