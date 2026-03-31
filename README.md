# BEAM — Building Energy Assessment with ML

> Predicting Heating & Cooling Loads Before Construction Begins

**Course:** MLOps: An Introduction — HS Lucerne  
**Student:** Nilabh Pandey | MSCITDS.2501  
**Instructor:** Forooz Shahbazi Avarvand

---

## Overview

BEAM predicts a building's **heating load** and **cooling load** (kWh/m²) from 8 architectural design features — enabling energy-efficient decisions **before construction begins**.

## Dataset

| Property | Value |
|----------|-------|
| Name | UCI Energy Efficiency Dataset |
| Source | [UCI ML Repository](https://archive.ics.uci.edu/dataset/242/energy+efficiency) |
| Rows | 768, zero missing values |
| Origin | Ecotect simulation — Tsanas & Xifara (2012) |

**8 inputs:** Relative Compactness, Surface Area, Wall Area, Roof Area, Overall Height, Orientation, Glazing Area, Glazing Area Distribution  
**2 outputs:** Heating Load (Y1), Cooling Load (Y2)

---

## Project Structure

```
beam-project/
├── data/
│   └── ENB2012_data.csv              # Dataset
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py          # Data loading + scaling
│   ├── train.py                       # MLflow training (standalone)
│   ├── prefect_train.py               # Prefect-orchestrated training pipeline
│   └── app.py                         # FastAPI web service
├── monitoring/
│   ├── __init__.py
│   ├── monitor.py                     # Prefect monitoring pipeline
│   └── init_db.sql                    # Postgres schema
├── grafana/
│   ├── provisioning/
│   │   ├── datasources/datasource.yml # Postgres connection
│   │   └── dashboards/dashboard.yml   # Dashboard auto-provisioning
│   └── dashboards/
│       └── beam_monitoring.json       # Monitoring dashboard
├── models/                            # Generated after training
├── mlruns/                            # MLflow tracking data
├── scripts/test_api.py                # API test script
├── Dockerfile                         # Single image for all services
├── docker-compose.yml                 # Full stack: train + API + monitoring
├── requirements.txt
└── README.md
```

---

## Quick Start — Full Pipeline with Docker

### Prerequisites

- Docker and Docker Compose installed
- The dataset file `data/ENB2012_data.csv` in the data folder

### One Command to Run Everything

```bash
docker-compose up --build
```

This starts 7 services in order:
1. **beam-train** — Trains all models with Prefect + MLflow (exits after completion)
2. **beam-api** — FastAPI web service on port 8000
3. **beam-mlflow** — MLflow UI on port 5000
4. **postgres** — PostgreSQL database for monitoring data
5. **adminer** — Database web UI on port 8080
6. **beam-monitor** — Runs the Prefect monitoring pipeline (exits after completion)
7. **grafana** — Monitoring dashboard on port 3000

### Access Points

| Service | URL | Credentials |
|---------|-----|-------------|
| FastAPI (Swagger) | http://localhost:8000/docs | — |
| MLflow UI | http://localhost:5000 | — |
| Adminer | http://localhost:8080 | Server: `postgres`, User: `beam_user`, Pass: `beam_pass`, DB: `beam_monitoring` |
| Grafana | http://localhost:3000 | User: `admin`, Pass: `beam` |

---

## Running Locally (without Docker)

### 1. Install dependencies

```bash
conda activate nlp_env
pip install -r requirements.txt
```

### 2. Train with Prefect

```bash
python -m src.prefect_train
```

### 3. Start MLflow UI

```bash
mlflow ui --backend-store-uri file:///C:/Users/nilab/Documents/beam-project/mlruns
```

### 4. Start FastAPI

```bash
uvicorn src.app:app --host 0.0.0.0 --port 8000 --reload
```

### 5. Start Postgres + Adminer + Grafana via Docker, then run monitoring

```bash
docker-compose up postgres adminer grafana -d
python -m monitoring.monitor
```

---

## Orchestration with Prefect

Both the training and monitoring pipelines are orchestrated with **Prefect**:

- **Training pipeline** (`src/prefect_train.py`): data loading → preprocessing → model training (7 models, 2 experiments) → best model selection → model saving
- **Monitoring pipeline** (`monitoring/monitor.py`): model loading → batch generation (with simulated drift) → evaluation → metrics storage in Postgres

Each step is a Prefect `@task`, and the full workflow is a Prefect `@flow`.

---

## Monitoring Pipeline

The monitoring pipeline simulates real-world model deployment:

1. **Generates batches** of new data sampled from the original dataset
2. **Adds increasing drift** (noise) to simulate distribution shift over time
3. **Evaluates the model** on each batch
4. **Stores metrics** (R², RMSE, MAE) and individual predictions in Postgres
5. **Grafana dashboard** visualizes performance degradation

Drift schedule: `[0, 0, 0, 0.05, 0.1, 0.1, 0.15, 0.2, 0.3, 0.5]`
→ First 3 batches are clean, then drift gradually increases, showing how model performance degrades.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| GET | `/model-info` | Model metadata |
| POST | `/predict` | Single building prediction |
| POST | `/predict/batch` | Batch predictions |
| GET | `/docs` | Swagger UI |

---

## Reproducibility

The entire pipeline is reproducible via Docker:

```bash
git clone https://github.com/YOUR_USERNAME/beam-mlops.git
cd beam-mlops
# Place ENB2012_data.csv in data/ folder
docker-compose up --build
```

This trains the model from scratch, deploys it, runs monitoring, and populates the dashboard — all from a single command.

---

## References

- Tsanas, A. & Xifara, A. (2012). Accurate quantitative estimation of energy performance of residential buildings using statistical machine learning tools.
- Chip Huyen — *Designing Machine Learning Systems*
- MLOps Zoomcamp
