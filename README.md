# BEAM — Building Energy Assessment Model

Reproducible MLOps pipeline for predicting heating and cooling loads of residential buildings, with end-to-end Docker deployment and live monitoring.

**Dataset:** UCI Energy Efficiency (Tsanas & Xifara, 2012) — 768 buildings, 8 features, 2 targets (Heating Load, Cooling Load).

---

## Project Phases

| Phase | Scope |
|---|---|
| **Part 1** | EDA, feature engineering, baseline modelling |
| **Part 2** | Model comparison (7 regressors), MLflow tracking, FastAPI inference service |
| **Part 3** | Dockerize the full pipeline, monitoring with Postgres + Adminer + Grafana, live drift detection |

This repository covers **Part 3** as the integrated final deliverable.

---

## Architecture

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  beam-train │───▶│  best_model │───▶│  beam-api   │
│  (Prefect)  │    │   .pkl      │    │  (FastAPI)  │
└─────────────┘    └─────────────┘    └──────┬──────┘
       │                                     │ POST /log_batch
       ▼                                     │  (drift + metrics)
 ┌─────────────┐                             ▼
 │  beam-mlflow│                      ┌─────────────┐
 │  (UI :5000) │                      │  postgres   │
 └─────────────┘                      └──────┬──────┘
                                             │
                          ┌──────────────────┼──────────────┐
                          ▼                  ▼              ▼
                   ┌────────────┐    ┌────────────┐  ┌────────────┐
                   │  adminer   │    │  grafana   │  │ beam-monitor│
                   │  (:8080)   │    │  (:3000)   │  │  (Prefect)  │
                   └────────────┘    └────────────┘  └────────────┘
```

**7 Docker services**, one `docker-compose up --build`.

---

## Quick Start

### Prerequisites
- Docker Desktop running
- ~4 GB free RAM
- Python 3.11+ (only if running scripts outside Docker)

### Run everything

```bash
git clone <this-repo>
cd beam-project
docker-compose up --build
```

First build takes 5–15 minutes (image download + dependency install). Subsequent runs are ~30 seconds.

### Open the UIs

| Service | URL | Credentials |
|---|---|---|
| FastAPI Swagger | http://localhost:8000/docs | — |
| MLflow | http://localhost:5000 | — |
| Adminer | http://localhost:8080 | System=`PostgreSQL`, Server=`postgres`, User=`beam_user`, Password=`beam_pass`, DB=`beam_monitoring` |
| Grafana | http://localhost:3000 | `admin` / `beam` |

In Grafana → **Dashboards → BEAM — Model Monitoring Dashboard**.

---

## Live Demo: Send a Batch and Watch Grafana Update

After the stack is up and the initial monitoring run has populated 10 batches, run:

```bash
# No drift — model performs well
python -m monitoring.simulate_batch --drift 0.0

# Moderate drift — metrics start degrading
python -m monitoring.simulate_batch --drift 0.2

# Heavy drift — clear performance drop, drift_alert flips to true
python -m monitoring.simulate_batch --drift 0.5
```

Each call POSTs a synthetic batch to `POST /log_batch`, which:
1. Runs predictions through the deployed model.
2. Computes regression metrics (R², RMSE, MAE, per-target R²).
3. Measures data drift via the **Kolmogorov-Smirnov statistic** averaged across all 8 features.
4. Raises `drift_alert=true` when `drift_score ≥ 0.20`.
5. Persists everything to Postgres.

Refresh Grafana → the new datapoint appears within seconds.

---

## Data Drift: How It Works

For each incoming feature column, we run the two-sample KS test against the training distribution. The KS statistic is the maximum vertical distance between the two empirical CDFs — bounded in `[0, 1]`. A value near 0 means identical distributions; near 1 means completely separated. We average the statistic across the 8 features to get a single `drift_score`.

**What it tells you:**
- If `drift_score` rises while `R²` drops, the model's degradation is explained by input distribution shift — retraining with fresh data will likely recover performance.
- If `R²` drops but `drift_score` is flat, the issue is concept drift (relationship between X and y changed) — retraining alone may not be enough.

The simulated `drift_level` parameter is the *injected* noise scale; `drift_score` is the *measured* drift. They correlate strongly, which validates the detector.

---

## Repo Structure

```
beam-project/
├── data/
│   └── ENB2012_data.xlsx
├── src/
│   ├── app.py                  # FastAPI service (predict + log_batch)
│   ├── prefect_train.py        # Training pipeline (Prefect flow)
│   └── ...
├── monitoring/
│   ├── monitor.py              # Initial monitoring sweep (10 batches)
│   ├── simulate_batch.py       # Live demo: send one batch
│   └── init_db.sql             # Postgres schema
├── grafana/
│   ├── provisioning/           # Auto-loaded datasource + dashboard config
│   └── dashboards/
│       └── beam_monitoring.json
├── models/                     # Trained model artifacts (gitignored)
├── mlruns/                     # MLflow runs (gitignored)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## Useful Commands

```bash
docker-compose ps                          # service status
docker-compose logs -f beam-api            # follow API logs
docker-compose logs --tail=50 beam-train   # last 50 lines of training
docker-compose down                        # stop everything
docker-compose down -v                     # stop + wipe Postgres + Grafana data
docker-compose up --build beam-api         # rebuild a single service
```

---

## References

- Tsanas, A. & Xifara, A. (2012). *Accurate quantitative estimation of energy performance of residential buildings using statistical machine learning tools.* Energy and Buildings.
- Chip Huyen — *Designing Machine Learning Systems*
- DataTalksClub — MLOps Zoomcamp
