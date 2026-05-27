# BEAM — Building Energy Assessment Model

Reproducible MLOps pipeline for predicting heating and cooling loads of residential buildings. End-to-end Docker deployment, live drift monitoring, and GitHub Actions CI/CD with Docker image publishing.

**Dataset:** UCI Energy Efficiency (Tsanas & Xifara, 2012) — 768 buildings, 8 features, 2 targets (Heating Load, Cooling Load).

---

## Project Phases

| Phase | Scope |
|---|---|
| **Part 1** | EDA, feature engineering, baseline modelling |
| **Part 2** | Model comparison (7 regressors), MLflow tracking, FastAPI inference service |
| **Part 3** | Dockerize the full pipeline, monitoring with Postgres + Adminer + Grafana, live drift detection |
| **Part 4** | CI/CD pipeline with GitHub Actions — automated testing, schema integrity, Docker build, and image publishing to ghcr.io |

This repository delivers all four parts as one integrated stack.

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

See `BEAM_MLOps_Pipeline_Flowchart.png` for the full visual flow.

---

## Quick Start

### Prerequisites
- Docker Desktop running
- ~4 GB free RAM
- Python 3.11+ (only if running scripts outside Docker)

### Run everything

```bash
git clone https://github.com/ProNilabh/beam-project.git
cd beam-project
docker-compose up --build
```

First build takes 5–15 minutes (image download + dependency install). Subsequent runs are ~30 seconds.

### Or pull the pre-built image

Instead of building locally, you can pull the published image from GitHub Container Registry — automatically published by the CD pipeline on every push to `main`:

```bash
docker pull ghcr.io/pronilabh/beam-project:latest
```

### Open the UIs

| Service | URL | Credentials |
|---|---|---|
| FastAPI Swagger | http://localhost:8000/docs | — |
| MLflow | http://localhost:5000 | — |
| Adminer | http://localhost:8080 | System=`PostgreSQL`, Server=`postgres`, User=`beam_user`, Password=`beam_pass`, DB=`beam_monitoring` |
| Grafana | http://localhost:3000 | `admin` / `beam` |

In Grafana → **Dashboards → BEAM — Model Monitoring Dashboard**.

For a step-by-step walkthrough including troubleshooting, see `RUN_GUIDE.md`.

---

## Live Demo: Send a Batch and Watch Grafana Update

After the stack is up and the initial monitoring run has populated 10 batches, run:

```bash
python -m monitoring.simulate_batch --drift 0.0
python -m monitoring.simulate_batch --drift 0.2
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

## CI/CD Pipeline (Part 4)

Two GitHub Actions workflows run automatically — CI on every push and pull request, CD on every push to `main`.

### CI: Quality Gates (`.github/workflows/ci.yml`)

Three jobs run in parallel on every push and PR:

| Job | What it does | Catches |
|---|---|---|
| **test** | Lints code with `ruff`, runs unit tests on drift detection, batch generation, and cross-module consistency | Silent bugs from refactors that touch one module but not its peers |
| **schema** | Spins up a real Postgres 15 service container, applies `init_db.sql`, runs the exact INSERT statements the application code uses | Schema drift between `init_db.sql` and `monitor.py` / `app.py` |
| **docker** | Builds the Docker image and runs an import smoke test inside the container | Dependency conflicts, broken imports, build failures |

### CD: Image Publishing (`.github/workflows/cd.yml`)

After CI passes on `main`, the CD workflow:
1. Builds the Docker image using the same build pipeline as CI
2. Tags it three ways: `latest`, the short commit SHA, and the branch name
3. Pushes it to GitHub Container Registry (`ghcr.io/pronilabh/beam-project`)

Every commit on `main` produces a pullable, reproducible image — anyone can `docker pull` any version by tag or commit SHA.

### What's actually tested

The tests deliberately target **integration boundaries** — the places bugs hide in a multi-service ML system:

- **`FEATURES` list consistency** across all four Python modules (`app.py`, `prefect_train.py`, `monitor.py`, `simulate_batch.py`). If they drift, predictions silently become garbage. Parsed via AST so tests don't require heavy ML libraries.
- **`docker-compose.yml` volume paths** all point to real files (catches renamed dashboards, deleted SQL).
- **Postgres schema** matches the code's INSERT statements column-for-column.
- **KS drift detector** behaves correctly: zero drift → low score, heavy drift → high score, monotonically increasing, bounded in [0, 1], no crashes on edge cases (constant features, single-row batches).
- **Batch generation** produces correct size, all required columns, preserves distribution shape with zero drift, widens it with high drift, and produces a payload that matches the FastAPI Pydantic contract.
- **Docker image smoke test** verifies the built image can locate all four core modules — catches missing files, import errors, and dependency conflicts before the image ever runs in production.

### Run the tests locally

```bash
pip install pytest pyyaml ruff
pytest tests/ -v
```

You'll see 27 passing tests; the 3 schema tests skip locally (they need a live Postgres) and run only in CI where the service container provides one.

### Why this design

Most CI/CD demos test trivial things — a single `assert 2 + 2 == 4`. This pipeline tests the actual failure modes of a multi-service ML system:

> *"I went line-by-line through my three pipelines — training, monitoring, and inference — and identified where they break silently. The FEATURES list is duplicated across four files, so I test it's consistent. The Postgres schema is referenced by both the code and the Grafana dashboard, so I spin up a real Postgres in CI and verify the INSERT statements match the schema. The Docker build job confirms the image still builds and all modules import. CD then publishes the verified image to ghcr.io — every commit on main is automatically tested, built, and made pullable by SHA. The tests target integration boundaries — that's where bugs actually live."*

### Demonstrating the pipeline catches bugs

To prove the CI works rather than just runs, follow the **red → green** demo in `RUN_GUIDE.md`. Push a deliberately broken commit (e.g., remove `X8` from the `FEATURES` list in one file), watch CI turn red, fix it, push again, watch CI turn green and CD publish the corrected image.

---

## Repo Structure

```
beam-project/
├── .github/
│   └── workflows/
│       ├── ci.yml                        # CI: lint, test, schema, Docker build
│       └── cd.yml                        # CD: publish image to ghcr.io
├── data/
│   ├── ENB2012_data.xlsx                 # UCI dataset
│   ├── ENB2012_data.csv
│   └── README.txt
├── grafana/
│   ├── dashboards/
│   │   └── beam_monitoring.json          # Auto-loaded 7-panel dashboard
│   └── provisioning/
│       ├── dashboards/dashboard.yml
│       └── datasources/datasource.yml
├── monitoring/
│   ├── __init__.py
│   ├── init_db.sql                       # Postgres schema
│   ├── monitor.py                        # Initial monitoring sweep (Prefect)
│   └── simulate_batch.py                 # Live demo script
├── src/
│   ├── __init__.py
│   ├── app.py                            # FastAPI inference + /log_batch
│   └── prefect_train.py                  # Training pipeline (Prefect)
├── tests/
│   ├── __init__.py
│   ├── test_consistency.py               # Cross-module + file existence checks
│   ├── test_drift.py                     # KS statistic behaviour
│   ├── test_batch.py                     # Batch generation + payload shape
│   └── test_schema.py                    # Postgres schema integrity (CI-only)
├── models/                               # Trained model artifacts (gitignored)
├── mlruns/                               # MLflow runs (gitignored)
├── .dockerignore
├── .gitignore
├── BEAM_MLOps_Pipeline_Flowchart.png     # Architecture diagram
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── README.md
└── RUN_GUIDE.md                          # Step-by-step setup walkthrough
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

pytest tests/ -v                           # run the test suite locally
docker pull ghcr.io/pronilabh/beam-project:latest   # pull the latest published image
```

---

## References

- Tsanas, A. & Xifara, A. (2012). *Accurate quantitative estimation of energy performance of residential buildings using statistical machine learning tools.* Energy and Buildings.
- Chip Huyen — *Designing Machine Learning Systems*
- DataTalksClub — MLOps Zoomcamp