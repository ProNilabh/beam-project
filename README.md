# BEAM — Building Energy Assessment Model

[![CI](https://github.com/ProNilabh/beam-project/actions/workflows/ci.yml/badge.svg)](https://github.com/ProNilabh/beam-project/actions/workflows/ci.yml)
[![CD](https://github.com/ProNilabh/beam-project/actions/workflows/cd.yml/badge.svg)](https://github.com/ProNilabh/beam-project/actions/workflows/cd.yml)

End-to-end MLOps pipeline that predicts heating and cooling loads of residential buildings. Trains and compares seven regression models, serves the best via FastAPI, monitors live drift through a Postgres + Grafana stack, and ships as a reproducible Docker deployment with GitHub Actions CI/CD.

**Dataset:** UCI Energy Efficiency (Tsanas & Xifara, 2012) — 768 buildings, 8 features, 2 targets (Heating Load, Cooling Load).

**Image:** [ghcr.io/pronilabh/beam-project](https://github.com/ProNilabh/beam-project/pkgs/container/beam-project)

---

## Project Phases

| Phase | Scope |
|---|---|
| **Part 1** | EDA, feature engineering, baseline modelling |
| **Part 2** | Model comparison (7 regressors), MLflow tracking, FastAPI inference service |
| **Part 3** | Dockerized stack, monitoring with Postgres + Adminer + Grafana, live drift detection |
| **Part 4** | CI/CD with GitHub Actions — automated tests, schema integrity, Docker build, image publishing to ghcr.io |

---

## Architecture

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  beam-train │───▶│  best_model │───▶│  beam-api   │
│  (Prefect)  │    │   .pkl      │    │  (FastAPI)  │
└─────────────┘    └─────────────┘    └──────┬──────┘
       │                                     │ POST /log_batch
       ▼                                     ▼
 ┌─────────────┐                      ┌─────────────┐
 │  beam-mlflow│                      │  postgres   │
 │  (UI :5000) │                      └──────┬──────┘
 └─────────────┘                             │
                          ┌──────────────────┼──────────────┐
                          ▼                  ▼              ▼
                   ┌────────────┐    ┌────────────┐  ┌─────────────┐
                   │  adminer   │    │  grafana   │  │ beam-monitor│
                   │  (:8080)   │    │  (:3000)   │  │  (Prefect)  │
                   └────────────┘    └────────────┘  └─────────────┘
```

**7 Docker services**, one `docker-compose up --build`. See `BEAM_MLOps_Pipeline_Flowchart.png` for the full visual flow.

---

## Quick Start

```bash
git clone https://github.com/ProNilabh/beam-project.git
cd beam-project
docker-compose up --build
```

Then open:

| Service | URL | Credentials |
|---|---|---|
| FastAPI | http://localhost:8000/docs | — |
| MLflow | http://localhost:5000 | — |
| Adminer | http://localhost:8080 | Server `postgres`, user `beam_user`, pass `beam_pass`, db `beam_monitoring` |
| Grafana | http://localhost:3000 | `admin` / `beam` |

Full step-by-step setup in `RUN_GUIDE.md`.

---

## Live Demo: Send a Batch and Watch Grafana Update

```bash
python -m monitoring.simulate_batch --drift 0.5
```

Each call POSTs a synthetic batch to `/log_batch`, which runs predictions, computes metrics, measures drift via KS test, and persists to Postgres. Refresh Grafana — the new datapoint appears within seconds.

---

## Data Drift: How It Works

For each incoming feature, we run a two-sample Kolmogorov-Smirnov test against the training distribution. The KS statistic measures the maximum vertical distance between the two empirical CDFs — bounded in `[0, 1]`. We average across the 8 features for a single `drift_score`, and trigger `drift_alert` when it crosses `0.20`.

The injected `drift_level` is what we simulated; `drift_score` is what the detector measured. Their correlation across batches is what validates the system works.

---

## CI/CD Pipeline

Two GitHub Actions workflows run automatically on push:

- **CI** (`ci.yml`) — lint, unit tests, Postgres schema integrity, Docker build smoke test.
- **CD** (`cd.yml`) — on push to `main`: builds and publishes the verified image to GitHub Container Registry.

Tests target integration boundaries — feature-list consistency across the four Python modules, schema-vs-INSERT alignment between `init_db.sql` and the application code, KS detector behaviour, and the payload contract between the simulator and the API. 27 tests total; the 3 schema tests skip locally and activate in CI where a Postgres service container is available.

---

## Repo Structure

```
beam-project/
├── .github/workflows/        # ci.yml, cd.yml
├── data/                     # UCI dataset (xlsx + csv)
├── grafana/                  # Provisioning + dashboard JSON
├── monitoring/               # monitor.py, simulate_batch.py, init_db.sql
├── src/                      # app.py (FastAPI), prefect_train.py
├── tests/                    # 27 tests across 4 files
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── README.md
└── RUN_GUIDE.md              # Step-by-step setup walkthrough
```

---

## Useful Commands

```bash
docker-compose up --build                  # build and start everything
docker-compose ps                          # service status
docker-compose logs -f beam-api            # follow API logs
docker-compose down                        # stop everything
docker-compose down -v                     # stop + wipe Postgres + Grafana data

pytest tests/ -v                           # run the test suite
docker pull ghcr.io/pronilabh/beam-project:latest
```

---

## References

- Tsanas, A. & Xifara, A. (2012). *Accurate quantitative estimation of energy performance of residential buildings using statistical machine learning tools.* Energy and Buildings.
- Chip Huyen — *Designing Machine Learning Systems*
- DataTalksClub — MLOps Zoomcamp

---

## Acknowledgements

A big thank you to **Prof. Forooz Shahbazi** for the guidance, feedback, and well-structured course that made this project possible. The depth of the MLOps requirements — from reproducibility through monitoring to CI/CD — pushed me to engage with production patterns I would not have explored on my own.