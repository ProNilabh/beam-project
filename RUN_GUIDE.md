# BEAM — Step-by-Step Run Guide

Complete commands to get the BEAM MLOps project running from scratch, plus CI/CD setup.

---

## Part 1: Run the Full Project Stack

### Prerequisites

Make sure you have these installed:

1. **Docker Desktop** — running. Whale icon in tray must show "Docker Desktop is running."
2. **Python 3.11+** — for the `simulate_batch.py` script (the rest runs in Docker).
3. **Git** — for pushing to GitHub.

Optional but recommended:
- Conda environment `nlp_env` activated for the local Python parts.

### Step 1: Extract and enter the project

```cmd
cd C:\Users\nilab\Documents
```

Unzip `beam-project.zip` here. You should have:
```
C:\Users\nilab\Documents\beam-project\
```

```cmd
cd beam-project
```

### Step 2: Start Docker Desktop

Wait until the whale icon shows "Docker Desktop is running." This takes ~30 seconds.

### Step 3: Build and start the stack

```cmd
docker-compose up --build
```

**What happens:**
- First run: pulls images + builds custom image (5-15 min)
- Subsequent runs: ~30 seconds
- Training, monitoring, and serving all start automatically

**You'll see logs like:**
```
beam-train   | Loaded 768 rows from /app/data/ENB2012_data.xlsx
beam-train   | Best model: XGBoost (R²=0.9952)
beam-monitor | --- Batch 10/10 (drift_level=0.5) ---
beam-monitor |   R²=-0.0446  drift_score=0.2099
```

Once you see `beam-monitor` exit cleanly and `beam-api` listening on 8000, the stack is ready.

### Step 4: Open the dashboards

In your browser, open these tabs:

| Service | URL | Login |
|---|---|---|
| **FastAPI** | http://localhost:8000/docs | none |
| **MLflow** | http://localhost:5000 | none |
| **Adminer** | http://localhost:8080 | System=`PostgreSQL`, Server=`postgres`, User=`beam_user`, Pass=`beam_pass`, DB=`beam_monitoring` |
| **Grafana** | http://localhost:3000/d/beam-monitoring | `admin` / `beam` |

In Grafana, set the time range to **"Last 7 days"** in the top right.

### Step 5: Test the live demo

Open a **second terminal** (keep the docker-compose one running).

```cmd
cd C:\Users\nilab\Documents\beam-project
conda activate nlp_env
pip install requests openpyxl scipy pandas numpy
```

Then send batches:

```cmd
python -m monitoring.simulate_batch --drift 0.0
python -m monitoring.simulate_batch --drift 0.2
python -m monitoring.simulate_batch --drift 0.5
python -m monitoring.simulate_batch --drift 0.8
```

After each, refresh Grafana — the new batch appears at the right edge of every chart.

### Step 6: Shut down cleanly

In the docker-compose terminal, press `Ctrl+C` once. Then:

```cmd
docker-compose down
```

This keeps your data (postgres, mlflow, models). To wipe everything fresh:
```cmd
docker-compose down -v
```

---

## Part 2: Push to GitHub

### Step 1: Initialize git (if not already)

```cmd
cd C:\Users\nilab\Documents\beam-project
git init
git branch -M main
```

### Step 2: Add files and commit

```cmd
git add .
git commit -m "BEAM Part 3 - Full MLOps stack with CI/CD"
```

### Step 3: Create the GitHub repo

1. Go to https://github.com/new
2. Repository name: `beam-project`
3. Public or private (recommend public so Forooz doesn't need invite)
4. **Do not** initialize with README — we have one
5. Click "Create repository"

### Step 4: Push

GitHub will show you commands like these:

```cmd
git remote add origin https://github.com/YOUR-USERNAME/beam-project.git
git push -u origin main
```

Replace `YOUR-USERNAME` with your GitHub username and run them.

### Step 5: Verify

Open `https://github.com/YOUR-USERNAME/beam-project` — you should see all files.

---

## Part 3: Run the CI/CD Pipeline

The CI files are already in `.github/workflows/ci.yml`. As soon as you push, GitHub Actions runs automatically.

### Step 1: View the CI run

1. Go to your repo on GitHub
2. Click the **"Actions"** tab at the top
3. You'll see your push triggered a workflow run
4. Click on it — three jobs run in parallel:
   - **test** — lint + unit tests (consistency, drift, batch)
   - **schema** — Postgres schema integrity
   - **docker** — Docker image build + import smoke test

All three should turn green ✅.

### Step 2: Test the CI locally first (optional but recommended)

Before pushing, you can run the tests on your machine:

```cmd
cd C:\Users\nilab\Documents\beam-project
conda activate nlp_env
pip install pytest pyyaml ruff
pytest tests/test_consistency.py tests/test_drift.py tests/test_batch.py -v
```

Expected output:
```
============================== 27 passed in 1.48s ==============================
```

The schema test will skip locally (it needs Postgres) but will run in CI.

### Step 3: Demo CI catching a bug (for the presentation)

This is the impressive demo for Forooz. Show that CI actually catches real bugs:

1. **Break something on purpose.** Edit `monitoring/simulate_batch.py`:
   ```python
   FEATURES = ["X1", "X2", "X3", "X4", "X5", "X6", "X7"]
   ```
   (Removed X8.)

2. **Commit + push:**
   ```cmd
   git add monitoring/simulate_batch.py
   git commit -m "Demo: break feature list to show CI catches it"
   git push
   ```

3. Watch GitHub Actions → it turns **red ❌**. The `test_feature_lists_consistent_across_modules` test fails.

4. **Fix it:**
   ```python
   FEATURES = ["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8"]
   ```

5. **Commit + push:**
   ```cmd
   git add monitoring/simulate_batch.py
   git commit -m "Fix: restore X8"
   git push
   ```

6. CI turns green ✅ again.

This red → green → green cycle is the most convincing CI/CD demo you can show.

---

## Part 4: Quick Troubleshooting

### "Docker is not running"
Start Docker Desktop. Wait 30 seconds. Try again.

### "Port 5432/8000/3000 already in use"
Something else is using the port. Find and stop it:
```cmd
netstat -ano | findstr :5432
taskkill /PID <pid> /F
```

### "beam-api keeps restarting"
The model file isn't there. Check `beam-train` logs:
```cmd
docker-compose logs beam-train
```
Should end with "Best model: XGBoost (R²=0.9952)". If not, the training failed.

### "Grafana shows No data"
Time range issue. In the top-right of the dashboard, click the time picker and select **"Last 7 days"**.

### "simulate_batch.py says ConnectionError"
The API isn't ready yet. Run `docker-compose ps` and confirm `beam-api` is `Up`. Wait if it just started.

### Tests fail locally but pass in CI (or vice versa)
The schema tests need Postgres. Local pytest skips them automatically. CI provides Postgres as a service container.

### "ModuleNotFoundError: No module named 'X'"
Install missing packages:
```cmd
pip install pytest pyyaml ruff requests openpyxl scipy pandas numpy
```

---

## Project Layout

```
beam-project/
├── .github/
│   └── workflows/
│       └── ci.yml                        # CI/CD pipeline
├── data/
│   ├── ENB2012_data.xlsx                 # UCI dataset
│   ├── ENB2012_data.csv                  # Same data in CSV
│   └── README.txt
├── grafana/
│   ├── dashboards/
│   │   └── beam_monitoring.json          # 7-panel dashboard
│   └── provisioning/
│       ├── dashboards/
│       │   └── dashboard.yml
│       └── datasources/
│           └── datasource.yml
├── monitoring/
│   ├── __init__.py
│   ├── init_db.sql                       # Postgres schema
│   ├── monitor.py                        # Prefect monitoring sweep
│   └── simulate_batch.py                 # Live demo script
├── src/
│   ├── __init__.py
│   ├── app.py                            # FastAPI service
│   └── prefect_train.py                  # Prefect training pipeline
├── tests/
│   ├── __init__.py
│   ├── test_consistency.py               # Cross-module consistency
│   ├── test_drift.py                     # KS test unit tests
│   ├── test_batch.py                     # Batch generation tests
│   └── test_schema.py                    # Postgres schema integrity
├── .dockerignore
├── .gitignore
├── BEAM_MLOps_Pipeline_Flowchart.png
├── Dockerfile
├── docker-compose.yml
├── README.md
├── RUN_GUIDE.md                          # This file
└── requirements.txt
```

---

## What Each CI Test Catches

| Test | Bug class it prevents |
|---|---|
| `test_feature_lists_consistent_across_modules` | Silent prediction garbage when FEATURES list drifts between the 4 modules |
| `test_dataset_has_required_columns` | Dataset corruption or column renames |
| `test_compose_volume_paths_exist` | Renaming/deleting files that docker-compose mounts |
| `test_critical_file_exists` | Accidentally removing core project files |
| `test_no_drift_gives_low_score` | KS detector breaking (giving high scores on clean data) |
| `test_heavy_drift_gives_high_score` | KS detector breaking (missing real drift) |
| `test_drift_score_is_bounded` | Math errors in the KS implementation |
| `test_constant_feature_does_not_crash` | scipy edge case crashes |
| `test_payload_shape_matches_api_contract` | Pydantic schema drift in `/log_batch` |
| `test_model_metrics_insert_matches_schema` | Postgres schema drift between init_db.sql and code |
| Smoke test (docker job) | Dependency conflicts that break imports |

---

## Done

You have:
- ✅ Working 7-service Docker stack
- ✅ Repository on GitHub
- ✅ Automatic CI/CD on every push
- ✅ Tests that catch real bugs (proven by the red → green demo)

For the presentation Thursday, the killer move is the **red → green demo** from Part 3, Step 3. That's what wins.
