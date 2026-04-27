"""
BEAM — Prefect Monitoring Pipeline
====================================
Monitors model performance by:
1. Loading the deployed model
2. Simulating new incoming data (with optional drift)
3. Computing performance metrics
4. Storing results in Postgres
5. Repeating over multiple batches to populate the dashboard

Usage:
    cd beam-project
    python -m monitoring.monitor
"""

import os
import sys
import json
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import joblib
from sqlalchemy import create_engine, text
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from prefect import flow, task, get_run_logger

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data_preprocessing import load_data, prepare_data, FEATURE_NAMES, TARGET_NAMES


# ── Configuration ────────────────────────────────────────────────────────────

POSTGRES_URI = os.getenv(
    "POSTGRES_URI",
    "postgresql://beam_user:beam_pass@localhost:5432/beam_monitoring"
)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "ENB2012_data.csv")


# ── Prefect Tasks ────────────────────────────────────────────────────────────

@task(name="init_database")
def init_database():
    """Create the monitoring tables in Postgres if they don't exist."""
    logger = get_run_logger()
    engine = create_engine(POSTGRES_URI)

    create_sql = """
    CREATE TABLE IF NOT EXISTS model_metrics (
        id SERIAL PRIMARY KEY,
        timestamp TIMESTAMP NOT NULL,
        batch_id INTEGER NOT NULL,
        model_name VARCHAR(100),
        n_samples INTEGER,
        r2 FLOAT,
        rmse FLOAT,
        mae FLOAT,
        heating_r2 FLOAT,
        heating_rmse FLOAT,
        cooling_r2 FLOAT,
        cooling_rmse FLOAT,
        drift_level FLOAT DEFAULT 0.0
    );

    CREATE TABLE IF NOT EXISTS prediction_log (
        id SERIAL PRIMARY KEY,
        timestamp TIMESTAMP NOT NULL,
        batch_id INTEGER NOT NULL,
        relative_compactness FLOAT,
        surface_area FLOAT,
        wall_area FLOAT,
        roof_area FLOAT,
        overall_height FLOAT,
        orientation FLOAT,
        glazing_area FLOAT,
        glazing_area_distribution FLOAT,
        actual_heating FLOAT,
        actual_cooling FLOAT,
        predicted_heating FLOAT,
        predicted_cooling FLOAT,
        heating_error FLOAT,
        cooling_error FLOAT
    );
    """

    with engine.connect() as conn:
        for statement in create_sql.strip().split(";"):
            statement = statement.strip()
            if statement:
                conn.execute(text(statement))
        conn.commit()

    logger.info("✅ Database tables ready")
    return engine


@task(name="load_model")
def load_model():
    """Load the deployed model, scaler, and metadata."""
    logger = get_run_logger()

    model = joblib.load(os.path.join(MODEL_DIR, "best_model.joblib"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))

    with open(os.path.join(MODEL_DIR, "metadata.json")) as f:
        metadata = json.load(f)

    logger.info(f"Loaded model: {metadata['model_name']} (R²={metadata['test_r2']})")
    return model, scaler, metadata


@task(name="generate_batch")
def generate_batch(df: pd.DataFrame, batch_size: int = 50, drift_level: float = 0.0):
    """
    Simulate a batch of new data.
    - Samples randomly from the dataset
    - Optionally adds noise (drift) to features to simulate distribution shift
    """
    logger = get_run_logger()

    # Random sample
    batch = df.sample(n=batch_size, replace=True, random_state=None).copy()

    # Add drift (noise) to features — simulates real-world distribution shift
    if drift_level > 0:
        for col in FEATURE_NAMES:
            noise = np.random.normal(0, drift_level * batch[col].std(), size=batch_size)
            batch[col] = batch[col] + noise

    logger.info(f"Generated batch: {batch_size} samples, drift={drift_level:.2f}")
    return batch


@task(name="evaluate_batch")
def evaluate_batch(model, scaler, batch: pd.DataFrame):
    """Run predictions and compute metrics on a batch."""
    logger = get_run_logger()

    X = batch[FEATURE_NAMES].values
    y_actual = batch[TARGET_NAMES].values

    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)

    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 2)

    metrics = {
        "r2": float(r2_score(y_actual, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_actual, y_pred))),
        "mae": float(mean_absolute_error(y_actual, y_pred)),
        "heating_r2": float(r2_score(y_actual[:, 0], y_pred[:, 0])),
        "heating_rmse": float(np.sqrt(mean_squared_error(y_actual[:, 0], y_pred[:, 0]))),
        "cooling_r2": float(r2_score(y_actual[:, 1], y_pred[:, 1])),
        "cooling_rmse": float(np.sqrt(mean_squared_error(y_actual[:, 1], y_pred[:, 1]))),
    }

    logger.info(f"Batch metrics → R²={metrics['r2']:.4f}, RMSE={metrics['rmse']:.4f}")
    return metrics, y_actual, y_pred, X


@task(name="store_metrics")
def store_metrics(engine, metrics: dict, batch_id: int, model_name: str,
                  n_samples: int, drift_level: float, timestamp: datetime):
    """Insert batch metrics into Postgres."""
    logger = get_run_logger()

    insert_sql = text("""
        INSERT INTO model_metrics
            (timestamp, batch_id, model_name, n_samples, r2, rmse, mae,
             heating_r2, heating_rmse, cooling_r2, cooling_rmse, drift_level)
        VALUES
            (:ts, :bid, :mn, :ns, :r2, :rmse, :mae,
             :hr2, :hrmse, :cr2, :crmse, :dl)
    """)

    with engine.connect() as conn:
        conn.execute(insert_sql, {
            "ts": timestamp, "bid": batch_id, "mn": model_name,
            "ns": n_samples, "r2": metrics["r2"], "rmse": metrics["rmse"],
            "mae": metrics["mae"], "hr2": metrics["heating_r2"],
            "hrmse": metrics["heating_rmse"], "cr2": metrics["cooling_r2"],
            "crmse": metrics["cooling_rmse"], "dl": drift_level,
        })
        conn.commit()

    logger.info(f"Stored metrics for batch {batch_id}")


@task(name="store_predictions")
def store_predictions(engine, X, y_actual, y_pred, batch_id: int, timestamp: datetime):
    """Insert individual predictions into Postgres for detailed analysis."""
    logger = get_run_logger()

    rows = []
    for i in range(len(X)):
        rows.append({
            "timestamp": timestamp,
            "batch_id": batch_id,
            "relative_compactness": float(X[i][0]),
            "surface_area": float(X[i][1]),
            "wall_area": float(X[i][2]),
            "roof_area": float(X[i][3]),
            "overall_height": float(X[i][4]),
            "orientation": float(X[i][5]),
            "glazing_area": float(X[i][6]),
            "glazing_area_distribution": float(X[i][7]),
            "actual_heating": float(y_actual[i][0]),
            "actual_cooling": float(y_actual[i][1]),
            "predicted_heating": float(y_pred[i][0]),
            "predicted_cooling": float(y_pred[i][1]),
            "heating_error": float(y_actual[i][0] - y_pred[i][0]),
            "cooling_error": float(y_actual[i][1] - y_pred[i][1]),
        })

    pred_df = pd.DataFrame(rows)
    pred_df.to_sql("prediction_log", engine, if_exists="append", index=False)

    logger.info(f"Stored {len(rows)} predictions for batch {batch_id}")


# ── Prefect Flow ─────────────────────────────────────────────────────────────

@flow(name="BEAM_Monitoring_Pipeline", log_prints=True)
def monitoring_pipeline(
    n_batches: int = 10,
    batch_size: int = 50,
    drift_schedule: list = None,
):
    """
    Monitoring pipeline that simulates batches of new data,
    evaluates the model, and stores results in Postgres.

    drift_schedule: list of drift levels per batch.
        Example: [0, 0, 0, 0.1, 0.1, 0.2, 0.2, 0.3, 0.3, 0.5]
        This simulates gradual data drift over time.
    """

    if drift_schedule is None:
        # Default: starts clean, gradually increases drift
        drift_schedule = [0.0, 0.0, 0.0, 0.05, 0.1, 0.1, 0.15, 0.2, 0.3, 0.5]

    # Pad or trim drift_schedule to match n_batches
    if len(drift_schedule) < n_batches:
        drift_schedule.extend([drift_schedule[-1]] * (n_batches - len(drift_schedule)))
    drift_schedule = drift_schedule[:n_batches]

    print("=" * 60)
    print("  BEAM MONITORING PIPELINE")
    print("=" * 60)

    # Step 1: Init database
    engine = init_database()

    # Step 2: Load model
    model, scaler, metadata = load_model()

    # Step 3: Load reference data
    df = load_data(DATA_PATH)

    # Step 4: Process batches
    base_time = datetime.now() - timedelta(hours=n_batches)

    for batch_id in range(1, n_batches + 1):
        drift_level = drift_schedule[batch_id - 1]
        timestamp = base_time + timedelta(hours=batch_id)

        print(f"\n  Batch {batch_id}/{n_batches} | Drift: {drift_level:.2f} | Time: {timestamp}")

        # Generate batch
        batch = generate_batch(df, batch_size=batch_size, drift_level=drift_level)

        # Evaluate
        metrics, y_actual, y_pred, X = evaluate_batch(model, scaler, batch)

        # Store metrics
        store_metrics(engine, metrics, batch_id, metadata["model_name"],
                      batch_size, drift_level, timestamp)

        # Store individual predictions
        store_predictions(engine, X, y_actual, y_pred, batch_id, timestamp)

    print(f"\n{'='*60}")
    print(f"  ✅ Monitoring complete: {n_batches} batches processed")
    print(f"  📊 View in Adminer: http://localhost:8080")
    print(f"  📈 View in Grafana: http://localhost:3000")
    print(f"{'='*60}")


if __name__ == "__main__":
    monitoring_pipeline(
        n_batches=10,
        batch_size=50,
        drift_schedule=[0, 0, 0, 0.05, 0.1, 0.1, 0.15, 0.2, 0.3, 0.5],
    )