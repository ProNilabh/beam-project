import os
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from prefect import flow, task
from sqlalchemy import create_engine, text
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import ks_2samp

# Config
POSTGRES_URI = os.getenv(
    "POSTGRES_URI",
    "postgresql://beam_user:beam_pass@localhost:5432/beam_monitoring",
)
MODEL_PATH = os.getenv("MODEL_PATH", "/app/models/best_model.pkl")
DATA_PATH = os.getenv("DATA_PATH", "/app/data/ENB2012_data.xlsx")

FEATURES = ["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8"]
TARGETS = ["Y1", "Y2"]  # Heating Load, Cooling Load

# Tasks
@task
def load_artifacts():
    """Load trained model and reference (training) data."""
    model = joblib.load(MODEL_PATH)
    df = pd.read_excel(DATA_PATH)
    df = df.dropna()
    return model, df


@task
def generate_batch(reference_df: pd.DataFrame, batch_size: int, drift_level: float):
    sample = reference_df.sample(n=batch_size, replace=True).copy().reset_index(drop=True)
    if drift_level > 0:
        for col in FEATURES:
            std = reference_df[col].std()
            sample[col] = sample[col] + np.random.normal(0, drift_level * std, batch_size)
    return sample


@task
def measure_drift(reference_df: pd.DataFrame, batch_df: pd.DataFrame) -> float:
    ks_stats = []
    for col in FEATURES:
        stat, _ = ks_2samp(reference_df[col].values, batch_df[col].values)
        ks_stats.append(stat)
    return float(np.mean(ks_stats))


@task
def evaluate(model, batch_df: pd.DataFrame):
    X = batch_df[FEATURES]
    y_true = batch_df[TARGETS].values
    y_pred = model.predict(X)

    r2 = r2_score(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    heating_r2 = r2_score(y_true[:, 0], y_pred[:, 0])
    cooling_r2 = r2_score(y_true[:, 1], y_pred[:, 1])

    return {
        "r2": float(r2),
        "rmse": rmse,
        "mae": mae,
        "heating_r2": float(heating_r2),
        "cooling_r2": float(cooling_r2),
        "y_true": y_true,
        "y_pred": y_pred,
    }


@task
def write_to_postgres(batch_id: int, metrics: dict, drift_level: float, drift_score: float):
    engine = create_engine(POSTGRES_URI)
    timestamp = datetime.utcnow()

    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO model_metrics
                (batch_id, timestamp, model_name, n_samples, r2, rmse, mae,
                 heating_r2, cooling_r2, drift_level, drift_score)
                VALUES
                (:batch_id, :timestamp, :model_name, :n_samples, :r2, :rmse, :mae,
                 :heating_r2, :cooling_r2, :drift_level, :drift_score)
                """
            ),
            {
                "batch_id": batch_id,
                "timestamp": timestamp,
                "model_name": "best_model",
                "n_samples": len(metrics["y_true"]),
                "r2": metrics["r2"],
                "rmse": metrics["rmse"],
                "mae": metrics["mae"],
                "heating_r2": metrics["heating_r2"],
                "cooling_r2": metrics["cooling_r2"],
                "drift_level": drift_level,
                "drift_score": drift_score,
            },
        )

        rows = []
        for i in range(len(metrics["y_true"])):
            rows.append(
                {
                    "batch_id": batch_id,
                    "timestamp": timestamp,
                    "actual_heating": float(metrics["y_true"][i, 0]),
                    "predicted_heating": float(metrics["y_pred"][i, 0]),
                    "heating_error": float(metrics["y_pred"][i, 0] - metrics["y_true"][i, 0]),
                    "actual_cooling": float(metrics["y_true"][i, 1]),
                    "predicted_cooling": float(metrics["y_pred"][i, 1]),
                    "cooling_error": float(metrics["y_pred"][i, 1] - metrics["y_true"][i, 1]),
                }
            )

        conn.execute(
            text(
                """
                INSERT INTO prediction_log
                (batch_id, timestamp, actual_heating, predicted_heating, heating_error,
                 actual_cooling, predicted_cooling, cooling_error)
                VALUES
                (:batch_id, :timestamp, :actual_heating, :predicted_heating, :heating_error,
                 :actual_cooling, :predicted_cooling, :cooling_error)
                """
            ),
            rows,
        )

# Flow
@flow(name="BEAM_Monitoring_Pipeline")
def monitoring_pipeline(
    n_batches: int = 10,
    batch_size: int = 50,
    drift_schedule: list = None,
):
    if drift_schedule is None:
        drift_schedule = [0, 0, 0, 0.05, 0.1, 0.1, 0.15, 0.2, 0.3, 0.5]

    if len(drift_schedule) != n_batches:
        raise ValueError(
            f"drift_schedule length ({len(drift_schedule)}) must equal n_batches ({n_batches})"
        )

    print(f"Loading model and reference data...")
    model, reference_df = load_artifacts()
    print(f"Reference data: {len(reference_df)} rows")

    for i in range(n_batches):
        drift_level = drift_schedule[i]
        print(f"\n--- Batch {i + 1}/{n_batches} (drift_level={drift_level}) ---")

        batch_df = generate_batch(reference_df, batch_size, drift_level)
        drift_score = measure_drift(reference_df, batch_df)
        metrics = evaluate(model, batch_df)
        write_to_postgres(batch_id=i + 1, metrics=metrics, drift_level=drift_level, drift_score=drift_score)

        print(
            f"  R²={metrics['r2']:.4f}  RMSE={metrics['rmse']:.4f}  "
            f"MAE={metrics['mae']:.4f}  drift_score={drift_score:.4f}"
        )

    print("\nMonitoring pipeline complete.")


if __name__ == "__main__":
    monitoring_pipeline(
        n_batches=10,
        batch_size=50,
        drift_schedule=[0, 0, 0, 0.05, 0.1, 0.1, 0.15, 0.2, 0.3, 0.5],
    )
