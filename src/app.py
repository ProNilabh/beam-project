import os
from datetime import datetime
from typing import List

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from scipy.stats import ks_2samp
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sqlalchemy import create_engine, text

# Config
MODEL_PATH = os.getenv("MODEL_PATH", "/app/models/best_model.pkl")
DATA_PATH = os.getenv("DATA_PATH", "/app/data/ENB2012_data.xlsx")
POSTGRES_URI = os.getenv(
    "POSTGRES_URI",
    "postgresql://beam_user:beam_pass@postgres:5432/beam_monitoring",
)
DRIFT_ALERT_THRESHOLD = 0.20  # mean KS statistic above this → flag drift

FEATURES = ["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8"]
TARGETS = ["Y1", "Y2"]

# App startup — load model and reference data once
app = FastAPI(title="BEAM — Building Energy Assessment Model", version="3.0")
model = joblib.load(MODEL_PATH)
reference_df = pd.read_excel(DATA_PATH).dropna()
print(f"Loaded model from {MODEL_PATH}")
print(f"Loaded reference data: {len(reference_df)} rows from {DATA_PATH}")

# Schemas
class BuildingFeatures(BaseModel):
    X1: float
    X2: float
    X3: float
    X4: float
    X5: float
    X6: float
    X7: float
    X8: float


class BatchRow(BaseModel):
    X1: float
    X2: float
    X3: float
    X4: float
    X5: float
    X6: float
    X7: float
    X8: float
    Y1: float  # actual heating load
    Y2: float  # actual cooling load


class BatchPayload(BaseModel):
    drift_level: float = 0.0
    rows: List[BatchRow]

# Endpoints
@app.get("/")
def root():
    return {
        "service": "BEAM",
        "status": "ok",
        "model_path": MODEL_PATH,
        "reference_rows": len(reference_df),
    }


@app.post("/predict")
def predict(features: BuildingFeatures):
    X = np.array([[getattr(features, f) for f in FEATURES]])
    y = model.predict(X)[0]
    return {"heating_load": float(y[0]), "cooling_load": float(y[1])}


@app.post("/log_batch")
def log_batch(payload: BatchPayload):
    if len(payload.rows) == 0:
        raise HTTPException(status_code=400, detail="Empty batch")

    batch_df = pd.DataFrame([r.dict() for r in payload.rows])
    X = batch_df[FEATURES].values
    y_true = batch_df[TARGETS].values

    y_pred = model.predict(X)

    r2 = float(r2_score(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    heating_r2 = float(r2_score(y_true[:, 0], y_pred[:, 0]))
    cooling_r2 = float(r2_score(y_true[:, 1], y_pred[:, 1]))

    ks_stats = [
        ks_2samp(reference_df[col].values, batch_df[col].values).statistic
        for col in FEATURES
    ]
    drift_score = float(np.mean(ks_stats))
    drift_alert = drift_score >= DRIFT_ALERT_THRESHOLD

    engine = create_engine(POSTGRES_URI)
    timestamp = datetime.utcnow()

    with engine.begin() as conn:
        result = conn.execute(text("SELECT COALESCE(MAX(batch_id), 0) + 1 FROM model_metrics"))
        batch_id = int(result.scalar())

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
                "n_samples": len(batch_df),
                "r2": r2,
                "rmse": rmse,
                "mae": mae,
                "heating_r2": heating_r2,
                "cooling_r2": cooling_r2,
                "drift_level": float(payload.drift_level),
                "drift_score": drift_score,
            },
        )

        rows = []
        for i in range(len(batch_df)):
            rows.append(
                {
                    "batch_id": batch_id,
                    "timestamp": timestamp,
                    "actual_heating": float(y_true[i, 0]),
                    "predicted_heating": float(y_pred[i, 0]),
                    "heating_error": float(y_pred[i, 0] - y_true[i, 0]),
                    "actual_cooling": float(y_true[i, 1]),
                    "predicted_cooling": float(y_pred[i, 1]),
                    "cooling_error": float(y_pred[i, 1] - y_true[i, 1]),
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

    return {
        "batch_id": batch_id,
        "n_samples": len(batch_df),
        "r2": r2,
        "rmse": rmse,
        "mae": mae,
        "heating_r2": heating_r2,
        "cooling_r2": cooling_r2,
        "drift_level": float(payload.drift_level),
        "drift_score": drift_score,
        "drift_alert": drift_alert,
        "drift_threshold": DRIFT_ALERT_THRESHOLD,
    }
