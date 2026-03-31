"""
BEAM — Model Training with MLflow Experiment Tracking
=====================================================
Runs 2 experiments (baseline + tuned), logs everything to MLflow,
and registers the best model in the MLflow Model Registry.

Usage:
    cd beam-project
    python -m src.train
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import joblib

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
)
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

# ── project imports ──────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data_preprocessing import (
    load_data,
    prepare_data,
    FEATURE_NAMES,
    TARGET_NAMES,
)


# ── helpers ──────────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred):
    """Return a flat dict of regression metrics (overall + per-target)."""
    m = {}
    m["rmse"]  = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    m["mae"]   = float(mean_absolute_error(y_true, y_pred))
    m["r2"]    = float(r2_score(y_true, y_pred))
    for i, t in enumerate(TARGET_NAMES):
        m[f"{t}_rmse"] = float(np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i])))
        m[f"{t}_mae"]  = float(mean_absolute_error(y_true[:, i], y_pred[:, i]))
        m[f"{t}_r2"]   = float(r2_score(y_true[:, i], y_pred[:, i]))
    return m


def safe_log_params(params: dict):
    """Log only simple-typed params to MLflow (skip objects)."""
    for k, v in params.items():
        if isinstance(v, (int, float, str, bool, type(None))):
            try:
                mlflow.log_param(k[:250], v)
            except Exception:
                pass


# ── experiment definitions ───────────────────────────────────────────────────

def get_experiments():
    """
    Returns 2 experiments — this satisfies the requirement of
    'at least 2 experiments visible in MLflow'.
    """
    return {
        # ── Experiment 1: Baselines ──────────────────────────────────────
        "BEAM_Experiment_1_Baselines": {
            "Ridge_Regression": MultiOutputRegressor(
                Ridge(alpha=1.0)
            ),
            "Random_Forest_default": MultiOutputRegressor(
                RandomForestRegressor(n_estimators=100, random_state=42)
            ),
            "XGBoost_default": MultiOutputRegressor(
                XGBRegressor(
                    n_estimators=100, random_state=42, verbosity=0
                )
            ),
        },
        # ── Experiment 2: Tuned / advanced ───────────────────────────────
        "BEAM_Experiment_2_Tuned": {
            "Random_Forest_tuned": MultiOutputRegressor(
                RandomForestRegressor(
                    n_estimators=300,
                    max_depth=12,
                    min_samples_split=3,
                    min_samples_leaf=1,
                    random_state=42,
                )
            ),
            "XGBoost_tuned": MultiOutputRegressor(
                XGBRegressor(
                    n_estimators=300,
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    verbosity=0,
                )
            ),
            "Gradient_Boosting_tuned": MultiOutputRegressor(
                GradientBoostingRegressor(
                    n_estimators=300,
                    max_depth=5,
                    learning_rate=0.1,
                    subsample=0.8,
                    random_state=42,
                )
            ),
            "Neural_Network_MLP": MLPRegressor(
                hidden_layer_sizes=(128, 64, 32),
                activation="relu",
                solver="adam",
                max_iter=500,
                early_stopping=True,
                validation_fraction=0.15,
                random_state=42,
                learning_rate_init=0.001,
            ),
        },
    }


# ── main training loop ───────────────────────────────────────────────────────

def run_all_experiments(data_path: str = None):
    """
    Train every model, log to MLflow, pick & save the best one.
    MLflow tracking URI defaults to a local ./mlruns folder.
    """

    # MLflow setup — local file store (Windows-safe: use Path for forward slashes)
    from pathlib import Path
    mlruns_path = Path("mlruns").resolve().as_uri()  # gives file:///C:/Users/... with forward slashes
    mlflow.set_tracking_uri(mlruns_path)
    print(f"MLflow tracking URI: {mlruns_path}\n")

    # ── load data ────────────────────────────────────────────────────────
    df = load_data(data_path)
    X_train, X_test, y_train, y_test, scaler = prepare_data(
        df, test_size=0.2, random_state=42, scale=True
    )
    print(f"Dataset : {df.shape[0]} rows × {df.shape[1]} cols")
    print(f"Train   : {X_train.shape[0]}  |  Test: {X_test.shape[0]}")
    print("=" * 70)

    experiments = get_experiments()
    best_model    = None
    best_r2       = -np.inf
    best_name     = ""
    best_exp      = ""
    best_run_id   = ""
    all_results   = []

    # ── iterate experiments & models ─────────────────────────────────────
    for exp_name, models in experiments.items():
        print(f"\n{'='*70}")
        print(f"  EXPERIMENT : {exp_name}")
        print(f"{'='*70}")

        mlflow.set_experiment(exp_name)

        for model_name, model in models.items():
            print(f"\n  ▸ Training: {model_name} ... ", end="", flush=True)

            with mlflow.start_run(run_name=model_name) as run:

                # log params
                safe_log_params(model.get_params())
                mlflow.log_param("dataset_rows", df.shape[0])
                mlflow.log_param("n_features", 8)
                mlflow.log_param("test_size", 0.2)
                mlflow.log_param("scaling", "StandardScaler")

                # train
                model.fit(X_train, y_train)

                # predict
                y_pred_train = model.predict(X_train)
                y_pred_test  = model.predict(X_test)
                if y_pred_test.ndim == 1:
                    y_pred_test  = y_pred_test.reshape(-1, 2)
                    y_pred_train = y_pred_train.reshape(-1, 2)

                # metrics
                train_m = compute_metrics(y_train, y_pred_train)
                test_m  = compute_metrics(y_test,  y_pred_test)
                for k, v in test_m.items():
                    mlflow.log_metric(f"test_{k}", round(v, 5))
                for k, v in train_m.items():
                    mlflow.log_metric(f"train_{k}", round(v, 5))

                # log the sklearn model artifact
                mlflow.sklearn.log_model(model, artifact_path="model")

                # also log the scaler as an artifact
                scaler_tmp = os.path.join("models", "scaler.joblib")
                os.makedirs("models", exist_ok=True)
                joblib.dump(scaler, scaler_tmp)
                mlflow.log_artifact(scaler_tmp)

                print(f"done  →  Test R² = {test_m['r2']:.4f}")
                print(f"         Heating R²={test_m['Heating_Load_r2']:.4f}  "
                      f"Cooling R²={test_m['Cooling_Load_r2']:.4f}  "
                      f"RMSE={test_m['rmse']:.4f}")

                all_results.append({
                    "experiment": exp_name,
                    "model": model_name,
                    "run_id": run.info.run_id,
                    "test_r2": round(test_m["r2"], 4),
                    "test_rmse": round(test_m["rmse"], 4),
                    "test_mae": round(test_m["mae"], 4),
                    "heating_r2": round(test_m["Heating_Load_r2"], 4),
                    "cooling_r2": round(test_m["Cooling_Load_r2"], 4),
                })

                if test_m["r2"] > best_r2:
                    best_r2     = test_m["r2"]
                    best_model  = model
                    best_name   = model_name
                    best_exp    = exp_name
                    best_run_id = run.info.run_id

    # ── summary ──────────────────────────────────────────────────────────
    results_df = pd.DataFrame(all_results).sort_values("test_r2", ascending=False)
    print(f"\n{'='*70}")
    print("  RESULTS SUMMARY")
    print(f"{'='*70}")
    print(results_df[
        ["experiment", "model", "test_r2", "test_rmse", "heating_r2", "cooling_r2"]
    ].to_string(index=False))

    print(f"\n  🏆 BEST MODEL : {best_name}")
    print(f"     Experiment : {best_exp}")
    print(f"     Run ID     : {best_run_id}")
    print(f"     Test R²    : {best_r2:.4f}")

    # ── save best model locally for FastAPI ───────────────────────────────
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, "models/best_model.joblib")
    joblib.dump(scaler, "models/scaler.joblib")

    metadata = {
        "model_name": best_name,
        "experiment": best_exp,
        "mlflow_run_id": best_run_id,
        "test_r2": round(best_r2, 4),
        "features": FEATURE_NAMES,
        "targets": TARGET_NAMES,
    }
    with open("models/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    results_df.to_csv("models/experiment_results.csv", index=False)

    # ── register best model in MLflow Model Registry ─────────────────────
    model_uri = f"runs:/{best_run_id}/model"
    try:
        reg = mlflow.register_model(model_uri, "BEAM_Best_Model")
        print(f"\n  ✅ Model registered in MLflow as 'BEAM_Best_Model' version {reg.version}")
    except Exception as e:
        print(f"\n  ⚠️  Model Registry note: {e}")
        print("     (This is normal if using file-based tracking. Model is still logged.)")

    print(f"\n  ✅ Best model saved → models/best_model.joblib")
    print(f"  ✅ Scaler saved    → models/scaler.joblib")
    print(f"  ✅ Metadata saved  → models/metadata.json")
    print(f"\n  Next step: run  mlflow ui  then  uvicorn src.app:app --reload")


if __name__ == "__main__":
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "ENB2012_data.csv")
    run_all_experiments(data_path=data_path)
