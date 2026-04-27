import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from prefect import flow, task, get_run_logger

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data_preprocessing import load_data, prepare_data, FEATURE_NAMES, TARGET_NAMES


# ── Prefect Tasks ────────────────────────────────────────────────────────────

@task(name="load_dataset")
def task_load_data(data_path: str) -> pd.DataFrame:
    """Load the Energy Efficiency dataset."""
    logger = get_run_logger()
    df = load_data(data_path)
    logger.info(f"Loaded dataset: {df.shape[0]} rows × {df.shape[1]} cols")
    return df


@task(name="prepare_dataset")
def task_prepare_data(df: pd.DataFrame):
    """Split and scale the dataset."""
    logger = get_run_logger()
    X_train, X_test, y_train, y_test, scaler = prepare_data(
        df, test_size=0.2, random_state=42, scale=True
    )
    logger.info(f"Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")
    return X_train, X_test, y_train, y_test, scaler


@task(name="compute_metrics")
def task_compute_metrics(y_true, y_pred):
    """Compute regression metrics."""
    m = {}
    m["rmse"] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    m["mae"] = float(mean_absolute_error(y_true, y_pred))
    m["r2"] = float(r2_score(y_true, y_pred))
    for i, t in enumerate(TARGET_NAMES):
        m[f"{t}_rmse"] = float(np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i])))
        m[f"{t}_mae"] = float(mean_absolute_error(y_true[:, i], y_pred[:, i]))
        m[f"{t}_r2"] = float(r2_score(y_true[:, i], y_pred[:, i]))
    return m


@task(name="train_single_model")
def task_train_model(model, model_name: str, X_train, y_train, X_test, y_test, scaler, df_shape):
    """Train one model, log to MLflow, return metrics."""
    logger = get_run_logger()

    with mlflow.start_run(run_name=model_name) as run:
        # Log params
        params = model.get_params()
        for k, v in params.items():
            if isinstance(v, (int, float, str, bool, type(None))):
                try:
                    mlflow.log_param(k[:250], v)
                except Exception:
                    pass
        mlflow.log_param("dataset_rows", df_shape[0])
        mlflow.log_param("n_features", 8)
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("scaling", "StandardScaler")

        # Train
        model.fit(X_train, y_train)

        # Predict
        y_pred_test = model.predict(X_test)
        if y_pred_test.ndim == 1:
            y_pred_test = y_pred_test.reshape(-1, 2)

        # Metrics
        metrics = task_compute_metrics.fn(y_test, y_pred_test)
        for k, v in metrics.items():
            mlflow.log_metric(f"test_{k}", round(v, 5))

        # Log model artifact
        mlflow.sklearn.log_model(model, artifact_path="model")

        # Log scaler
        os.makedirs("models", exist_ok=True)
        scaler_path = os.path.join("models", "scaler.joblib")
        joblib.dump(scaler, scaler_path)
        mlflow.log_artifact(scaler_path)

        logger.info(f"{model_name} → R²={metrics['r2']:.4f} | RMSE={metrics['rmse']:.4f}")

        return {
            "model_name": model_name,
            "model_obj": model,
            "run_id": run.info.run_id,
            "metrics": metrics,
        }


@task(name="select_best_model")
def task_select_best(results: list):
    """Pick the model with highest test R²."""
    logger = get_run_logger()
    best = max(results, key=lambda r: r["metrics"]["r2"])
    logger.info(f"🏆 Best model: {best['model_name']} (R²={best['metrics']['r2']:.4f})")
    return best


@task(name="save_best_model")
def task_save_model(best, scaler):
    """Save best model + scaler + metadata to disk."""
    logger = get_run_logger()
    os.makedirs("models", exist_ok=True)

    joblib.dump(best["model_obj"], "models/best_model.joblib")
    joblib.dump(scaler, "models/scaler.joblib")

    metadata = {
        "model_name": best["model_name"],
        "mlflow_run_id": best["run_id"],
        "test_r2": round(best["metrics"]["r2"], 4),
        "test_rmse": round(best["metrics"]["rmse"], 4),
        "features": FEATURE_NAMES,
        "targets": TARGET_NAMES,
    }
    with open("models/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("✅ Model, scaler, and metadata saved to models/")


# ── Model Definitions ────────────────────────────────────────────────────────

def get_all_models():
    """Return dict of experiment_name → {model_name: model_object}."""
    return {
        "BEAM_Experiment_1_Baselines": {
            "Ridge_Regression": MultiOutputRegressor(Ridge(alpha=1.0)),
            "Random_Forest_default": MultiOutputRegressor(
                RandomForestRegressor(n_estimators=100, random_state=42)
            ),
            "XGBoost_default": MultiOutputRegressor(
                XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
            ),
        },
        "BEAM_Experiment_2_Tuned": {
            "Random_Forest_tuned": MultiOutputRegressor(
                RandomForestRegressor(
                    n_estimators=300, max_depth=12, min_samples_split=3,
                    min_samples_leaf=1, random_state=42
                )
            ),
            "XGBoost_tuned": MultiOutputRegressor(
                XGBRegressor(
                    n_estimators=300, max_depth=6, learning_rate=0.1,
                    subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0
                )
            ),
            "Gradient_Boosting_tuned": MultiOutputRegressor(
                GradientBoostingRegressor(
                    n_estimators=300, max_depth=5, learning_rate=0.1,
                    subsample=0.8, random_state=42
                )
            ),
            "Neural_Network_MLP": MLPRegressor(
                hidden_layer_sizes=(128, 64, 32), activation="relu",
                solver="adam", max_iter=500, early_stopping=True,
                validation_fraction=0.15, random_state=42, learning_rate_init=0.001,
            ),
        },
    }


# ── Prefect Flow ─────────────────────────────────────────────────────────────

@flow(name="BEAM_Training_Pipeline", log_prints=True)
def training_pipeline(data_path: str = ""):
    """
    Full BEAM training pipeline orchestrated by Prefect.
    Trains all models across 2 experiments, logs to MLflow, saves best model.
    """
    if not data_path:
        data_path = os.path.join(os.path.dirname(__file__), "..", "data", "ENB2012_data.csv")

    # MLflow setup
    from pathlib import Path
    mlruns_path = Path("mlruns").resolve().as_uri()
    mlflow.set_tracking_uri(mlruns_path)
    print(f"MLflow tracking URI: {mlruns_path}")

    # Step 1: Load data
    df = task_load_data(data_path)

    # Step 2: Prepare data
    X_train, X_test, y_train, y_test, scaler = task_prepare_data(df)

    # Step 3: Train all models
    experiments = get_all_models()
    all_results = []

    for exp_name, models in experiments.items():
        mlflow.set_experiment(exp_name)
        print(f"\n{'='*60}")
        print(f"  EXPERIMENT: {exp_name}")
        print(f"{'='*60}")

        for model_name, model in models.items():
            result = task_train_model(
                model, model_name, X_train, y_train,
                X_test, y_test, scaler, df.shape
            )
            all_results.append(result)

    # Step 4: Select best model
    best = task_select_best(all_results)

    # Step 5: Save best model
    task_save_model(best, scaler)

    print(f"\n✅ Pipeline complete. Best model: {best['model_name']}")
    return best


if __name__ == "__main__":
    training_pipeline()
