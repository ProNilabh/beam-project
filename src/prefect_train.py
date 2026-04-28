import os
import joblib
import numpy as np
import pandas as pd
from prefect import flow, task

import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor

# Config
DATA_PATH = os.getenv("DATA_PATH", "/app/data/ENB2012_data.xlsx")
MODEL_DIR = os.getenv("MODEL_DIR", "/app/models")
MLFLOW_DIR = os.getenv("MLFLOW_DIR", "/app/mlruns")
EXPERIMENT = "BEAM_Training"

FEATURES = ["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8"]
TARGETS = ["Y1", "Y2"]

# Tasks
@task
def load_data() -> pd.DataFrame:
    """Load and clean the Energy Efficiency dataset."""
    df = pd.read_excel(DATA_PATH)
    df = df.dropna()
    print(f"Loaded {len(df)} rows from {DATA_PATH}")
    return df

@task
def split_data(df: pd.DataFrame):
    """80/20 train/validation split."""
    X = df[FEATURES]
    y = df[TARGETS]
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_val, y_train, y_val

@task
def get_models() -> dict:
    return {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0, random_state=42),
        "Lasso": Lasso(alpha=0.1, random_state=42),
        "DecisionTree": DecisionTreeRegressor(random_state=42),
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
        "GradientBoosting": MultiOutputRegressor(
            GradientBoostingRegressor(n_estimators=100, random_state=42)
        ),
        "XGBoost": MultiOutputRegressor(
            XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
        ),
    }

@task
def train_and_log(name, model, X_train, X_val, y_train, y_val):
    with mlflow.start_run(run_name=name):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        r2 = r2_score(y_val, y_pred)
        rmse = float(np.sqrt(mean_squared_error(y_val, y_pred)))
        heating_r2 = r2_score(y_val.iloc[:, 0], y_pred[:, 0])
        cooling_r2 = r2_score(y_val.iloc[:, 1], y_pred[:, 1])

        mlflow.log_param("model", name)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("heating_r2", heating_r2)
        mlflow.log_metric("cooling_r2", cooling_r2)
        try:
            mlflow.sklearn.log_model(model, name=name)
        except TypeError:
            # older MLflow signature
            mlflow.sklearn.log_model(model, artifact_path=name)

        print(f"  {name:20s}  R²={r2:.4f}  RMSE={rmse:.4f}")
        return {"name": name, "model": model, "r2": r2, "rmse": rmse}

@task
def save_best(results: list):
    """Pick the highest-R² model and persist it to /app/models/best_model.pkl."""
    best = max(results, key=lambda r: r["r2"])
    os.makedirs(MODEL_DIR, exist_ok=True)
    out = os.path.join(MODEL_DIR, "best_model.pkl")
    joblib.dump(best["model"], out)
    print(f"\nBest model: {best['name']} (R²={best['r2']:.4f})")
    print(f"Saved to {out}")
    return best

# Flow
@flow(name="BEAM_Training_Pipeline")
def training_pipeline():
    mlflow.set_tracking_uri(f"file://{MLFLOW_DIR}")
    mlflow.set_experiment(EXPERIMENT)

    df = load_data()
    X_train, X_val, y_train, y_val = split_data(df)
    models = get_models()

    print(f"\nTraining {len(models)} models...\n")
    results = []
    for name, model in models.items():
        results.append(train_and_log(name, model, X_train, X_val, y_train, y_val))

    save_best(results)
    print("\ntraining_pipeline complete")


if __name__ == "__main__":
    training_pipeline()
