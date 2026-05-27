import os

import pytest
from sqlalchemy import create_engine, text

POSTGRES_URI = os.getenv(
    "POSTGRES_URI",
    "postgresql://beam_user:beam_pass@localhost:5432/beam_monitoring",
)


def _postgres_available():
    try:
        engine = create_engine(POSTGRES_URI)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _postgres_available(),
    reason="No Postgres available - schema tests require the CI service container",
)


@pytest.fixture(scope="module")
def engine():
    eng = create_engine(POSTGRES_URI)
    raw = open("monitoring/init_db.sql").read()
    cleaned = "\n".join(
        line for line in raw.splitlines() if not line.strip().startswith("--")
    )
    with eng.begin() as conn:
        for statement in cleaned.split(";"):
            if statement.strip():
                conn.execute(text(statement))
    return eng


def test_model_metrics_insert_matches_schema(engine):
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
                "batch_id": 999,
                "timestamp": "2026-01-01 00:00:00",
                "model_name": "ci_test",
                "n_samples": 50,
                "r2": 0.9,
                "rmse": 0.5,
                "mae": 0.3,
                "heating_r2": 0.9,
                "cooling_r2": 0.9,
                "drift_level": 0.1,
                "drift_score": 0.15,
            },
        )
        conn.execute(text("DELETE FROM model_metrics WHERE batch_id = 999"))


def test_prediction_log_insert_matches_schema(engine):
    with engine.begin() as conn:
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
            {
                "batch_id": 999,
                "timestamp": "2026-01-01 00:00:00",
                "actual_heating": 15.0,
                "predicted_heating": 15.5,
                "heating_error": 0.5,
                "actual_cooling": 21.0,
                "predicted_cooling": 20.8,
                "cooling_error": -0.2,
            },
        )
        conn.execute(text("DELETE FROM prediction_log WHERE batch_id = 999"))


def test_required_columns_present(engine):
    dashboard_columns = {
        "batch_id", "timestamp", "r2", "rmse", "mae",
        "heating_r2", "cooling_r2", "drift_level", "drift_score",
    }
    with engine.connect() as conn:
        result = conn.execute(
            text(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'model_metrics'"
            )
        )
        actual_columns = {row[0] for row in result}

    missing = dashboard_columns - actual_columns
    assert not missing, f"Dashboard queries reference missing columns: {missing}"
