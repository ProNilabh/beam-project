-- BEAM Monitoring Database Schema
-- Initialized automatically by Postgres Docker container on first run.

CREATE TABLE IF NOT EXISTS model_metrics (
    id              SERIAL PRIMARY KEY,
    batch_id        INTEGER NOT NULL,
    timestamp       TIMESTAMP NOT NULL,
    model_name      TEXT NOT NULL,
    n_samples       INTEGER NOT NULL,
    r2              DOUBLE PRECISION,
    rmse            DOUBLE PRECISION,
    mae             DOUBLE PRECISION,
    heating_r2      DOUBLE PRECISION,
    cooling_r2      DOUBLE PRECISION,
    drift_level     DOUBLE PRECISION,   -- simulated drift parameter (0..1)
    drift_score     DOUBLE PRECISION    -- measured drift via KS statistic
);

CREATE TABLE IF NOT EXISTS prediction_log (
    id                  SERIAL PRIMARY KEY,
    batch_id            INTEGER NOT NULL,
    timestamp           TIMESTAMP NOT NULL,
    actual_heating      DOUBLE PRECISION,
    predicted_heating   DOUBLE PRECISION,
    heating_error       DOUBLE PRECISION,
    actual_cooling      DOUBLE PRECISION,
    predicted_cooling   DOUBLE PRECISION,
    cooling_error       DOUBLE PRECISION
);

CREATE INDEX IF NOT EXISTS idx_metrics_timestamp  ON model_metrics(timestamp);
CREATE INDEX IF NOT EXISTS idx_metrics_batch      ON model_metrics(batch_id);
CREATE INDEX IF NOT EXISTS idx_predictions_batch  ON prediction_log(batch_id);
