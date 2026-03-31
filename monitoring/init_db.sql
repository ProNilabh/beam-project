-- BEAM Monitoring Database Initialization
-- Creates tables for model metrics and prediction logs

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
