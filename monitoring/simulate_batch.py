"""
BEAM Live Batch Simulator

Generates a synthetic batch (with optional drift) and POSTs it to the
FastAPI /log_batch endpoint. The endpoint runs predictions, measures drift,
and writes everything to Postgres → Grafana picks it up automatically.

Usage:
    # Default — batch of 50 rows, drift level 0.0
    python -m monitoring.simulate_batch

    # With drift
    python -m monitoring.simulate_batch --drift 0.3

    # Custom size and API URL
    python -m monitoring.simulate_batch --size 100 --drift 0.5 --url http://localhost:8000

This is the live-demo script for the presentation: run it during the talk
and the new datapoint appears in Grafana within a few seconds.
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import requests

DATA_PATH = os.getenv("DATA_PATH", "data/ENB2012_data.xlsx")
FEATURES = ["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8"]
TARGETS = ["Y1", "Y2"]


def make_batch(size: int, drift: float) -> pd.DataFrame:
    """Sample from training data, optionally inject Gaussian drift."""
    df = pd.read_excel(DATA_PATH).dropna()
    sample = df.sample(n=size, replace=True).copy().reset_index(drop=True)
    if drift > 0:
        for col in FEATURES:
            std = df[col].std()
            sample[col] = sample[col] + np.random.normal(0, drift * std, size)
    return sample


def main():
    parser = argparse.ArgumentParser(description="Send a simulated batch to BEAM API")
    parser.add_argument("--size", type=int, default=50, help="Batch size (default 50)")
    parser.add_argument(
        "--drift",
        type=float,
        default=0.0,
        help="Drift level — 0=no drift, 0.5=heavy drift (default 0.0)",
    )
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:8000",
        help="FastAPI base URL (default http://localhost:8000)",
    )
    args = parser.parse_args()

    print(f"Generating batch: size={args.size}, drift={args.drift}")
    batch = make_batch(args.size, args.drift)

    payload = {
        "drift_level": args.drift,
        "rows": batch[FEATURES + TARGETS].to_dict(orient="records"),
    }

    print(f"POSTing to {args.url}/log_batch ...")
    try:
        resp = requests.post(f"{args.url}/log_batch", json=payload, timeout=30)
    except requests.exceptions.ConnectionError:
        print(
            f"ERROR: Could not reach {args.url}. Is the FastAPI service running?\n"
            f"       Check with: docker-compose ps"
        )
        sys.exit(1)

    if resp.status_code != 200:
        print(f"ERROR {resp.status_code}: {resp.text}")
        sys.exit(1)

    result = resp.json()
    print("\n=== Batch logged successfully ===")
    print(f"  batch_id     : {result['batch_id']}")
    print(f"  n_samples    : {result['n_samples']}")
    print(f"  R²           : {result['r2']:.4f}")
    print(f"  RMSE         : {result['rmse']:.4f}")
    print(f"  MAE          : {result['mae']:.4f}")
    print(f"  drift_score  : {result['drift_score']:.4f}  (KS statistic)")
    print(f"  drift_alert  : {result['drift_alert']}")
    print("\nRefresh Grafana to see the new datapoint.")


if __name__ == "__main__":
    main()
