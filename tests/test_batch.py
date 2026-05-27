import numpy as np
import pandas as pd
import pytest

FEATURES = ["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8"]
TARGETS = ["Y1", "Y2"]


def make_batch(reference_df, size, drift, seed=0):
    rng = np.random.default_rng(seed)
    sample = reference_df.sample(n=size, replace=True, random_state=seed).copy().reset_index(drop=True)
    if drift > 0:
        for col in FEATURES:
            std = reference_df[col].std()
            sample[col] = sample[col] + rng.normal(0, drift * std, size)
    return sample


@pytest.fixture
def fake_reference():
    rng = np.random.default_rng(0)
    return pd.DataFrame({c: rng.normal(0, 1, 300) for c in FEATURES + TARGETS})


def test_batch_has_requested_size(fake_reference):
    batch = make_batch(fake_reference, size=50, drift=0.0)
    assert len(batch) == 50


def test_batch_has_all_columns(fake_reference):
    batch = make_batch(fake_reference, size=50, drift=0.5)
    for col in FEATURES + TARGETS:
        assert col in batch.columns


def test_zero_drift_preserves_distribution(fake_reference):
    batch = make_batch(fake_reference, size=1000, drift=0.0)
    for col in FEATURES:
        ref_std = fake_reference[col].std()
        batch_std = batch[col].std()
        assert abs(batch_std - ref_std) / ref_std < 0.30


def test_high_drift_widens_distribution(fake_reference):
    batch = make_batch(fake_reference, size=1000, drift=1.0)
    wider_count = sum(
        batch[col].std() > fake_reference[col].std() for col in FEATURES
    )
    assert wider_count >= 7


def test_payload_shape_matches_api_contract(fake_reference):
    batch = make_batch(fake_reference, size=5, drift=0.2)
    payload = {
        "drift_level": 0.2,
        "rows": batch[FEATURES + TARGETS].to_dict(orient="records"),
    }
    assert "drift_level" in payload
    assert len(payload["rows"]) == 5
    for row in payload["rows"]:
        assert set(row.keys()) == set(FEATURES + TARGETS)
