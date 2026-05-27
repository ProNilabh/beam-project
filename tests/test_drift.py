import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

FEATURES = ["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8"]


def compute_drift_score(reference_df, batch_df):
    ks_stats = [
        ks_2samp(reference_df[col].values, batch_df[col].values).statistic
        for col in FEATURES
    ]
    return float(np.mean(ks_stats))


def make_reference(n=300, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({f: rng.normal(0, 1, n) for f in FEATURES})


def test_no_drift_gives_low_score():
    rng = np.random.default_rng(1)
    ref = make_reference(seed=1)
    batch = pd.DataFrame({f: rng.normal(0, 1, 100) for f in FEATURES})
    score = compute_drift_score(ref, batch)
    assert score < 0.20, f"Clean data should not trigger alert, got {score:.3f}"


def test_heavy_drift_gives_high_score():
    rng = np.random.default_rng(2)
    ref = make_reference(seed=2)
    batch = pd.DataFrame({f: rng.normal(5, 1, 100) for f in FEATURES})
    score = compute_drift_score(ref, batch)
    assert score > 0.20, f"Heavy drift should trigger alert, got {score:.3f}"


def test_drift_score_is_bounded():
    rng = np.random.default_rng(3)
    ref = make_reference(seed=3)
    for _ in range(10):
        batch = pd.DataFrame({
            f: rng.normal(rng.uniform(-3, 3), rng.uniform(0.5, 3), 80)
            for f in FEATURES
        })
        score = compute_drift_score(ref, batch)
        assert 0.0 <= score <= 1.0


def test_more_drift_means_higher_score():
    ref = make_reference(seed=4)
    rng = np.random.default_rng(4)

    scores = []
    for shift in [0.0, 1.0, 3.0, 6.0]:
        batch = pd.DataFrame({f: rng.normal(shift, 1, 200) for f in FEATURES})
        scores.append(compute_drift_score(ref, batch))

    assert scores[0] < scores[1] < scores[2] <= scores[3], f"Non-monotonic: {scores}"


def test_constant_feature_does_not_crash():
    ref = make_reference(seed=5)
    batch = make_reference(seed=6)
    batch["X6"] = 3.0
    score = compute_drift_score(ref, batch)
    assert 0.0 <= score <= 1.0


def test_tiny_batch_does_not_crash():
    ref = make_reference(seed=7)
    batch = ref.iloc[:1].copy()
    score = compute_drift_score(ref, batch)
    assert 0.0 <= score <= 1.0
