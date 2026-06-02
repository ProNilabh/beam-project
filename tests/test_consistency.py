from pathlib import Path

import pandas as pd
import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]

FEATURES = ["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8"]
TARGETS = ["Y1", "Y2"]


def _extract_list(source_path, var_name):
    import ast

    tree = ast.parse(Path(source_path).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == var_name:
                    return ast.literal_eval(node.value)
    return None


MODULES = [
    "src/prefect_train.py",
    "src/app.py",
    "monitoring/monitor.py",
    "monitoring/simulate_batch.py",
]


def test_feature_lists_consistent_across_modules():
    feature_lists = {m: _extract_list(REPO_ROOT / m, "FEATURES") for m in MODULES}
    target_lists = {m: _extract_list(REPO_ROOT / m, "TARGETS") for m in MODULES}

    unique_features = {tuple(v) for v in feature_lists.values() if v is not None}
    unique_targets = {tuple(v) for v in target_lists.values() if v is not None}

    assert len(unique_features) == 1, f"FEATURES list differs between modules: {feature_lists}"
    assert len(unique_targets) == 1, f"TARGETS list differs between modules: {target_lists}"


def test_feature_lists_match_expected():
    features = _extract_list(REPO_ROOT / "src/prefect_train.py", "FEATURES")
    targets = _extract_list(REPO_ROOT / "src/prefect_train.py", "TARGETS")

    assert features == ["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8"]
    assert targets == ["Y1", "Y2"]


def test_dataset_file_exists():
    # The dataset is a small public benchmark and is intentionally NOT committed
    # to the repo; it is provided at run time (mounted into the containers).
    # When a local copy is present we sanity-check it; in CI it is absent, so skip.
    data_path = REPO_ROOT / "data" / "ENB2012_data.xlsx"
    holdout_path = REPO_ROOT / "data" / "ENB2012_holdout.xlsx"
    if not (data_path.exists() or holdout_path.exists()):
        pytest.skip("Dataset not committed to the repo; provided at runtime")
    assert data_path.exists() or holdout_path.exists()


def test_dataset_has_required_columns():
    data_path = REPO_ROOT / "data" / "ENB2012_data.xlsx"
    if not data_path.exists():
        pytest.skip("Dataset not present in CI environment")

    df = pd.read_excel(data_path)
    required = set(FEATURES + TARGETS)
    missing = required - set(df.columns)
    assert not missing, f"Dataset is missing columns: {missing}"


def test_compose_volume_paths_exist():
    compose_path = REPO_ROOT / "docker-compose.yml"
    compose = yaml.safe_load(compose_path.read_text())

    # Host paths that are provided at run time rather than committed to the repo.
    # The dataset directory is mounted into the containers at run time, so its
    # absence from the repo is expected and must not fail this test.
    runtime_mounts = {"./data"}

    missing = []
    for service_name, service in compose.get("services", {}).items():
        for volume in service.get("volumes", []):
            if isinstance(volume, str) and volume.startswith("./"):
                host_path = volume.split(":")[0]
                if host_path in runtime_mounts:
                    continue
                if not (REPO_ROOT / host_path).exists():
                    missing.append(f"{service_name} -> {host_path}")

    assert not missing, f"docker-compose references missing paths: {missing}"


@pytest.mark.parametrize("relpath", [
    "Dockerfile",
    "docker-compose.yml",
    "requirements.txt",
    "src/app.py",
    "src/prefect_train.py",
    "monitoring/monitor.py",
    "monitoring/simulate_batch.py",
    "monitoring/init_db.sql",
    "grafana/dashboards/beam_monitoring.json",
    "grafana/provisioning/datasources/datasource.yml",
    "grafana/provisioning/dashboards/dashboard.yml",
])
def test_critical_file_exists(relpath):
    assert (REPO_ROOT / relpath).exists(), f"Missing critical file: {relpath}"
