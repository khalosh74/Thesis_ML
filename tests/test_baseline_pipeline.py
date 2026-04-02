from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from Thesis_ML.pipelines.baseline import run_baseline


def test_run_baseline_creates_outputs(tmp_path: Path) -> None:
    reports_dir = tmp_path / "reports"
    models_dir = tmp_path / "models"

    result = run_baseline(seed=123, reports_dir=reports_dir, models_dir=models_dir)

    metrics_path = Path(result["metrics_path"])
    model_path = Path(result["model_path"])

    assert metrics_path.exists()
    assert model_path.exists()

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    expected_keys = {
        "mae",
        "rmse",
        "r2",
        "seed",
        "n_samples",
        "n_features",
        "noise_std",
        "test_fraction",
    }
    assert expected_keys <= set(metrics)

    with np.load(model_path) as artifact:
        assert artifact["weights"].ndim == 1
        assert artifact["intercept"].shape == ()


def test_run_baseline_is_deterministic(tmp_path: Path) -> None:
    result_a = run_baseline(
        seed=777,
        reports_dir=tmp_path / "reports_a",
        models_dir=tmp_path / "models_a",
    )
    result_b = run_baseline(
        seed=777,
        reports_dir=tmp_path / "reports_b",
        models_dir=tmp_path / "models_b",
    )

    metrics_a = result_a["metrics"]
    metrics_b = result_b["metrics"]

    for metric_name in ("mae", "rmse", "r2"):
        assert metrics_a[metric_name] == pytest.approx(metrics_b[metric_name], rel=0, abs=1e-12)

    with np.load(result_a["model_path"]) as model_a, np.load(result_b["model_path"]) as model_b:
        np.testing.assert_allclose(model_a["weights"], model_b["weights"], rtol=0, atol=0)
        np.testing.assert_allclose(model_a["intercept"], model_b["intercept"], rtol=0, atol=0)


def test_cli_smoke(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    reports_dir = tmp_path / "cli_reports"
    models_dir = tmp_path / "cli_models"

    command = [
        sys.executable,
        "-m",
        "Thesis_ML.cli.baseline",
        "--seed",
        "21",
        "--reports-dir",
        str(reports_dir),
        "--models-dir",
        str(models_dir),
    ]
    process = subprocess.run(command, cwd=project_root, check=False, capture_output=True, text=True)

    assert process.returncode == 0, process.stderr
    assert (reports_dir / "metrics.json").exists()
    assert (models_dir / "baseline_model.npz").exists()
