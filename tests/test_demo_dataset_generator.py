from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd


def _load_generator_module():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "generate_demo_dataset.py"
    spec = importlib.util.spec_from_file_location("generate_demo_dataset", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_demo_dataset_generation_is_deterministic(tmp_path: Path) -> None:
    module = _load_generator_module()
    output_dir = tmp_path / "demo_data" / "synthetic_v1"

    module.generate_demo_dataset(output_dir=output_dir, force=True)
    manifest_path = output_dir / "demo_dataset_manifest.json"
    manifest_a = json.loads(manifest_path.read_text(encoding="utf-8"))

    module.generate_demo_dataset(output_dir=output_dir, force=True)
    manifest_b = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest_a["dataset_id"] == "thesis_ml_synthetic_v1"
    assert manifest_a["index_csv_sha256"] == manifest_b["index_csv_sha256"]
    assert manifest_a["files"] == manifest_b["files"]

    index_df = pd.read_csv(output_dir / "dataset_index.csv")
    for column in (
        "sample_id",
        "subject",
        "session",
        "task",
        "modality",
        "beta_path",
        "mask_path",
        "regressor_label",
        "emotion",
        "coarse_affect",
        "subject_session",
        "subject_session_bas",
        "bas",
    ):
        assert column in index_df.columns


def test_checked_in_demo_dataset_manifest_exists_and_has_expected_shape() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    manifest_path = repo_root / "demo_data" / "synthetic_v1" / "demo_dataset_manifest.json"
    assert manifest_path.exists()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["manifest_schema_version"] == "demo-dataset-manifest-v1"
    assert payload["dataset_id"] == "thesis_ml_synthetic_v1"
    assert int(payload["n_subjects"]) >= 2
    assert int(payload["n_rows"]) > 0
