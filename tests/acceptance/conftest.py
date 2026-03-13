from __future__ import annotations

from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd
import pytest
from openpyxl import load_workbook

from Thesis_ML.config.schema_versions import (
    COMPILED_MANIFEST_SCHEMA_VERSION,
    WORKBOOK_SCHEMA_VERSION,
)
from Thesis_ML.data.index_dataset import build_dataset_index
from Thesis_ML.workbook.builder import build_workbook


def _write_nifti(path: Path, data: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = nib.Nifti1Image(data.astype(np.float32), affine=np.eye(4, dtype=np.float64))
    nib.save(image, str(path))


def _create_tiny_glm_session(glm_dir: Path, labels: list[str]) -> None:
    glm_dir.mkdir(parents=True, exist_ok=True)

    shape = (3, 3, 3)
    mask = np.zeros(shape, dtype=np.float32)
    mask[1:, 1:, 1:] = 1.0
    _write_nifti(glm_dir / "mask.nii", mask)
    pd.Series(labels).to_csv(glm_dir / "regressor_labels.csv", index=False, header=False)

    for idx, label in enumerate(labels, start=1):
        beta = np.full(shape, fill_value=float(idx), dtype=np.float32)
        if "_anger_" in label:
            beta[1:, 1:, 1:] += 2.0
        if "_happiness_" in label:
            beta[1:, 1:, 1:] -= 2.0
        _write_nifti(glm_dir / f"beta_{idx:04d}.nii", beta)


@pytest.fixture
def acceptance_sample_data(tmp_path: Path) -> dict[str, Path]:
    labels = [
        "run-1_passive_anger_audio",
        "run-1_passive_happiness_audio",
        "run-1_passive_anger_video",
        "run-1_passive_happiness_video",
    ]
    data_root = tmp_path / "Data"
    _create_tiny_glm_session(data_root / "sub-001" / "ses-01" / "BAS2", labels)
    _create_tiny_glm_session(data_root / "sub-001" / "ses-02" / "BAS2", labels)

    index_csv = tmp_path / "dataset_index.csv"
    build_dataset_index(data_root=data_root, out_csv=index_csv)

    return {
        "data_root": data_root,
        "index_csv": index_csv,
        "cache_dir": tmp_path / "cache",
        "output_root": tmp_path / "outputs" / "artifacts" / "decision_support",
        "workbook_output_dir": tmp_path / "outputs" / "workbooks",
    }


@pytest.fixture
def acceptance_sample_workbook(tmp_path: Path) -> Path:
    workbook_path = tmp_path / "thesis_experiment_program.xlsx"
    wb = build_workbook()
    wb.save(workbook_path)

    workbook = load_workbook(workbook_path)
    ws = workbook["Experiment_Definitions"]
    headers = [ws.cell(1, col).value for col in range(1, ws.max_column + 1)]
    col = {str(name): idx + 1 for idx, name in enumerate(headers) if name is not None}

    ws.cell(2, col["experiment_id"], "E16")
    ws.cell(2, col["enabled"], "Yes")
    ws.cell(2, col["start_section"], "dataset_selection")
    ws.cell(2, col["end_section"], "evaluation")
    ws.cell(2, col["base_artifact_id"], "")
    ws.cell(2, col["target"], "coarse_affect")
    ws.cell(2, col["cv"], "within_subject_loso_session")
    ws.cell(2, col["model"], "ridge")
    ws.cell(2, col["subject"], "sub-001")
    ws.cell(2, col["train_subject"], "")
    ws.cell(2, col["test_subject"], "")
    ws.cell(2, col["filter_task"], "")
    ws.cell(2, col["filter_modality"], "")
    ws.cell(2, col["reuse_policy"], "auto")
    ws.cell(2, col["search_space_id"], "")
    workbook.save(workbook_path)
    return workbook_path


@pytest.fixture
def acceptance_expected_manifest_shape() -> dict[str, Any]:
    return {
        "schema_version": WORKBOOK_SCHEMA_VERSION,
        "compiled_manifest_schema_version": COMPILED_MANIFEST_SCHEMA_VERSION,
        "experiment_id": "E16",
        "trial_count": 1,
        "required_trial_params": {
            "target": "coarse_affect",
            "cv": "within_subject_loso_session",
            "model": "ridge",
        },
    }


@pytest.fixture
def acceptance_expected_output_shape() -> dict[str, str]:
    return {
        "trial_status": "completed",
        "summary_type": "best_full_pipeline",
        "run_log_result": "completed",
    }
