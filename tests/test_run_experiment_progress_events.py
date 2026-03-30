from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from Thesis_ML.data.index_dataset import build_dataset_index
from Thesis_ML.experiments.progress import ProgressEvent
from Thesis_ML.experiments.run_experiment import run_experiment


def _write_nifti(path: Path, data: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = nib.Nifti1Image(data.astype(np.float32), affine=np.eye(4, dtype=np.float64))
    nib.save(image, str(path))


def _create_glm_session(glm_dir: Path, labels: list[str]) -> None:
    glm_dir.mkdir(parents=True, exist_ok=True)
    mask = np.zeros((3, 3, 3), dtype=np.float32)
    mask[1:, 1:, 1:] = 1.0
    _write_nifti(glm_dir / "mask.nii", mask)
    pd.Series(labels).to_csv(glm_dir / "regressor_labels.csv", index=False, header=False)
    for idx, _ in enumerate(labels, start=1):
        beta = np.full((3, 3, 3), fill_value=float(idx), dtype=np.float32)
        _write_nifti(glm_dir / f"beta_{idx:04d}.nii", beta)


def test_run_experiment_writes_progress_events_and_preserves_existing_callback(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "Data"
    labels = [
        "run-1_passive_anger_audio",
        "run-1_passive_happiness_audio",
        "run-1_passive_anger_video",
        "run-1_passive_happiness_video",
    ]
    _create_glm_session(data_root / "sub-001" / "ses-01" / "BAS2", labels)
    _create_glm_session(data_root / "sub-001" / "ses-02" / "BAS2", labels)

    index_csv = tmp_path / "dataset_index.csv"
    build_dataset_index(data_root=data_root, out_csv=index_csv)

    captured: list[ProgressEvent] = []

    def _callback(event: ProgressEvent) -> None:
        captured.append(event)

    result = run_experiment(
        index_csv=index_csv,
        data_root=data_root,
        cache_dir=tmp_path / "cache",
        target="coarse_affect",
        model="ridge",
        cv="within_subject_loso_session",
        subject="sub-001",
        seed=42,
        run_id="progress_events_test",
        reports_root=tmp_path / "reports",
        progress_callback=_callback,
    )

    assert captured
    assert any(
        event.stage == "run" and "starting run execution" in event.message for event in captured
    )
    assert any(
        event.stage == "run" and "finished run execution" in event.message for event in captured
    )

    report_dir = Path(result["report_dir"])
    progress_events_path = report_dir / "progress_events.jsonl"
    assert progress_events_path.exists()
    rows = [
        json.loads(line)
        for line in progress_events_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows
    assert all(row.get("timestamp_utc") for row in rows)
    assert any(str(row.get("stage")) == "run" for row in rows)
