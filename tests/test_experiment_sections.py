from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from Thesis_ML.artifacts.registry import (
    ARTIFACT_TYPE_EXPERIMENT_REPORT,
    ARTIFACT_TYPE_INTERPRETABILITY_BUNDLE,
    ARTIFACT_TYPE_METRICS_BUNDLE,
    get_artifact,
    list_artifacts_for_run,
)
from Thesis_ML.data.index_dataset import build_dataset_index
from Thesis_ML.experiments.run_experiment import run_experiment
from Thesis_ML.experiments.sections import DatasetSelectionInput, dataset_selection


def _write_nifti(path: Path, data: np.ndarray, affine: np.ndarray | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    matrix = np.eye(4, dtype=np.float64) if affine is None else np.asarray(affine, dtype=np.float64)
    image = nib.Nifti1Image(data.astype(np.float32), affine=matrix)
    nib.save(image, str(path))


def _create_glm_session(
    glm_dir: Path,
    labels: list[str],
    class_signal: bool = False,
    shape: tuple[int, int, int] = (3, 3, 3),
) -> None:
    glm_dir.mkdir(parents=True, exist_ok=True)

    mask = np.zeros(shape, dtype=np.float32)
    mask[1:, 1:, 1:] = 1.0
    _write_nifti(glm_dir / "mask.nii", mask)
    pd.Series(labels).to_csv(glm_dir / "regressor_labels.csv", index=False, header=False)

    for idx, label in enumerate(labels, start=1):
        beta = np.full(shape, fill_value=float(idx), dtype=np.float32)
        if class_signal:
            if "_anger_" in label:
                beta[1:, 1:, 1:] += 5.0
            if "_happiness_" in label:
                beta[1:, 1:, 1:] -= 5.0
        _write_nifti(glm_dir / f"beta_{idx:04d}.nii", beta)


def test_dataset_selection_section_isolation(tmp_path: Path) -> None:
    index_csv = tmp_path / "dataset_index.csv"
    index_df = pd.DataFrame(
        [
            {
                "sample_id": "s1",
                "subject": "sub-001",
                "session": "ses-01",
                "task": "passive",
                "modality": "audio",
                "emotion": "anger",
            },
            {
                "sample_id": "s2",
                "subject": "sub-001",
                "session": "ses-02",
                "task": "passive",
                "modality": "video",
                "emotion": "happiness",
            },
            {
                "sample_id": "s3",
                "subject": "sub-002",
                "session": "ses-01",
                "task": "passive",
                "modality": "audio",
                "emotion": "anger",
            },
        ]
    )
    index_df.to_csv(index_csv, index=False)

    section_output = dataset_selection(
        DatasetSelectionInput(
            index_csv=index_csv,
            target_column="coarse_affect",
            cv_mode="within_subject_loso_session",
            subject="sub-001",
        )
    )
    selected = section_output.selected_index_df

    assert set(selected["subject"].astype(str).tolist()) == {"sub-001"}
    assert set(selected["sample_id"].astype(str).tolist()) == {"s1", "s2"}
    assert set(selected["coarse_affect"].astype(str).tolist()) == {"negative", "positive"}


def test_run_experiment_registers_section_boundary_artifacts(tmp_path: Path) -> None:
    data_root = tmp_path / "Data"
    labels = [
        "run-1_passive_anger_audio",
        "run-1_passive_happiness_audio",
        "run-1_passive_anger_video",
        "run-1_passive_happiness_video",
    ]
    for subject in ("sub-001", "sub-002"):
        for session in ("ses-01", "ses-02"):
            _create_glm_session(
                glm_dir=data_root / subject / session / "BAS2",
                labels=labels,
                class_signal=True,
            )

    index_csv = tmp_path / "dataset_index.csv"
    build_dataset_index(data_root=data_root, out_csv=index_csv)

    run_id = "section_boundary_smoke"
    reports_root = tmp_path / "reports" / "experiments"
    result = run_experiment(
        index_csv=index_csv,
        data_root=data_root,
        cache_dir=tmp_path / "cache",
        target="emotion",
        model="ridge",
        cv="loso_session",
        seed=7,
        run_id=run_id,
        reports_root=reports_root,
    )

    artifact_ids = result["artifact_ids"]
    assert "feature_cache" in artifact_ids
    assert "feature_matrix_bundle" in artifact_ids
    assert ARTIFACT_TYPE_METRICS_BUNDLE in artifact_ids
    assert ARTIFACT_TYPE_INTERPRETABILITY_BUNDLE in artifact_ids
    assert ARTIFACT_TYPE_EXPERIMENT_REPORT in artifact_ids

    registry_path = reports_root / "artifact_registry.sqlite3"
    records = list_artifacts_for_run(registry_path=registry_path, run_id=run_id)
    record_types = {record.artifact_type for record in records}
    assert {
        "feature_cache",
        "feature_matrix_bundle",
        ARTIFACT_TYPE_METRICS_BUNDLE,
        ARTIFACT_TYPE_INTERPRETABILITY_BUNDLE,
        ARTIFACT_TYPE_EXPERIMENT_REPORT,
    } <= record_types

    feature_matrix_id = artifact_ids["feature_matrix_bundle"]
    metrics_id = artifact_ids[ARTIFACT_TYPE_METRICS_BUNDLE]
    interpretability_id = artifact_ids[ARTIFACT_TYPE_INTERPRETABILITY_BUNDLE]
    report_id = artifact_ids[ARTIFACT_TYPE_EXPERIMENT_REPORT]

    metrics_record = get_artifact(registry_path=registry_path, artifact_id=metrics_id)
    interpretability_record = get_artifact(
        registry_path=registry_path, artifact_id=interpretability_id
    )
    report_record = get_artifact(registry_path=registry_path, artifact_id=report_id)

    assert metrics_record is not None
    assert interpretability_record is not None
    assert report_record is not None
    assert metrics_record.upstream_artifact_ids == [feature_matrix_id]
    assert interpretability_record.upstream_artifact_ids == [feature_matrix_id]
    assert set(report_record.upstream_artifact_ids) == {metrics_id, interpretability_id}
