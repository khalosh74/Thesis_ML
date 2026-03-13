from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from Thesis_ML.artifacts.registry import (
    ARTIFACT_TYPE_FEATURE_CACHE,
    ARTIFACT_TYPE_FEATURE_MATRIX_BUNDLE,
    ARTIFACT_TYPE_INTERPRETABILITY_BUNDLE,
    ARTIFACT_TYPE_METRICS_BUNDLE,
    list_artifacts_for_run,
)
from Thesis_ML.data.index_dataset import build_dataset_index
from Thesis_ML.experiments.run_experiment import run_experiment
from Thesis_ML.experiments.segment_execution import plan_section_path


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


@pytest.fixture
def prepared_dataset(tmp_path: Path) -> dict[str, Path]:
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
    return {
        "data_root": data_root,
        "index_csv": index_csv,
        "cache_dir": tmp_path / "cache",
        "reports_root": tmp_path / "reports" / "experiments",
    }


def _base_run_kwargs(prepared_dataset: dict[str, Path]) -> dict[str, object]:
    return {
        "index_csv": prepared_dataset["index_csv"],
        "data_root": prepared_dataset["data_root"],
        "cache_dir": prepared_dataset["cache_dir"],
        "target": "emotion",
        "model": "ridge",
        "cv": "loso_session",
        "seed": 13,
        "reports_root": prepared_dataset["reports_root"],
    }


def test_plan_section_path_feature_matrix_to_evaluation() -> None:
    path = plan_section_path(start_section="feature_matrix_load", end_section="evaluation")
    assert [section.value for section in path] == [
        "feature_matrix_load",
        "spatial_validation",
        "model_fit",
        "interpretability",
        "evaluation",
    ]


def test_full_pipeline_execution_remains_supported(prepared_dataset: dict[str, Path]) -> None:
    result = run_experiment(
        **_base_run_kwargs(prepared_dataset),
        run_id="full_pipeline_default",
    )

    assert result["planned_sections"] == [
        "dataset_selection",
        "feature_cache_build",
        "feature_matrix_load",
        "spatial_validation",
        "model_fit",
        "interpretability",
        "evaluation",
    ]
    assert result["executed_sections"] == result["planned_sections"]
    assert result["metrics"]["n_folds"] >= 2
    assert Path(result["metrics_path"]).exists()


def test_segment_execution_feature_matrix_to_evaluation(
    prepared_dataset: dict[str, Path],
) -> None:
    base_result = run_experiment(
        **_base_run_kwargs(prepared_dataset),
        run_id="segment_base_run",
    )
    base_feature_cache_id = base_result["artifact_ids"]["feature_cache"]

    segment_result = run_experiment(
        **_base_run_kwargs(prepared_dataset),
        run_id="segment_only_run",
        start_section="feature_matrix_load",
        end_section="evaluation",
        base_artifact_id=base_feature_cache_id,
        reuse_policy="require_explicit_base",
    )

    assert segment_result["executed_sections"] == [
        "feature_matrix_load",
        "spatial_validation",
        "model_fit",
        "interpretability",
        "evaluation",
    ]
    assert Path(segment_result["metrics_path"]).exists()
    assert segment_result["metrics"]["n_folds"] >= 2

    registry_path = Path(segment_result["artifact_registry_path"])
    run_records = list_artifacts_for_run(registry_path=registry_path, run_id="segment_only_run")
    record_types = {record.artifact_type for record in run_records}
    assert ARTIFACT_TYPE_FEATURE_MATRIX_BUNDLE in record_types
    assert ARTIFACT_TYPE_INTERPRETABILITY_BUNDLE in record_types
    assert ARTIFACT_TYPE_METRICS_BUNDLE in record_types
    assert ARTIFACT_TYPE_FEATURE_CACHE not in record_types


def test_invalid_start_end_combination_raises(prepared_dataset: dict[str, Path]) -> None:
    with pytest.raises(ValueError, match="start_section must be before or equal to end_section"):
        run_experiment(
            **_base_run_kwargs(prepared_dataset),
            run_id="invalid_start_end",
            start_section="evaluation",
            end_section="feature_matrix_load",
        )


def test_incompatible_base_artifact_raises(prepared_dataset: dict[str, Path]) -> None:
    base_result = run_experiment(
        **_base_run_kwargs(prepared_dataset),
        run_id="incompatible_base_source",
    )
    metrics_artifact_id = base_result["artifact_ids"][ARTIFACT_TYPE_METRICS_BUNDLE]

    with pytest.raises(ValueError, match="Incompatible base artifact"):
        run_experiment(
            **_base_run_kwargs(prepared_dataset),
            run_id="incompatible_base_target",
            start_section="feature_matrix_load",
            end_section="evaluation",
            base_artifact_id=metrics_artifact_id,
            reuse_policy="require_explicit_base",
        )
