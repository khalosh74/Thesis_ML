from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from Thesis_ML.artifacts.registry import (
    ARTIFACT_TYPE_EXPERIMENT_REPORT,
    ARTIFACT_TYPE_INTERPRETABILITY_BUNDLE,
    ARTIFACT_TYPE_METRICS_BUNDLE,
    get_artifact,
    list_artifacts_for_run,
)
from Thesis_ML.data.index_dataset import build_dataset_index
from Thesis_ML.experiments import sections_impl as sections_impl_module
from Thesis_ML.experiments.progress import ProgressEvent
from Thesis_ML.experiments.run_experiment import (
    _build_pipeline,
    _compute_interpretability_stability,
    _evaluate_permutations,
    _extract_linear_coefficients,
    _scores_for_predictions,
    run_experiment,
)
from Thesis_ML.experiments.stage_registry import PERMUTATION_REFERENCE_EXECUTOR_ID
from Thesis_ML.experiments.stage_registry import (
    SPECIALIZED_RIDGE_TUNING_EXECUTOR_ID,
    TUNING_GENERIC_EXECUTOR_ID,
)
from Thesis_ML.experiments.stage_execution import StageAssignment, StageBackendFamily, StageKey
from Thesis_ML.experiments.tuning_search_spaces import (
    LINEAR_GROUPED_NESTED_SEARCH_SPACE_ID,
    LINEAR_GROUPED_NESTED_SEARCH_SPACE_VERSION,
)
from Thesis_ML.verification.backend_parity import compare_tuned_null_parity
from Thesis_ML.features.feature_qc import FEATURE_QC_SAMPLE_FIELDS
from Thesis_ML.experiments.sections import (
    DatasetSelectionInput,
    EvaluationInput,
    InterpretabilityInput,
    ModelFitInput,
    dataset_selection,
    evaluation,
    interpretability,
    model_fit,
)


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


def _build_grouped_nested_evaluation_input(
    *,
    tmp_path: Path,
    evidence_run_role: str = "primary",
    n_permutations: int = 2,
    model: str = "linearsvc",
) -> EvaluationInput:
    report_dir = tmp_path / f"eval_{evidence_run_role}"
    report_dir.mkdir(parents=True, exist_ok=True)
    metadata_df = pd.DataFrame(
        [
            {"sample_id": "s1", "subject": "sub-001", "session": "ses-01", "coarse_affect": "negative"},
            {"sample_id": "s2", "subject": "sub-001", "session": "ses-01", "coarse_affect": "positive"},
            {"sample_id": "s3", "subject": "sub-001", "session": "ses-02", "coarse_affect": "negative"},
            {"sample_id": "s4", "subject": "sub-001", "session": "ses-02", "coarse_affect": "positive"},
            {"sample_id": "s5", "subject": "sub-001", "session": "ses-03", "coarse_affect": "negative"},
            {"sample_id": "s6", "subject": "sub-001", "session": "ses-03", "coarse_affect": "positive"},
        ]
    )
    for field_name in FEATURE_QC_SAMPLE_FIELDS:
        metadata_df[field_name] = 0.0
    x_matrix = np.asarray(
        [
            [2.0, 0.1, 0.0, 0.2],
            [-2.0, -0.1, 0.0, -0.2],
            [2.1, 0.2, 0.1, 0.1],
            [-2.1, -0.2, -0.1, -0.1],
            [2.2, 0.3, 0.2, 0.0],
            [-2.2, -0.3, -0.2, 0.0],
        ],
        dtype=np.float64,
    )
    y = metadata_df["coarse_affect"].astype(str).to_numpy()
    splits = [
        (np.asarray([2, 3, 4, 5], dtype=np.int64), np.asarray([0, 1], dtype=np.int64)),
        (np.asarray([0, 1, 4, 5], dtype=np.int64), np.asarray([2, 3], dtype=np.int64)),
        (np.asarray([0, 1, 2, 3], dtype=np.int64), np.asarray([4, 5], dtype=np.int64)),
    ]
    fold_rows = [
        {
            "fold": fold_index,
            "n_train": 4,
            "n_test": 2,
            "accuracy": 1.0,
            "balanced_accuracy": 1.0,
            "macro_f1": 1.0,
        }
        for fold_index in range(3)
    ]
    split_rows = [
        {"fold": fold_index, "partition": "train", "sample_id": f"train_{fold_index}"}
        for fold_index in range(3)
    ]
    return EvaluationInput(
        x_matrix=x_matrix,
        metadata_df=metadata_df,
        y=y,
        splits=splits,
        fold_rows=fold_rows,
        split_rows=split_rows,
        prediction_rows=[],
        y_true_all=y.tolist(),
        y_pred_all=y.tolist(),
        subject="sub-001",
        n_permutations=n_permutations,
        primary_metric_name="balanced_accuracy",
        permutation_metric_name="balanced_accuracy",
        methodology_policy_name="grouped_nested_tuning",
        evidence_run_role=evidence_run_role,
        tuning_enabled=True,
        tuning_search_space_id=LINEAR_GROUPED_NESTED_SEARCH_SPACE_ID,
        tuning_search_space_version=LINEAR_GROUPED_NESTED_SEARCH_SPACE_VERSION,
        tuning_inner_cv_scheme="grouped_leave_one_group_out",
        tuning_inner_group_field="session",
        subgroup_reporting_enabled=False,
        calibration_enabled=False,
        emit_feature_qc_artifacts=False,
        spatial_compatibility={
            "status": "passed",
            "passed": True,
            "n_groups_checked": 3,
            "reference_group_id": "sub-001_ses-01",
            "affine_atol": 1e-5,
        },
        spatial_report_path=report_dir / "spatial_compatibility_report.json",
        interpretability_summary={
            "enabled": False,
            "performed": False,
            "status": "not_applicable",
            "fold_artifacts_path": str((report_dir / "interpretability_fold_explanations.csv").resolve()),
            "stability": {},
        },
        interpretability_summary_path=report_dir / "interpretability_summary.json",
        fold_metrics_path=report_dir / "fold_metrics.csv",
        fold_splits_path=report_dir / "fold_splits.csv",
        predictions_path=report_dir / "predictions.csv",
        config_filename="config.json",
        build_pipeline_fn=_build_pipeline,
        evaluate_permutations_fn=_evaluate_permutations,
        run_id=f"eval_{evidence_run_role}",
        artifact_registry_path=tmp_path / "artifact_registry.sqlite3",
        upstream_feature_matrix_artifact_id="feature_matrix_bundle_test",
        metrics_path=report_dir / "metrics.json",
        model=str(model),
        target_column="coarse_affect",
        cv_mode="within_subject_loso_session",
        seed=23,
    )


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


def test_dataset_selection_binary_valence_like_drops_neutral(tmp_path: Path) -> None:
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
                "session": "ses-01",
                "task": "passive",
                "modality": "video",
                "emotion": "happiness",
            },
            {
                "sample_id": "s3",
                "subject": "sub-001",
                "session": "ses-02",
                "task": "passive",
                "modality": "audio",
                "emotion": "neutral",
            },
        ]
    )
    index_df.to_csv(index_csv, index=False)

    selection_output = dataset_selection(
        DatasetSelectionInput(
            index_csv=index_csv,
            target_column="binary_valence_like",
            cv_mode="within_subject_loso_session",
            subject="sub-001",
        )
    )
    selected = selection_output.selected_index_df
    exclusion_manifest = selection_output.selection_exclusion_manifest_df

    assert set(selected["sample_id"].astype(str).tolist()) == {"s1", "s2"}
    assert set(selected["binary_valence_like"].astype(str).tolist()) == {"negative", "positive"}

    assert int(len(exclusion_manifest)) == 1
    assert exclusion_manifest.iloc[0]["sample_id"] == "s3"
    assert exclusion_manifest.iloc[0]["exclusion_stage"] == "target_cleanup"
    assert exclusion_manifest.iloc[0]["exclusion_reason"] == "target_missing_after_derivation"


def test_dataset_selection_rejects_inconsistent_stored_coarse_affect(
    tmp_path: Path,
) -> None:
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
                "coarse_affect": "positive",
            },
            {
                "sample_id": "s2",
                "subject": "sub-001",
                "session": "ses-02",
                "task": "passive",
                "modality": "video",
                "emotion": "happiness",
                "coarse_affect": "positive",
            },
        ]
    )
    index_df.to_csv(index_csv, index=False)

    with pytest.raises(ValueError, match="inconsistent stored derived labels"):
        dataset_selection(
            DatasetSelectionInput(
                index_csv=index_csv,
                target_column="coarse_affect",
                cv_mode="within_subject_loso_session",
                subject="sub-001",
            )
        )


def test_dataset_selection_rejects_inconsistent_stored_binary_valence_like(
    tmp_path: Path,
) -> None:
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
                "coarse_affect": "negative",
                "binary_valence_like": "positive",
            },
            {
                "sample_id": "s2",
                "subject": "sub-001",
                "session": "ses-02",
                "task": "passive",
                "modality": "video",
                "emotion": "happiness",
                "coarse_affect": "positive",
                "binary_valence_like": "positive",
            },
        ]
    )
    index_df.to_csv(index_csv, index=False)

    with pytest.raises(ValueError, match="inconsistent stored derived labels"):
        dataset_selection(
            DatasetSelectionInput(
                index_csv=index_csv,
                target_column="binary_valence_like",
                cv_mode="within_subject_loso_session",
                subject="sub-001",
            )
        )


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
    report_dir = Path(result["report_dir"])
    config_payload = json.loads((report_dir / "config.json").read_text(encoding="utf-8"))
    metrics_payload = json.loads((report_dir / "metrics.json").read_text(encoding="utf-8"))
    feature_qc_summary_payload = json.loads(
        (report_dir / "feature_qc_summary.json").read_text(encoding="utf-8")
    )

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
    assert config_payload["feature_recipe_id"] == "baseline_standard_scaler_v1"
    assert metrics_payload["feature_recipe_id"] == "baseline_standard_scaler_v1"
    assert feature_qc_summary_payload["feature_recipe_id"] == "baseline_standard_scaler_v1"
    assert (report_dir / "feature_qc_selected_samples.csv").exists()


def test_extracted_sections_model_fit_interpretability_evaluation(tmp_path: Path) -> None:
    report_dir = tmp_path / "run"
    report_dir.mkdir(parents=True, exist_ok=True)
    registry_path = tmp_path / "artifact_registry.sqlite3"

    metadata_df = pd.DataFrame(
        [
            {
                "sample_id": "s1",
                "subject": "sub-001",
                "session": "ses-01",
                "bas": "BAS2",
                "task": "passive",
                "modality": "audio",
                "emotion": "anger",
                "coarse_affect": "negative",
            },
            {
                "sample_id": "s2",
                "subject": "sub-001",
                "session": "ses-01",
                "bas": "BAS2",
                "task": "passive",
                "modality": "video",
                "emotion": "happiness",
                "coarse_affect": "positive",
            },
            {
                "sample_id": "s3",
                "subject": "sub-001",
                "session": "ses-02",
                "bas": "BAS2",
                "task": "passive",
                "modality": "audio",
                "emotion": "anger",
                "coarse_affect": "negative",
            },
            {
                "sample_id": "s4",
                "subject": "sub-001",
                "session": "ses-02",
                "bas": "BAS2",
                "task": "passive",
                "modality": "video",
                "emotion": "happiness",
                "coarse_affect": "positive",
            },
            {
                "sample_id": "s5",
                "subject": "sub-001",
                "session": "ses-03",
                "bas": "BAS2",
                "task": "passive",
                "modality": "audio",
                "emotion": "anger",
                "coarse_affect": "negative",
            },
            {
                "sample_id": "s6",
                "subject": "sub-001",
                "session": "ses-03",
                "bas": "BAS2",
                "task": "passive",
                "modality": "video",
                "emotion": "happiness",
                "coarse_affect": "positive",
            },
        ]
    )
    x_matrix = np.asarray(
        [
            [4.0, 0.2, 0.1, 0.0],
            [-4.0, -0.2, -0.1, 0.0],
            [3.8, 0.1, 0.2, 0.1],
            [-3.9, -0.1, -0.2, -0.1],
            [4.1, 0.3, 0.2, 0.2],
            [-4.2, -0.3, -0.2, -0.2],
        ],
        dtype=np.float32,
    )

    fit_output = model_fit(
        ModelFitInput(
            x_matrix=x_matrix,
            metadata_df=metadata_df,
            target_column="coarse_affect",
            cv_mode="within_subject_loso_session",
            model="ridge",
            subject="sub-001",
            seed=11,
            run_id="sections_unit_smoke",
            config_filename="config.json",
            report_dir=report_dir,
            build_pipeline_fn=_build_pipeline,
            scores_for_predictions_fn=_scores_for_predictions,
            extract_linear_coefficients_fn=_extract_linear_coefficients,
        )
    )
    assert fit_output.interpretability_enabled is True
    assert len(fit_output.splits) == 3
    assert len(fit_output.fold_rows) == 3
    assert len(fit_output.prediction_rows) == 6

    interpretability_output = interpretability(
        InterpretabilityInput(
            interpretability_enabled=fit_output.interpretability_enabled,
            interpretability_fold_rows=fit_output.interpretability_fold_rows,
            interpretability_vectors=fit_output.interpretability_vectors,
            fold_artifacts_path=report_dir / "interpretability_fold_explanations.csv",
            summary_path=report_dir / "interpretability_summary.json",
            compute_interpretability_stability_fn=_compute_interpretability_stability,
            run_id="sections_unit_smoke",
            artifact_registry_path=registry_path,
            upstream_feature_matrix_artifact_id="feature_matrix_bundle_test",
            cv_mode="within_subject_loso_session",
            model="ridge",
            target_column="coarse_affect",
            subject="sub-001",
        )
    )
    summary = interpretability_output.interpretability_summary
    assert summary["performed"] is True
    assert Path(str(summary["fold_artifacts_path"])).exists()
    assert Path(report_dir / "interpretability_summary.json").exists()

    evaluation_output = evaluation(
        EvaluationInput(
            x_matrix=x_matrix,
            y=fit_output.y,
            splits=fit_output.splits,
            fold_rows=fit_output.fold_rows,
            split_rows=fit_output.split_rows,
            prediction_rows=fit_output.prediction_rows,
            y_true_all=fit_output.y_true_all,
            y_pred_all=fit_output.y_pred_all,
            subject="sub-001",
            n_permutations=0,
            spatial_compatibility={
                "status": "passed",
                "passed": True,
                "n_groups_checked": 3,
                "reference_group_id": "sub-001_ses-01_BAS2",
                "affine_atol": 1e-5,
            },
            spatial_report_path=report_dir / "spatial_compatibility_report.json",
            interpretability_summary=summary,
            interpretability_summary_path=report_dir / "interpretability_summary.json",
            fold_metrics_path=report_dir / "fold_metrics.csv",
            fold_splits_path=report_dir / "fold_splits.csv",
            predictions_path=report_dir / "predictions.csv",
            config_filename="config.json",
            build_pipeline_fn=_build_pipeline,
            evaluate_permutations_fn=_evaluate_permutations,
            run_id="sections_unit_smoke",
            artifact_registry_path=registry_path,
            upstream_feature_matrix_artifact_id="feature_matrix_bundle_test",
            metrics_path=report_dir / "metrics.json",
            model="ridge",
            target_column="coarse_affect",
            cv_mode="within_subject_loso_session",
            seed=11,
        )
    )
    assert Path(report_dir / "metrics.json").exists()
    metrics = json.loads((report_dir / "metrics.json").read_text(encoding="utf-8"))
    assert {"accuracy", "balanced_accuracy", "macro_f1", "interpretability"} <= set(metrics)
    assert "calibration" in metrics
    assert metrics["calibration"]["status"] in {"performed", "not_applicable", "failed"}
    assert Path(report_dir / "calibration_summary.json").exists()
    assert Path(report_dir / "calibration_table.csv").exists()
    assert evaluation_output.metrics["n_folds"] == 3

    metrics_record = get_artifact(
        registry_path=registry_path,
        artifact_id=evaluation_output.metrics_artifact_id,
    )
    assert metrics_record is not None
    assert metrics_record.upstream_artifact_ids == ["feature_matrix_bundle_test"]


def test_primary_metric_aggregation_controls_headline_metric_value(tmp_path: Path) -> None:
    def _build_input(*, aggregation: str) -> EvaluationInput:
        report_dir = tmp_path / aggregation
        report_dir.mkdir(parents=True, exist_ok=True)
        metadata_df = pd.DataFrame(
            [
                {
                    "sample_id": "s1",
                    "subject": "sub-001",
                    "session": "ses-01",
                    "task": "passive",
                    "modality": "audio",
                    "coarse_affect": "negative",
                },
                {
                    "sample_id": "s2",
                    "subject": "sub-001",
                    "session": "ses-01",
                    "task": "passive",
                    "modality": "audio",
                    "coarse_affect": "negative",
                },
                {
                    "sample_id": "s3",
                    "subject": "sub-001",
                    "session": "ses-02",
                    "task": "passive",
                    "modality": "audio",
                    "coarse_affect": "positive",
                },
                {
                    "sample_id": "s4",
                    "subject": "sub-001",
                    "session": "ses-03",
                    "task": "passive",
                    "modality": "audio",
                    "coarse_affect": "positive",
                },
            ]
        )
        for field_name in FEATURE_QC_SAMPLE_FIELDS:
            metadata_df[field_name] = 0.0
        x_matrix = np.asarray(
            [
                [1.0, 0.0],
                [0.8, 0.1],
                [-1.0, 0.0],
                [-0.9, -0.1],
            ],
            dtype=np.float64,
        )
        y = metadata_df["coarse_affect"].astype(str).to_numpy()
        y_true_all = ["negative", "negative", "positive", "positive"]
        y_pred_all = ["negative", "negative", "positive", "negative"]
        fold_rows = [
            {"fold": 0, "n_train": 1, "n_test": 3, "accuracy": 1.0, "balanced_accuracy": 1.0, "macro_f1": 1.0},
            {"fold": 1, "n_train": 3, "n_test": 1, "accuracy": 0.0, "balanced_accuracy": 0.0, "macro_f1": 0.0},
        ]
        split_rows = [
            {"fold": 0, "partition": "train", "sample_id": "s4"},
            {"fold": 1, "partition": "train", "sample_id": "s1"},
        ]
        prediction_rows = [
            {"fold": 0, "sample_id": "s1", "y_true": "negative", "y_pred": "negative"},
            {"fold": 0, "sample_id": "s2", "y_true": "negative", "y_pred": "negative"},
            {"fold": 0, "sample_id": "s3", "y_true": "positive", "y_pred": "positive"},
            {"fold": 1, "sample_id": "s4", "y_true": "positive", "y_pred": "negative"},
        ]
        return EvaluationInput(
            x_matrix=x_matrix,
            metadata_df=metadata_df,
            y=y,
            splits=[
                (np.asarray([3], dtype=np.int64), np.asarray([0, 1, 2], dtype=np.int64)),
                (np.asarray([0, 1, 2], dtype=np.int64), np.asarray([3], dtype=np.int64)),
            ],
            fold_rows=fold_rows,
            split_rows=split_rows,
            prediction_rows=prediction_rows,
            y_true_all=y_true_all,
            y_pred_all=y_pred_all,
            subject="sub-001",
            n_permutations=0,
            primary_metric_name="accuracy",
            primary_metric_aggregation=aggregation,
            subgroup_reporting_enabled=False,
            calibration_enabled=False,
            emit_feature_qc_artifacts=False,
            spatial_compatibility={
                "status": "passed",
                "passed": True,
                "n_groups_checked": 2,
                "reference_group_id": "sub-001_ses-01",
                "affine_atol": 1e-5,
            },
            spatial_report_path=report_dir / "spatial_compatibility_report.json",
            interpretability_summary={
                "enabled": False,
                "performed": False,
                "status": "not_applicable",
                "fold_artifacts_path": str(
                    (report_dir / "interpretability_fold_explanations.csv").resolve()
                ),
                "stability": {},
            },
            interpretability_summary_path=report_dir / "interpretability_summary.json",
            fold_metrics_path=report_dir / "fold_metrics.csv",
            fold_splits_path=report_dir / "fold_splits.csv",
            predictions_path=report_dir / "predictions.csv",
            config_filename="config.json",
            build_pipeline_fn=_build_pipeline,
            evaluate_permutations_fn=_evaluate_permutations,
            run_id=f"aggregation_{aggregation}",
            artifact_registry_path=tmp_path / "artifact_registry.sqlite3",
            upstream_feature_matrix_artifact_id="feature_matrix_bundle_test",
            metrics_path=report_dir / "metrics.json",
            model="ridge",
            target_column="coarse_affect",
            cv_mode="within_subject_loso_session",
            seed=17,
        )

    mean_metrics = evaluation(_build_input(aggregation="mean_fold_scores")).metrics
    pooled_metrics = evaluation(_build_input(aggregation="pooled_held_out_predictions")).metrics

    assert mean_metrics["primary_metric_value_mean_fold"] != mean_metrics["primary_metric_value_pooled"]
    assert mean_metrics["primary_metric_value"] == pytest.approx(
        mean_metrics["primary_metric_value_mean_fold"]
    )
    assert pooled_metrics["primary_metric_value"] == pytest.approx(
        pooled_metrics["primary_metric_value_pooled"]
    )


def test_model_fit_record_random_split_generates_non_empty_folds(tmp_path: Path) -> None:
    report_dir = tmp_path / "record_split_fit"
    report_dir.mkdir(parents=True, exist_ok=True)
    metadata_df = pd.DataFrame(
        [
            {
                "sample_id": "s1",
                "subject": "sub-001",
                "session": "ses-01",
                "bas": "BAS2",
                "task": "passive",
                "modality": "audio",
                "emotion": "anger",
                "coarse_affect": "negative",
            },
            {
                "sample_id": "s2",
                "subject": "sub-001",
                "session": "ses-01",
                "bas": "BAS2",
                "task": "passive",
                "modality": "video",
                "emotion": "happiness",
                "coarse_affect": "positive",
            },
            {
                "sample_id": "s3",
                "subject": "sub-001",
                "session": "ses-02",
                "bas": "BAS2",
                "task": "passive",
                "modality": "audio",
                "emotion": "anger",
                "coarse_affect": "negative",
            },
            {
                "sample_id": "s4",
                "subject": "sub-001",
                "session": "ses-02",
                "bas": "BAS2",
                "task": "passive",
                "modality": "video",
                "emotion": "happiness",
                "coarse_affect": "positive",
            },
            {
                "sample_id": "s5",
                "subject": "sub-002",
                "session": "ses-01",
                "bas": "BAS2",
                "task": "passive",
                "modality": "audio",
                "emotion": "anger",
                "coarse_affect": "negative",
            },
            {
                "sample_id": "s6",
                "subject": "sub-002",
                "session": "ses-01",
                "bas": "BAS2",
                "task": "passive",
                "modality": "video",
                "emotion": "happiness",
                "coarse_affect": "positive",
            },
            {
                "sample_id": "s7",
                "subject": "sub-002",
                "session": "ses-02",
                "bas": "BAS2",
                "task": "passive",
                "modality": "audio",
                "emotion": "anger",
                "coarse_affect": "negative",
            },
            {
                "sample_id": "s8",
                "subject": "sub-002",
                "session": "ses-02",
                "bas": "BAS2",
                "task": "passive",
                "modality": "video",
                "emotion": "happiness",
                "coarse_affect": "positive",
            },
        ]
    )
    x_matrix = np.asarray(
        [
            [4.2, 0.2, 0.1, 0.0],
            [-4.1, -0.2, -0.1, 0.0],
            [4.0, 0.1, 0.2, 0.1],
            [-4.0, -0.1, -0.2, -0.1],
            [4.1, 0.3, 0.2, 0.2],
            [-4.2, -0.3, -0.2, -0.2],
            [4.3, 0.2, 0.3, 0.1],
            [-4.3, -0.2, -0.3, -0.1],
        ],
        dtype=np.float32,
    )

    fit_output = model_fit(
        ModelFitInput(
            x_matrix=x_matrix,
            metadata_df=metadata_df,
            target_column="coarse_affect",
            cv_mode="record_random_split",
            model="ridge",
            seed=19,
            run_id="record_random_split_model_fit",
            config_filename="config.json",
            report_dir=report_dir,
            build_pipeline_fn=_build_pipeline,
            scores_for_predictions_fn=_scores_for_predictions,
            extract_linear_coefficients_fn=_extract_linear_coefficients,
        )
    )

    assert fit_output.interpretability_enabled is False
    assert len(fit_output.splits) == 4
    for train_idx, test_idx in fit_output.splits:
        assert int(len(train_idx)) > 0
        assert int(len(test_idx)) > 0


def test_model_fit_emits_fold_progress_events(tmp_path: Path) -> None:
    events: list[ProgressEvent] = []

    def _capture(event: ProgressEvent) -> None:
        events.append(event)

    report_dir = tmp_path / "run"
    report_dir.mkdir(parents=True, exist_ok=True)

    metadata_df = pd.DataFrame(
        [
            {
                "sample_id": "s1",
                "subject": "sub-001",
                "session": "ses-01",
                "bas": "BAS2",
                "task": "passive",
                "modality": "audio",
                "emotion": "anger",
                "coarse_affect": "negative",
            },
            {
                "sample_id": "s2",
                "subject": "sub-001",
                "session": "ses-01",
                "bas": "BAS2",
                "task": "passive",
                "modality": "video",
                "emotion": "happiness",
                "coarse_affect": "positive",
            },
            {
                "sample_id": "s3",
                "subject": "sub-001",
                "session": "ses-02",
                "bas": "BAS2",
                "task": "passive",
                "modality": "audio",
                "emotion": "anger",
                "coarse_affect": "negative",
            },
            {
                "sample_id": "s4",
                "subject": "sub-001",
                "session": "ses-02",
                "bas": "BAS2",
                "task": "passive",
                "modality": "video",
                "emotion": "happiness",
                "coarse_affect": "positive",
            },
        ]
    )
    x_matrix = np.asarray(
        [
            [0.0, 0.1, 0.0, 0.1],
            [0.1, 0.0, 0.1, 0.0],
            [2.0, 2.1, 2.0, 2.1],
            [2.1, 2.0, 2.1, 2.0],
        ],
        dtype=np.float64,
    )

    fit_output = model_fit(
        ModelFitInput(
            x_matrix=x_matrix,
            metadata_df=metadata_df,
            target_column="coarse_affect",
            cv_mode="within_subject_loso_session",
            model="ridge",
            subject="sub-001",
            seed=7,
            run_id="progress_test",
            config_filename="config.json",
            report_dir=report_dir,
            build_pipeline_fn=_build_pipeline,
            scores_for_predictions_fn=_scores_for_predictions,
            extract_linear_coefficients_fn=_extract_linear_coefficients,
            progress_callback=_capture,
        )
    )

    assert fit_output.fold_rows
    assert events
    fold_events = [event for event in events if event.stage == "fold"]
    assert fold_events
    assert any("starting outer fold" in event.message for event in fold_events)
    assert any("finished outer fold" in event.message for event in fold_events)


def test_dataset_selection_rejects_unsupported_emotion_for_coarse_affect(
    tmp_path: Path,
) -> None:
    index_csv = tmp_path / "dataset_index.csv"
    index_df = pd.DataFrame(
        [
            {
                "sample_id": "s1",
                "subject": "sub-001",
                "session": "ses-01",
                "task": "passive",
                "modality": "audio",
                "emotion": "joy",
            },
            {
                "sample_id": "s2",
                "subject": "sub-001",
                "session": "ses-02",
                "task": "passive",
                "modality": "video",
                "emotion": "happiness",
            },
        ]
    )
    index_df.to_csv(index_csv, index=False)

    with pytest.raises(ValueError, match="unsupported or missing source labels"):
        dataset_selection(
            DatasetSelectionInput(
                index_csv=index_csv,
                target_column="coarse_affect",
                cv_mode="within_subject_loso_session",
                subject="sub-001",
            )
        )


def test_dataset_selection_rejects_unsupported_upstream_emotion_for_binary_valence_like(
    tmp_path: Path,
) -> None:
    index_csv = tmp_path / "dataset_index.csv"
    index_df = pd.DataFrame(
        [
            {
                "sample_id": "s1",
                "subject": "sub-001",
                "session": "ses-01",
                "task": "passive",
                "modality": "audio",
                "emotion": "joy",
            },
            {
                "sample_id": "s2",
                "subject": "sub-001",
                "session": "ses-02",
                "task": "passive",
                "modality": "video",
                "emotion": "happiness",
            },
        ]
    )
    index_df.to_csv(index_csv, index=False)

    with pytest.raises(ValueError, match="unsupported or missing source labels"):
        dataset_selection(
            DatasetSelectionInput(
                index_csv=index_csv,
                target_column="binary_valence_like",
                cv_mode="within_subject_loso_session",
                subject="sub-001",
            )
        )


def test_dataset_selection_records_task_filter_exclusions(tmp_path: Path) -> None:
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
                "task": "emo",
                "modality": "audio",
                "emotion": "happiness",
            },
        ]
    )
    index_df.to_csv(index_csv, index=False)

    selection_output = dataset_selection(
        DatasetSelectionInput(
            index_csv=index_csv,
            target_column="coarse_affect",
            cv_mode="within_subject_loso_session",
            subject="sub-001",
            filter_task="passive",
        )
    )

    selected = selection_output.selected_index_df
    exclusion_manifest = selection_output.selection_exclusion_manifest_df

    assert set(selected["sample_id"].astype(str).tolist()) == {"s1"}
    assert int(len(exclusion_manifest)) == 1
    assert exclusion_manifest.iloc[0]["sample_id"] == "s2"
    assert exclusion_manifest.iloc[0]["exclusion_stage"] == "filter_task"
    assert exclusion_manifest.iloc[0]["exclusion_reason"] == "task_mismatch"


def test_grouped_nested_tuned_permutation_reapplies_tuning_under_null(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evaluation_input = _build_grouped_nested_evaluation_input(tmp_path=tmp_path, evidence_run_role="primary")
    tuning_calls: list[dict[str, object]] = []

    def _fake_tuning_executor(**kwargs: object) -> dict[str, object]:
        tuning_calls.append(dict(kwargs))
        estimator = clone(kwargs["pipeline_template"])
        estimator.fit(kwargs["x_outer_train"], kwargs["y_outer_train"])
        return {"estimator": estimator}

    def _unexpected_permutation_executor(**_: object) -> dict[str, object]:
        raise AssertionError("run_permutation_executor should not be used for tuned primary runs.")

    monkeypatch.setattr(sections_impl_module, "run_tuning_executor", _fake_tuning_executor)
    monkeypatch.setattr(sections_impl_module, "run_permutation_executor", _unexpected_permutation_executor)

    result = evaluation(evaluation_input)
    permutation_payload = result.metrics.get("permutation_test")

    assert isinstance(permutation_payload, dict)
    assert len(tuning_calls) == int(evaluation_input.n_permutations) * len(evaluation_input.splits)
    assert all(len(call["inner_splits"]) >= 2 for call in tuning_calls)
    assert all(bool(call["param_grid"]) for call in tuning_calls)
    assert permutation_payload["execution_mode"] == "grouped_nested_tuning_reference"
    assert permutation_payload["tuning_reapplied_under_null"] is True
    assert permutation_payload["null_matches_confirmatory_setup"] is True
    assert permutation_payload["null_tuning_search_space_id"] == LINEAR_GROUPED_NESTED_SEARCH_SPACE_ID
    assert (
        permutation_payload["null_tuning_search_space_version"]
        == LINEAR_GROUPED_NESTED_SEARCH_SPACE_VERSION
    )
    assert permutation_payload["null_inner_cv_scheme"] == "grouped_leave_one_group_out"
    assert permutation_payload["null_inner_group_field"] == "session"
    assert permutation_payload["permutation_executor_id"] == PERMUTATION_REFERENCE_EXECUTOR_ID


def test_grouped_nested_tuned_permutation_ridge_specialized_matches_generic_null_path(
    tmp_path: Path,
) -> None:
    specialized_input = _build_grouped_nested_evaluation_input(
        tmp_path=tmp_path,
        evidence_run_role="ridge_specialized",
        n_permutations=3,
        model="ridge",
    )
    specialized_result = evaluation(specialized_input)
    specialized_payload = specialized_result.metrics.get("permutation_test")
    assert isinstance(specialized_payload, dict)
    assert specialized_payload.get("null_tuning_executor_id") == SPECIALIZED_RIDGE_TUNING_EXECUTOR_ID

    generic_input = _build_grouped_nested_evaluation_input(
        tmp_path=tmp_path,
        evidence_run_role="ridge_generic",
        n_permutations=3,
        model="ridge",
    ).model_copy(
        update={
            "tuning_assignment": StageAssignment(
                stage=StageKey.TUNING,
                backend_family=StageBackendFamily.SKLEARN_CPU,
                compute_lane="cpu",
                source="stage_planner_v1",
                reason="test_force_generic_null_tuning",
                executor_id=TUNING_GENERIC_EXECUTOR_ID,
                equivalence_class="exact_reference_equivalent",
                official_admitted=True,
                fallback_used=False,
                fallback_reason=None,
            ),
            "tuning_fallback_executor_id": None,
        }
    )
    generic_result = evaluation(generic_input)
    generic_payload = generic_result.metrics.get("permutation_test")
    assert isinstance(generic_payload, dict)
    assert generic_payload.get("null_tuning_executor_id") == TUNING_GENERIC_EXECUTOR_ID

    parity = compare_tuned_null_parity(
        reference_payload=generic_payload,
        candidate_payload=specialized_payload,
        category="exact",
    )
    assert parity.passed is True


def test_untuned_baseline_still_uses_reference_permutation_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evaluation_input = _build_grouped_nested_evaluation_input(
        tmp_path=tmp_path,
        evidence_run_role="untuned_baseline",
    )
    called = {"permutation": False}

    def _fake_permutation_executor(**_: object) -> dict[str, object]:
        called["permutation"] = True
        return {
            "metric_name": "balanced_accuracy",
            "n_permutations": int(evaluation_input.n_permutations),
            "observed_score": 0.5,
            "permutation_seed": int(evaluation_input.seed),
            "p_value": 1.0,
            "null_summary": {
                "mean": 0.5,
                "std": 0.0,
                "min": 0.5,
                "max": 0.5,
                "q25": 0.5,
                "q50": 0.5,
                "q75": 0.5,
            },
            "null_scores": [0.5] * int(evaluation_input.n_permutations),
            "execution_mode": "cached_scaled_cpu",
            "permutation_executor_id": PERMUTATION_REFERENCE_EXECUTOR_ID,
            "permutation_executor_fallback_reason": None,
        }

    def _unexpected_tuning_executor(**_: object) -> dict[str, object]:
        raise AssertionError("run_tuning_executor should not be used for untuned_baseline permutations.")

    monkeypatch.setattr(sections_impl_module, "run_permutation_executor", _fake_permutation_executor)
    monkeypatch.setattr(sections_impl_module, "run_tuning_executor", _unexpected_tuning_executor)

    result = evaluation(evaluation_input)
    permutation_payload = result.metrics.get("permutation_test")

    assert called["permutation"] is True
    assert isinstance(permutation_payload, dict)
    assert permutation_payload["execution_mode"] == "cached_scaled_cpu"
