from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from Thesis_ML.config.methodology import MethodologyPolicy, MetricPolicy, SubgroupReportingPolicy
from Thesis_ML.experiments.run_experiment import (
    _build_pipeline,
    _extract_linear_coefficients,
    _scores_for_predictions,
)
from Thesis_ML.experiments.sections import ModelFitInput, model_fit


def _metadata() -> pd.DataFrame:
    return pd.DataFrame(
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


def _x_matrix() -> np.ndarray:
    return np.asarray(
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


def test_methodology_policy_contracts_validate() -> None:
    MethodologyPolicy.model_validate(
        {
            "policy_name": "fixed_baselines_only",
            "class_weight_policy": "none",
            "tuning_enabled": False,
        }
    )
    MethodologyPolicy.model_validate(
        {
            "policy_name": "grouped_nested_tuning",
            "class_weight_policy": "balanced",
            "tuning_enabled": True,
            "inner_cv_scheme": "grouped_leave_one_group_out",
            "inner_group_field": "session",
            "tuning_search_space_id": "linear-grouped-nested-v1",
            "tuning_search_space_version": "1.0.0",
        }
    )
    MetricPolicy.model_validate({"primary_metric": "balanced_accuracy"})
    SubgroupReportingPolicy.model_validate(
        {"enabled": True, "subgroup_dimensions": ["label", "task"], "min_samples_per_group": 1}
    )


def test_fixed_policy_rejects_tuning_configuration() -> None:
    with pytest.raises(ValueError, match="forbids tuning_enabled=true"):
        MethodologyPolicy.model_validate(
            {
                "policy_name": "fixed_baselines_only",
                "class_weight_policy": "none",
                "tuning_enabled": True,
            }
        )


def test_grouped_nested_tuning_writes_tuning_artifacts(tmp_path: Path) -> None:
    report_dir = tmp_path / "run"
    report_dir.mkdir(parents=True, exist_ok=True)
    tuning_summary_path = report_dir / "tuning_summary.json"
    tuning_params_path = report_dir / "best_params_per_fold.csv"

    fit_output = model_fit(
        ModelFitInput(
            x_matrix=_x_matrix(),
            metadata_df=_metadata(),
            target_column="coarse_affect",
            cv_mode="within_subject_loso_session",
            model="ridge",
            subject="sub-001",
            seed=11,
            primary_metric_name="balanced_accuracy",
            methodology_policy_name="grouped_nested_tuning",
            class_weight_policy="none",
            tuning_enabled=True,
            tuning_search_space_id="linear-grouped-nested-v1",
            tuning_search_space_version="1.0.0",
            tuning_inner_cv_scheme="grouped_leave_one_group_out",
            tuning_inner_group_field="session",
            tuning_summary_path=tuning_summary_path,
            tuning_best_params_path=tuning_params_path,
            run_id="nested_tuning_unit",
            config_filename="config.json",
            report_dir=report_dir,
            build_pipeline_fn=_build_pipeline,
            scores_for_predictions_fn=_scores_for_predictions,
            extract_linear_coefficients_fn=_extract_linear_coefficients,
        )
    )
    assert fit_output.tuning_summary_path.exists()
    assert fit_output.tuning_best_params_path.exists()
    summary = json.loads(fit_output.tuning_summary_path.read_text(encoding="utf-8"))
    assert summary["methodology_policy_name"] == "grouped_nested_tuning"
    assert summary["tuning_enabled"] is True
    assert int(summary["n_tuned_folds"]) > 0
    params_df = pd.read_csv(fit_output.tuning_best_params_path)
    assert set(params_df["status"].astype(str).tolist()) == {"tuned"}
