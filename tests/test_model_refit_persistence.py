from __future__ import annotations

import json
from pathlib import Path

from Thesis_ML.config.metric_policy import EffectiveMetricPolicy
from Thesis_ML.experiments.run_artifacts import RunIdentity, stamp_metrics_artifact


def test_metrics_include_model_persistence_block_when_enabled(tmp_path: Path) -> None:
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(
        json.dumps({"model": "ridge", "balanced_accuracy": 0.75}),
        encoding="utf-8",
    )

    tuning_summary_path = tmp_path / "tuning_summary.json"
    tuning_summary_path.write_text("{}\n", encoding="utf-8")
    tuning_best_params_path = tmp_path / "best_params_per_fold.csv"
    tuning_best_params_path.write_text("fold,best_params_json\n", encoding="utf-8")
    fit_timing_summary_path = tmp_path / "fit_timing_summary.json"
    fit_timing_summary_path.write_text("{}\n", encoding="utf-8")
    feature_qc_summary_path = tmp_path / "feature_qc_summary.json"
    feature_qc_summary_path.write_text("{}\n", encoding="utf-8")
    feature_qc_selected_samples_path = tmp_path / "feature_qc_selected_samples.csv"
    feature_qc_selected_samples_path.write_text("sample_id\n", encoding="utf-8")
    subgroup_metrics_json_path = tmp_path / "subgroup_metrics.json"
    subgroup_metrics_json_path.write_text("{}\n", encoding="utf-8")
    subgroup_metrics_csv_path = tmp_path / "subgroup_metrics.csv"
    subgroup_metrics_csv_path.write_text("subgroup_key\n", encoding="utf-8")

    metric_policy = EffectiveMetricPolicy(
        primary_metric="balanced_accuracy",
        secondary_metrics=("macro_f1",),
        decision_metric="balanced_accuracy",
        tuning_metric="balanced_accuracy",
        permutation_metric="balanced_accuracy",
        higher_is_better=True,
    )

    payload = stamp_metrics_artifact(
        metrics_path=metrics_path,
        canonical_run=False,
        framework_mode="exploratory",
        repeat_id=1,
        repeat_count=1,
        base_run_id="run_1",
        evidence_run_role="primary",
        evidence_policy_effective={},
        methodology_policy_name="fixed_baselines_only",
        class_weight_policy="none",
        tuning_enabled=False,
        model_cost_tier="medium",
        projected_runtime_seconds=30,
        primary_metric_aggregation="mean_fold_scores",
        preprocessing_kind="standard_scaler",
        preprocessing_strategy="none",
        feature_recipe_id="baseline_standard_scaler_v1",
        tuning_summary_path=tuning_summary_path,
        tuning_best_params_path=tuning_best_params_path,
        fit_timing_summary_path=fit_timing_summary_path,
        feature_qc_summary_path=feature_qc_summary_path,
        feature_qc_selected_samples_path=feature_qc_selected_samples_path,
        subgroup_metrics_json_path=subgroup_metrics_json_path,
        subgroup_metrics_csv_path=subgroup_metrics_csv_path,
        metric_policy_effective=metric_policy,
        data_policy_effective=None,
        data_artifacts=None,
        identity=RunIdentity(
            protocol_id=None,
            protocol_version=None,
            protocol_schema_version=None,
            suite_id=None,
            claim_ids=None,
            comparison_id=None,
            comparison_version=None,
            comparison_variant_id=None,
        ),
        model_persistence={
            "enabled": True,
            "fold_models_saved": True,
            "n_fold_models": 2,
            "model_summary_path": str(tmp_path / "models" / "model_summary.json"),
            "model_artifacts_csv_path": str(tmp_path / "models" / "model_artifacts.csv"),
            "final_refit_saved": True,
            "final_refit_model_path": str(tmp_path / "models" / "final_refit_model.joblib"),
            "artifact_ids": {
                "model_bundle": "model_bundle_x",
                "model_refit_bundle": "model_refit_bundle_x",
            },
        },
    )

    assert isinstance(payload, dict)
    assert "model_persistence" in payload
    assert payload["model_persistence"]["enabled"] is True
