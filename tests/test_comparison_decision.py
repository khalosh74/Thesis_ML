from __future__ import annotations

from Thesis_ML.comparisons.artifacts import build_comparison_decision
from Thesis_ML.comparisons.models import (
    ComparisonRunResult,
    ComparisonSpec,
    ComparisonStatus,
    CompiledComparisonManifest,
    CompiledComparisonRunControls,
    CompiledComparisonRunSpec,
)
from Thesis_ML.config.framework_mode import FrameworkMode


def _base_comparison_spec() -> ComparisonSpec:
    return ComparisonSpec.model_validate(
        {
            "comparison_schema_version": "comparison-spec-v1",
            "framework_mode": "locked_comparison",
            "comparison_id": "cmp-decision-test",
            "comparison_version": "1.0.0",
            "status": "locked",
            "description": "Decision logic test comparison.",
            "comparison_dimension": "model_family",
            "scientific_contract": {
                "target": "coarse_affect",
                "split_mode": "within_subject_loso_session",
                "grouping_policy": "session",
                "seed_policy": {"global_seed": 42},
                "subject_policy": {"source": "explicit", "subjects": ["sub-001"]},
                "transfer_policy": {"source": "all_ordered_pairs_from_index", "pairs": []},
            },
            "methodology_policy": {
                "policy_name": "fixed_baselines_only",
                "class_weight_policy": "none",
                "tuning_enabled": False,
            },
            "metric_policy": {
                "primary_metric": "balanced_accuracy",
                "secondary_metrics": ["macro_f1", "accuracy"],
            },
            "control_policy": {
                "permutation_enabled": False,
                "permutation_metric": "balanced_accuracy",
                "n_permutations": 0,
                "dummy_baseline_enabled": False,
            },
            "subgroup_reporting_policy": {
                "enabled": True,
                "subgroup_dimensions": ["label", "task", "modality", "session", "subject"],
                "min_samples_per_group": 1,
            },
            "decision_policy": {
                "primary_metric": "balanced_accuracy",
                "require_all_runs_completed": True,
                "invalid_on_missing_metrics": True,
                "require_permutation_control_pass": False,
                "permutation_p_value_threshold": 0.05,
                "tie_tolerance": 1e-09,
                "status_on_tie": "inconclusive",
                "allow_mixed_methodology_policies": False,
                "block_on_subgroup_failures": False,
            },
            "evidence_policy": {
                "repeat_evaluation": {"repeat_count": 1, "seed_stride": 1000},
                "confidence_intervals": {
                    "method": "grouped_bootstrap_percentile",
                    "confidence_level": 0.95,
                    "n_bootstrap": 1000,
                    "seed": 2026,
                },
                "paired_comparisons": {
                    "method": "paired_sign_flip_permutation",
                    "n_permutations": 2000,
                    "alpha": 0.05,
                    "require_significant_win": False,
                },
                "permutation": {
                    "alpha": 0.05,
                    "minimum_permutations": 0,
                    "require_pass_for_validity": False,
                },
                "calibration": {
                    "enabled": True,
                    "n_bins": 10,
                    "require_probabilities_for_validity": False,
                },
                "required_package": {
                    "require_dummy_baseline": False,
                    "require_permutation_control": False,
                    "require_untuned_baseline_if_tuning": False,
                },
            },
            "interpretability_policy": {"enabled": False, "allowed_models": []},
            "allowed_variants": [
                {"variant_id": "ridge", "model": "ridge", "claim_ids": ["c1"]},
                {"variant_id": "logreg", "model": "logreg", "claim_ids": ["c2"]},
            ],
        }
    )


def _compiled_manifest() -> CompiledComparisonManifest:
    controls = CompiledComparisonRunControls(
        permutation_enabled=False,
        permutation_metric="balanced_accuracy",
        n_permutations=0,
        dummy_baseline_enabled=False,
    )
    runs = [
        CompiledComparisonRunSpec(
            run_id="r1",
            base_run_id="r1",
            framework_mode=FrameworkMode.LOCKED_COMPARISON.value,
            canonical_run=False,
            comparison_id="cmp-decision-test",
            comparison_version="1.0.0",
            variant_id="ridge",
            claim_ids=["c1"],
            target="coarse_affect",
            model="ridge",
            cv_mode="within_subject_loso_session",
            subject="sub-001",
            seed=42,
            primary_metric="balanced_accuracy",
            controls=controls,
            interpretability_enabled=False,
            methodology_policy_name="fixed_baselines_only",
            class_weight_policy="none",
            tuning_enabled=False,
            subgroup_reporting_enabled=True,
            subgroup_dimensions=["label", "task", "modality", "session", "subject"],
            subgroup_min_samples_per_group=1,
        ),
        CompiledComparisonRunSpec(
            run_id="r2",
            base_run_id="r2",
            framework_mode=FrameworkMode.LOCKED_COMPARISON.value,
            canonical_run=False,
            comparison_id="cmp-decision-test",
            comparison_version="1.0.0",
            variant_id="logreg",
            claim_ids=["c2"],
            target="coarse_affect",
            model="logreg",
            cv_mode="within_subject_loso_session",
            subject="sub-001",
            seed=42,
            primary_metric="balanced_accuracy",
            controls=controls,
            interpretability_enabled=False,
            methodology_policy_name="fixed_baselines_only",
            class_weight_policy="none",
            tuning_enabled=False,
            subgroup_reporting_enabled=True,
            subgroup_dimensions=["label", "task", "modality", "session", "subject"],
            subgroup_min_samples_per_group=1,
        ),
    ]
    spec = _base_comparison_spec()
    return CompiledComparisonManifest(
        framework_mode=FrameworkMode.LOCKED_COMPARISON.value,
        comparison_id="cmp-decision-test",
        comparison_version="1.0.0",
        status=ComparisonStatus.LOCKED,
        comparison_dimension="model_family",
        methodology_policy=spec.methodology_policy,
        metric_policy=spec.metric_policy,
        subgroup_reporting_policy=spec.subgroup_reporting_policy,
        data_policy=spec.data_policy,
        decision_policy=spec.decision_policy,
        evidence_policy=spec.evidence_policy,
        variant_ids=["ridge", "logreg"],
        runs=runs,
        claim_to_run_map={"c1": ["r1"], "c2": ["r2"]},
    )


def test_comparison_decision_winner_selected() -> None:
    spec = _base_comparison_spec()
    manifest = _compiled_manifest()
    run_results = [
        ComparisonRunResult(
            run_id="r1",
            framework_mode=FrameworkMode.LOCKED_COMPARISON.value,
            comparison_id="cmp-decision-test",
            comparison_version="1.0.0",
            variant_id="ridge",
            status="completed",
            metrics={"balanced_accuracy": 0.80},
        ),
        ComparisonRunResult(
            run_id="r2",
            framework_mode=FrameworkMode.LOCKED_COMPARISON.value,
            comparison_id="cmp-decision-test",
            comparison_version="1.0.0",
            variant_id="logreg",
            status="completed",
            metrics={"balanced_accuracy": 0.60},
        ),
    ]
    decision = build_comparison_decision(
        comparison=spec,
        compiled_manifest=manifest,
        run_results=run_results,
    )
    assert decision["decision_status"] == "winner_selected"
    assert decision["selected_variant"] == "ridge"


def test_comparison_decision_inconclusive_on_tie() -> None:
    spec = _base_comparison_spec()
    manifest = _compiled_manifest()
    run_results = [
        ComparisonRunResult(
            run_id="r1",
            framework_mode=FrameworkMode.LOCKED_COMPARISON.value,
            comparison_id="cmp-decision-test",
            comparison_version="1.0.0",
            variant_id="ridge",
            status="completed",
            metrics={"balanced_accuracy": 0.70},
        ),
        ComparisonRunResult(
            run_id="r2",
            framework_mode=FrameworkMode.LOCKED_COMPARISON.value,
            comparison_id="cmp-decision-test",
            comparison_version="1.0.0",
            variant_id="logreg",
            status="completed",
            metrics={"balanced_accuracy": 0.70},
        ),
    ]
    decision = build_comparison_decision(
        comparison=spec,
        compiled_manifest=manifest,
        run_results=run_results,
    )
    assert decision["decision_status"] == "inconclusive"
    assert decision["selected_variant"] is None


def test_comparison_decision_invalid_on_missing_metrics() -> None:
    spec = _base_comparison_spec()
    manifest = _compiled_manifest()
    run_results = [
        ComparisonRunResult(
            run_id="r1",
            framework_mode=FrameworkMode.LOCKED_COMPARISON.value,
            comparison_id="cmp-decision-test",
            comparison_version="1.0.0",
            variant_id="ridge",
            status="completed",
            metrics={},
        ),
        ComparisonRunResult(
            run_id="r2",
            framework_mode=FrameworkMode.LOCKED_COMPARISON.value,
            comparison_id="cmp-decision-test",
            comparison_version="1.0.0",
            variant_id="logreg",
            status="completed",
            metrics={"balanced_accuracy": 0.61},
        ),
    ]
    decision = build_comparison_decision(
        comparison=spec,
        compiled_manifest=manifest,
        run_results=run_results,
    )
    assert decision["decision_status"] == "invalid_comparison"


def test_comparison_decision_uses_declared_primary_metric_for_selection() -> None:
    spec = _base_comparison_spec().model_copy(deep=True)
    spec.metric_policy.primary_metric = "macro_f1"
    spec.metric_policy.secondary_metrics = ["balanced_accuracy", "accuracy"]
    spec.decision_policy.primary_metric = "macro_f1"
    spec.control_policy.permutation_metric = "macro_f1"

    manifest = _compiled_manifest().model_copy(deep=True)
    manifest.metric_policy.primary_metric = "macro_f1"
    manifest.metric_policy.secondary_metrics = ["balanced_accuracy", "accuracy"]
    manifest.decision_policy.primary_metric = "macro_f1"
    for run in manifest.runs:
        run.primary_metric = "macro_f1"
        run.controls.permutation_metric = "macro_f1"

    run_results = [
        ComparisonRunResult(
            run_id="r1",
            framework_mode=FrameworkMode.LOCKED_COMPARISON.value,
            comparison_id="cmp-decision-test",
            comparison_version="1.0.0",
            variant_id="ridge",
            status="completed",
            metrics={"balanced_accuracy": 0.90, "macro_f1": 0.40},
        ),
        ComparisonRunResult(
            run_id="r2",
            framework_mode=FrameworkMode.LOCKED_COMPARISON.value,
            comparison_id="cmp-decision-test",
            comparison_version="1.0.0",
            variant_id="logreg",
            status="completed",
            metrics={"balanced_accuracy": 0.60, "macro_f1": 0.70},
        ),
    ]
    decision = build_comparison_decision(
        comparison=spec,
        compiled_manifest=manifest,
        run_results=run_results,
    )
    assert decision["decision_status"] == "winner_selected"
    assert decision["primary_metric"] == "macro_f1"
    assert decision["selected_variant"] == "logreg"
