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
        decision_policy=spec.decision_policy,
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
