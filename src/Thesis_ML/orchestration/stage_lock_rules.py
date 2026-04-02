from __future__ import annotations

from dataclasses import dataclass
from typing import Any

DEFAULT_MIN_MARGIN_BALANCED_ACCURACY = 0.02
DEFAULT_MIN_BASELINE_DELTA_BALANCED_ACCURACY = 0.02


@dataclass(frozen=True)
class PreflightStageRule:
    experiment_id: str
    expected_completed_variants: int
    manipulated_factor_fields: tuple[str, ...]
    comparison_slice_fields: tuple[str, ...]
    min_margin_balanced_accuracy: float = DEFAULT_MIN_MARGIN_BALANCED_ACCURACY
    min_baseline_delta_balanced_accuracy: float = DEFAULT_MIN_BASELINE_DELTA_BALANCED_ACCURACY
    auto_lock_allowed: bool = True


_PREFLIGHT_RULES: dict[str, PreflightStageRule] = {
    "E01": PreflightStageRule(
        experiment_id="E01",
        expected_completed_variants=6,
        manipulated_factor_fields=("target",),
        comparison_slice_fields=("subject",),
    ),
    "E02": PreflightStageRule(
        experiment_id="E02",
        expected_completed_variants=8,
        manipulated_factor_fields=("task_pooling_choice",),
        comparison_slice_fields=("subject",),
        auto_lock_allowed=False,
    ),
    "E03": PreflightStageRule(
        experiment_id="E03",
        expected_completed_variants=8,
        manipulated_factor_fields=("modality_pooling_choice",),
        comparison_slice_fields=("subject",),
        auto_lock_allowed=False,
    ),
    "E04": PreflightStageRule(
        experiment_id="E04",
        expected_completed_variants=8,
        manipulated_factor_fields=("cv",),
        comparison_slice_fields=("subject", "filter_task"),
    ),
    "E05": PreflightStageRule(
        experiment_id="E05",
        expected_completed_variants=4,
        manipulated_factor_fields=("transfer_direction",),
        comparison_slice_fields=("filter_task",),
    ),
    "E06": PreflightStageRule(
        experiment_id="E06",
        expected_completed_variants=12,
        manipulated_factor_fields=("model",),
        comparison_slice_fields=("subject", "filter_task"),
    ),
    "E07": PreflightStageRule(
        experiment_id="E07",
        expected_completed_variants=8,
        manipulated_factor_fields=("class_weight_policy",),
        comparison_slice_fields=("subject", "filter_task"),
    ),
    "E08": PreflightStageRule(
        experiment_id="E08",
        expected_completed_variants=8,
        manipulated_factor_fields=("methodology_policy_name",),
        comparison_slice_fields=("subject", "filter_task"),
    ),
    "E09": PreflightStageRule(
        experiment_id="E09",
        expected_completed_variants=8,
        manipulated_factor_fields=("feature_space",),
        comparison_slice_fields=("subject", "filter_task"),
    ),
    "E10": PreflightStageRule(
        experiment_id="E10",
        expected_completed_variants=8,
        manipulated_factor_fields=("dimensionality_strategy",),
        comparison_slice_fields=("subject", "filter_task"),
    ),
    "E11": PreflightStageRule(
        experiment_id="E11",
        expected_completed_variants=8,
        manipulated_factor_fields=("preprocessing_strategy",),
        comparison_slice_fields=("subject", "filter_task"),
    ),
}


def get_preflight_stage_rule(experiment_id: str) -> PreflightStageRule:
    key = str(experiment_id).strip().upper()
    rule = _PREFLIGHT_RULES.get(key)
    if rule is None:
        allowed = ", ".join(sorted(_PREFLIGHT_RULES))
        raise ValueError(f"Unsupported preflight experiment_id '{experiment_id}'. Allowed: {allowed}")
    return rule


def preflight_experiment_ids() -> tuple[str, ...]:
    return tuple(sorted(_PREFLIGHT_RULES))


def evaluate_stage_lock_decision(
    *,
    rule: PreflightStageRule,
    completed_variants: int,
    failed_variants: int,
    blocked_variants: int,
    consistency_pass: bool,
    mean_margin_balanced_accuracy: float | None,
    baseline_delta_pass: bool,
    margin_vs_fold_std_pass: bool,
    dependency_reruns_required: bool,
) -> dict[str, Any]:
    reasons: list[str] = []

    expected_completed_pass = int(completed_variants) == int(rule.expected_completed_variants)
    if not expected_completed_pass:
        reasons.append(
            "expected_completed_variants_mismatch"
            f"(expected={rule.expected_completed_variants}, completed={completed_variants})"
        )

    failed_pass = int(failed_variants) == 0
    if not failed_pass:
        reasons.append(f"failed_variants_present(n={int(failed_variants)})")

    blocked_pass = int(blocked_variants) == 0
    if not blocked_pass:
        reasons.append(f"blocked_variants_present(n={int(blocked_variants)})")

    if not bool(consistency_pass):
        reasons.append("slice_winner_inconsistency")

    min_margin_pass = (
        mean_margin_balanced_accuracy is not None
        and float(mean_margin_balanced_accuracy) >= float(rule.min_margin_balanced_accuracy)
    )
    if not min_margin_pass:
        reasons.append(
            "margin_below_threshold"
            f"(min={rule.min_margin_balanced_accuracy:.4f}, observed={mean_margin_balanced_accuracy})"
        )

    if not bool(baseline_delta_pass):
        reasons.append(
            "baseline_delta_below_threshold"
            f"(min={rule.min_baseline_delta_balanced_accuracy:.4f})"
        )

    if not bool(margin_vs_fold_std_pass):
        reasons.append("margin_below_winner_fold_std")

    if bool(dependency_reruns_required):
        reasons.append("dependency_reruns_required")

    if not bool(rule.auto_lock_allowed):
        reasons.append("advisory_only_stage")

    auto_lock_passed = (
        bool(rule.auto_lock_allowed)
        and expected_completed_pass
        and failed_pass
        and blocked_pass
        and bool(consistency_pass)
        and min_margin_pass
        and bool(baseline_delta_pass)
        and bool(margin_vs_fold_std_pass)
        and not bool(dependency_reruns_required)
    )
    manual_review_required = not auto_lock_passed

    if auto_lock_passed:
        lock_status = "auto_lock_passed"
    elif not bool(rule.auto_lock_allowed):
        lock_status = "advisory_only"
    else:
        lock_status = "manual_review_required"

    return {
        "lock_status": lock_status,
        "auto_lock_passed": bool(auto_lock_passed),
        "manual_review_required": bool(manual_review_required),
        "expected_completed_pass": bool(expected_completed_pass),
        "failed_pass": bool(failed_pass),
        "blocked_pass": bool(blocked_pass),
        "consistency_pass": bool(consistency_pass),
        "min_margin_pass": bool(min_margin_pass),
        "baseline_delta_pass": bool(baseline_delta_pass),
        "margin_vs_fold_std_pass": bool(margin_vs_fold_std_pass),
        "dependency_reruns_required": bool(dependency_reruns_required),
        "auto_lock_allowed": bool(rule.auto_lock_allowed),
        "reasons": list(reasons),
    }


__all__ = [
    "DEFAULT_MIN_BASELINE_DELTA_BALANCED_ACCURACY",
    "DEFAULT_MIN_MARGIN_BALANCED_ACCURACY",
    "PreflightStageRule",
    "evaluate_stage_lock_decision",
    "get_preflight_stage_rule",
    "preflight_experiment_ids",
]
