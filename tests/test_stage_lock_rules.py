from __future__ import annotations

from Thesis_ML.orchestration.stage_lock_rules import (
    evaluate_stage_lock_decision,
    get_preflight_stage_rule,
)


def test_incomplete_stage_requires_manual_review() -> None:
    rule = get_preflight_stage_rule("E04")
    decision = evaluate_stage_lock_decision(
        rule=rule,
        completed_variants=7,
        failed_variants=0,
        blocked_variants=0,
        consistency_pass=True,
        mean_margin_balanced_accuracy=0.05,
        baseline_delta_pass=True,
        margin_vs_fold_std_pass=True,
        dependency_reruns_required=False,
    )
    assert decision["manual_review_required"] is True
    assert decision["auto_lock_passed"] is False


def test_failed_or_blocked_stage_requires_manual_review() -> None:
    rule = get_preflight_stage_rule("E04")
    decision = evaluate_stage_lock_decision(
        rule=rule,
        completed_variants=8,
        failed_variants=1,
        blocked_variants=1,
        consistency_pass=True,
        mean_margin_balanced_accuracy=0.05,
        baseline_delta_pass=True,
        margin_vs_fold_std_pass=True,
        dependency_reruns_required=False,
    )
    assert decision["manual_review_required"] is True
    assert decision["auto_lock_passed"] is False


def test_inconsistent_slice_winners_require_manual_review() -> None:
    rule = get_preflight_stage_rule("E06")
    decision = evaluate_stage_lock_decision(
        rule=rule,
        completed_variants=12,
        failed_variants=0,
        blocked_variants=0,
        consistency_pass=False,
        mean_margin_balanced_accuracy=0.05,
        baseline_delta_pass=True,
        margin_vs_fold_std_pass=True,
        dependency_reruns_required=False,
    )
    assert decision["manual_review_required"] is True
    assert decision["auto_lock_passed"] is False


def test_margin_below_threshold_requires_manual_review() -> None:
    rule = get_preflight_stage_rule("E06")
    decision = evaluate_stage_lock_decision(
        rule=rule,
        completed_variants=12,
        failed_variants=0,
        blocked_variants=0,
        consistency_pass=True,
        mean_margin_balanced_accuracy=0.01,
        baseline_delta_pass=True,
        margin_vs_fold_std_pass=True,
        dependency_reruns_required=False,
    )
    assert decision["manual_review_required"] is True
    assert decision["auto_lock_passed"] is False


def test_strong_consistent_winner_auto_locks() -> None:
    rule = get_preflight_stage_rule("E06")
    decision = evaluate_stage_lock_decision(
        rule=rule,
        completed_variants=12,
        failed_variants=0,
        blocked_variants=0,
        consistency_pass=True,
        mean_margin_balanced_accuracy=0.05,
        baseline_delta_pass=True,
        margin_vs_fold_std_pass=True,
        dependency_reruns_required=False,
    )
    assert decision["manual_review_required"] is False
    assert decision["auto_lock_passed"] is True
    assert decision["lock_status"] == "auto_lock_passed"
