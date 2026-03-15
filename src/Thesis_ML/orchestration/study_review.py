from __future__ import annotations

import itertools
from collections.abc import Callable
from typing import Any

from Thesis_ML.orchestration.contracts import (
    AnalysisPlanSpec,
    ConstraintSpec,
    StudyDesignSpec,
    StudyIntent,
    StudyReviewSummary,
    StudyRigorChecklistSpec,
    StudyType,
)


def append_warning(warnings: list[str], message: str) -> None:
    text = str(message).strip()
    if text:
        warnings.append(text)


def _constraint_descriptions(study: StudyDesignSpec) -> list[str]:
    descriptions: list[str] = []
    for constraint in study.constraints:
        reason = f" ({constraint.reason})" if constraint.reason else ""
        descriptions.append(
            "if "
            f"{constraint.if_factor}={constraint.if_level} then disallow "
            f"{constraint.disallow_factor}={constraint.disallow_level}{reason}"
        )
    return descriptions


def _expected_design_counts(
    *,
    study: StudyDesignSpec,
    raw_custom_rows: list[dict[str, Any]],
    repeat_plan_fn: Callable[[StudyDesignSpec], list[dict[str, Any]]],
    disallowed_combination_fn: Callable[[dict[str, Any], list[ConstraintSpec]], bool],
) -> tuple[int, int, int]:
    if study.study_type == StudyType.CUSTOM_MATRIX:
        expected_cells = len(raw_custom_rows)
        return expected_cells, expected_cells, 0

    repeats = len(repeat_plan_fn(study))
    factor_names = [factor.factor_name for factor in study.factors]
    level_space = [factor.levels for factor in study.factors]
    all_combinations = list(itertools.product(*level_space)) if level_space else [tuple()]
    total_cells = len(all_combinations)
    valid_cells = 0
    for combo in all_combinations:
        factor_settings = {name: combo[idx] for idx, name in enumerate(factor_names)}
        if disallowed_combination_fn(factor_settings, study.constraints):
            continue
        valid_cells += 1
    excluded = max(0, total_cells - valid_cells)
    return valid_cells, valid_cells * repeats, excluded


def _checklist_status(checklist: StudyRigorChecklistSpec | None) -> str:
    if checklist is None:
        return "missing"
    missing_core = (
        not bool(checklist.leakage_risk_reviewed)
        or not bool(checklist.unit_of_analysis_defined)
        or not bool(checklist.data_hierarchy_defined)
    )
    return "partial" if missing_core else "complete"


def _analysis_status(analysis_plan: AnalysisPlanSpec | None) -> str:
    if analysis_plan is None:
        return "missing"
    missing_core = not bool(str(analysis_plan.primary_contrast or "").strip()) or not bool(
        str(analysis_plan.interpretation_rules or "").strip()
    )
    return "partial" if missing_core else "complete"


def build_study_review(
    *,
    study: StudyDesignSpec,
    checklist: StudyRigorChecklistSpec | None,
    analysis_plan: AnalysisPlanSpec | None,
    field_presence: dict[str, bool] | None,
    raw_custom_rows: list[dict[str, Any]],
    enum_text_fn: Callable[[Any], str],
    fixed_controls_fn: Callable[[StudyDesignSpec], dict[str, Any]],
    repeat_plan_fn: Callable[[StudyDesignSpec], list[dict[str, Any]]],
    disallowed_combination_fn: Callable[[dict[str, Any], list[ConstraintSpec]], bool],
) -> StudyReviewSummary:
    # Guardrail policy:
    # - Core fields (question/generalization_claim/primary_metric/cv_scheme) are hard requirements
    #   for all enabled studies.
    # - Confirmatory studies treat rigor-plan gaps as hard errors.
    # - Exploratory studies downgrade non-core rigor gaps to warnings.
    strict = study.intent == StudyIntent.CONFIRMATORY
    core_presence = {
        "question": bool(str(study.question or "").strip()),
        "generalization_claim": bool(str(study.generalization_claim or "").strip()),
        "primary_metric": bool(str(study.primary_metric or "").strip()),
        "cv_scheme": bool(str(study.cv_scheme or "").strip()),
    }
    if field_presence:
        for key in ("question", "generalization_claim", "primary_metric", "cv_scheme"):
            if key in field_presence:
                core_presence[key] = bool(field_presence[key])

    errors: list[str] = []
    warnings: list[str] = []
    missing_fields: list[str] = []

    for key, is_present in core_presence.items():
        if is_present:
            continue
        missing_fields.append(key)
        errors.append(f"Missing core field '{key}'.")

    checklist_missing: list[str] = []
    if checklist is None:
        checklist_missing.extend(
            [
                "leakage_risk_reviewed",
                "unit_of_analysis_defined",
                "data_hierarchy_defined",
                "confirmatory_lock_applied",
            ]
        )
    else:
        if not checklist.leakage_risk_reviewed:
            checklist_missing.append("leakage_risk_reviewed")
        if not checklist.unit_of_analysis_defined:
            checklist_missing.append("unit_of_analysis_defined")
        if not checklist.data_hierarchy_defined:
            checklist_missing.append("data_hierarchy_defined")
        if not checklist.confirmatory_lock_applied:
            checklist_missing.append("confirmatory_lock_applied")

    analysis_missing: list[str] = []
    if analysis_plan is None:
        analysis_missing.extend(["primary_contrast", "interpretation_rules"])
    else:
        if not str(analysis_plan.primary_contrast or "").strip():
            analysis_missing.append("primary_contrast")
        if not str(analysis_plan.interpretation_rules or "").strip():
            analysis_missing.append("interpretation_rules")
        if strict and not str(analysis_plan.multiplicity_handling or "").strip():
            analysis_missing.append("multiplicity_handling")

    for field_name in checklist_missing:
        if field_name not in missing_fields:
            missing_fields.append(field_name)
        if strict:
            errors.append(f"Missing required rigor checklist field '{field_name}'.")
        else:
            warnings.append(f"Incomplete rigor checklist field '{field_name}'.")

    for field_name in analysis_missing:
        if field_name not in missing_fields:
            missing_fields.append(field_name)
        if strict:
            errors.append(f"Missing required analysis plan field '{field_name}'.")
        else:
            warnings.append(f"Incomplete analysis plan field '{field_name}'.")

    expected_cells, expected_trials, excluded = _expected_design_counts(
        study=study,
        raw_custom_rows=raw_custom_rows,
        repeat_plan_fn=repeat_plan_fn,
        disallowed_combination_fn=disallowed_combination_fn,
    )
    disposition: str
    eligibility_status: str
    if errors:
        disposition = "blocked"
        eligibility_status = "blocked"
    elif warnings:
        disposition = "warning"
        eligibility_status = "eligible_with_warnings"
    else:
        disposition = "allowed"
        eligibility_status = "eligible"

    factor_map = {factor.factor_name: list(factor.levels) for factor in study.factors}
    fixed_controls = fixed_controls_fn(study)
    return StudyReviewSummary.model_validate(
        {
            "study_id": study.study_id,
            "study_name": study.study_name,
            "intent": enum_text_fn(study.intent),
            "question": study.question,
            "generalization_claim": study.generalization_claim,
            "start_section": enum_text_fn(study.start_section),
            "end_section": enum_text_fn(study.end_section),
            "factors": factor_map,
            "fixed_controls": fixed_controls,
            "blocked_constraints": _constraint_descriptions(study),
            "excluded_combination_count": excluded,
            "expected_design_cells": expected_cells,
            "expected_trials": expected_trials,
            "primary_metric": study.primary_metric,
            "secondary_metrics": study.secondary_metrics,
            "cv_scheme": study.cv_scheme,
            "nested_cv": study.nested_cv,
            "external_validation_planned": study.external_validation_planned,
            "blocking_strategy": study.blocking_strategy,
            "randomization_strategy": study.randomization_strategy,
            "replication_strategy": study.replication_strategy,
            "replication_mode": study.replication_mode,
            "num_repeats": int(study.num_repeats),
            "random_seed_policy": study.random_seed_policy,
            "rigor_checklist_status": _checklist_status(checklist),
            "analysis_plan_status": _analysis_status(analysis_plan),
            "execution_eligibility_status": eligibility_status,
            "execution_disposition": disposition,
            "warning_count": len(warnings),
            "error_count": len(errors),
            "missing_fields": missing_fields,
            "warnings": warnings,
            "errors": errors,
        }
    )
