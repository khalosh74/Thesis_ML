from __future__ import annotations

from collections.abc import Callable
from typing import Any

from openpyxl.workbook.workbook import Workbook
from pydantic import ValidationError


def build_study_design_layer(
    workbook: Workbook,
    *,
    _sheet_rows: Callable[..., Any],
    _STUDY_DESIGN_REQUIRED_COLUMNS: list[str],
    _normalize_text: Callable[[Any], str],
    _read_cell: Callable[[tuple[Any, ...], dict[str, int], str], Any],
    _parse_enabled: Callable[..., bool],
    StudyType: Any,
    StudyIntent: Any,
    _parse_optional_yes_no: Callable[..., bool | None],
    _validated_section: Callable[..., str],
    SectionName: Any,
    _FACTOR_COLUMNS: list[str],
    _parse_levels: Callable[..., list[Any]],
    FactorSpec: Any,
    _FIXED_CONTROL_COLUMNS: list[str],
    FixedControlSpec: Any,
    _parse_scalar_value: Callable[[Any], Any],
    _CONSTRAINT_COLUMNS: list[str],
    ConstraintSpec: Any,
    _BLOCKING_AND_REPLICATION_COLUMNS: list[str],
    BlockingReplicationSpec: Any,
    _GENERATED_DESIGN_MATRIX_COLUMNS: list[str],
    _parse_json_object: Callable[..., dict[str, Any]],
    _STUDY_RIGOR_CHECKLIST_COLUMNS: list[str],
    _CHECKLIST_YES_NO_COLUMNS: tuple[str, ...],
    _parse_yes_no: Callable[..., bool],
    StudyRigorChecklistSpec: Any,
    _ANALYSIS_PLAN_COLUMNS: list[str],
    AnalysisPlanSpec: Any,
    StudyDesignSpec: Any,
    _build_study_review: Callable[..., Any],
    _append_warning: Callable[[list[str], str], None],
    _enum_text: Callable[[Any], str],
    _fixed_controls_map: Callable[[Any], dict[str, Any]],
    _repeat_plan_for_study: Callable[[Any], list[dict[str, Any]]],
    _is_disallowed_combination: Callable[..., bool],
    _build_custom_matrix_cells: Callable[..., list[Any]],
    _build_factorial_cells: Callable[..., list[Any]],
    _trial_spec_from_cell: Callable[[Any, Any], dict[str, Any]],
    _default_stage_for_study: Callable[[Any], str],
) -> tuple[
    list[Any],
    list[Any],
    list[Any],
    list[Any],
    list[Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[str],
]:
    if "Study_Design" not in workbook.sheetnames:
        return [], [], [], [], [], [], [], []

    study_header, study_rows = _sheet_rows(
        workbook,
        sheet_name="Study_Design",
        required_columns=_STUDY_DESIGN_REQUIRED_COLUMNS,
    )
    raw_studies: dict[str, dict[str, Any]] = {}
    field_presence_by_study: dict[str, dict[str, bool]] = {}
    for row_index, row in study_rows:
        study_id = _normalize_text(_read_cell(row, study_header, "study_id"))
        if not study_id:
            continue
        if study_id in raw_studies:
            raise ValueError(f"Study_Design has duplicate study_id '{study_id}' (row {row_index}).")
        enabled = _parse_enabled(
            _read_cell(row, study_header, "enabled"),
            row_index=row_index,
            sheet_name="Study_Design",
        )
        study_type = (
            _normalize_text(_read_cell(row, study_header, "study_type"))
            or StudyType.SINGLE_EXPERIMENT.value
        )
        if enabled and study_type == StudyType.FRACTIONAL_FACTORIAL.value:
            raise ValueError(
                "Study_Design row "
                f"{row_index} uses unsupported study_type='fractional_factorial'. "
                "Use full_factorial or custom_matrix."
            )
        raw_primary_metric = _normalize_text(_read_cell(row, study_header, "primary_metric"))
        raw_cv_scheme = _normalize_text(_read_cell(row, study_header, "cv_scheme"))
        raw_question = _normalize_text(_read_cell(row, study_header, "question"))
        raw_generalization_claim = _normalize_text(
            _read_cell(row, study_header, "generalization_claim")
        )
        field_presence_by_study[study_id] = {
            "question": bool(raw_question),
            "generalization_claim": bool(raw_generalization_claim),
            "primary_metric": bool(raw_primary_metric),
            "cv_scheme": bool(raw_cv_scheme),
        }
        raw_studies[study_id] = {
            "study_id": study_id,
            "study_name": _normalize_text(_read_cell(row, study_header, "study_name")) or study_id,
            "enabled": enabled,
            "study_type": study_type,
            "intent": _normalize_text(_read_cell(row, study_header, "intent"))
            or StudyIntent.EXPLORATORY.value,
            "question": raw_question or None,
            "generalization_claim": raw_generalization_claim or None,
            "start_section": _validated_section(
                _read_cell(row, study_header, "start_section"),
                field_name="start_section",
                row_index=row_index,
                sheet_name="Study_Design",
            )
            or SectionName.DATASET_SELECTION.value,
            "end_section": _validated_section(
                _read_cell(row, study_header, "end_section"),
                field_name="end_section",
                row_index=row_index,
                sheet_name="Study_Design",
            )
            or SectionName.EVALUATION.value,
            "base_artifact_id": _normalize_text(_read_cell(row, study_header, "base_artifact_id"))
            or None,
            "primary_metric": raw_primary_metric,
            "secondary_metrics": _normalize_text(_read_cell(row, study_header, "secondary_metrics"))
            or None,
            "cv_scheme": raw_cv_scheme or None,
            "nested_cv": _parse_optional_yes_no(
                _read_cell(row, study_header, "nested_cv"),
                row_index=row_index,
                sheet_name="Study_Design",
                field_name="nested_cv",
            ),
            "external_validation_planned": _parse_optional_yes_no(
                _read_cell(row, study_header, "external_validation_planned"),
                row_index=row_index,
                sheet_name="Study_Design",
                field_name="external_validation_planned",
            ),
            "blocking_strategy": _normalize_text(_read_cell(row, study_header, "blocking_strategy"))
            or None,
            "randomization_strategy": _normalize_text(
                _read_cell(row, study_header, "randomization_strategy")
            )
            or None,
            "replication_mode": _normalize_text(_read_cell(row, study_header, "replication_mode"))
            or "none",
            "replication_strategy": _normalize_text(
                _read_cell(row, study_header, "replication_strategy")
            )
            or None,
            "num_repeats": int(_read_cell(row, study_header, "num_repeats") or 1),
            "random_seed_policy": _normalize_text(
                _read_cell(row, study_header, "random_seed_policy")
            )
            or "fixed",
            "stopping_rule": _normalize_text(_read_cell(row, study_header, "stopping_rule"))
            or None,
            "notes": _normalize_text(_read_cell(row, study_header, "notes")) or None,
            "factors": [],
            "fixed_controls": [],
            "constraints": [],
            "blocking_replication": [],
        }

    factor_header, factor_rows = _sheet_rows(
        workbook,
        sheet_name="Factors",
        required_columns=_FACTOR_COLUMNS,
    )
    for row_index, row in factor_rows:
        study_id = _normalize_text(_read_cell(row, factor_header, "study_id"))
        if not study_id:
            continue
        if study_id not in raw_studies:
            raise ValueError(f"Factors row {row_index} references unknown study_id '{study_id}'.")
        factor_name = _normalize_text(_read_cell(row, factor_header, "factor_name"))
        raw_studies[study_id]["factors"].append(
            FactorSpec.model_validate(
                {
                    "study_id": study_id,
                    "factor_name": factor_name,
                    "section_name": (
                        _validated_section(
                            _read_cell(row, factor_header, "section_name"),
                            field_name="section_name",
                            row_index=row_index,
                            sheet_name="Factors",
                        )
                        or None
                    ),
                    "parameter_path": _normalize_text(
                        _read_cell(row, factor_header, "parameter_path")
                    ),
                    "factor_type": _normalize_text(_read_cell(row, factor_header, "factor_type"))
                    or "categorical",
                    "levels": _parse_levels(
                        _read_cell(row, factor_header, "levels"),
                        row_index=row_index,
                        factor_name=factor_name,
                        study_id=study_id,
                    ),
                }
            )
        )

    control_header, control_rows = _sheet_rows(
        workbook,
        sheet_name="Fixed_Controls",
        required_columns=_FIXED_CONTROL_COLUMNS,
    )
    for row_index, row in control_rows:
        study_id = _normalize_text(_read_cell(row, control_header, "study_id"))
        if not study_id:
            continue
        if study_id not in raw_studies:
            raise ValueError(
                f"Fixed_Controls row {row_index} references unknown study_id '{study_id}'."
            )
        raw_studies[study_id]["fixed_controls"].append(
            FixedControlSpec.model_validate(
                {
                    "study_id": study_id,
                    "parameter_path": _normalize_text(
                        _read_cell(row, control_header, "parameter_path")
                    ),
                    "value": _parse_scalar_value(_read_cell(row, control_header, "value")),
                }
            )
        )

    constraint_header, constraint_rows = _sheet_rows(
        workbook,
        sheet_name="Constraints",
        required_columns=_CONSTRAINT_COLUMNS,
    )
    for row_index, row in constraint_rows:
        study_id = _normalize_text(_read_cell(row, constraint_header, "study_id"))
        if not study_id:
            continue
        if study_id not in raw_studies:
            raise ValueError(
                f"Constraints row {row_index} references unknown study_id '{study_id}'."
            )
        raw_studies[study_id]["constraints"].append(
            ConstraintSpec.model_validate(
                {
                    "study_id": study_id,
                    "if_factor": _normalize_text(_read_cell(row, constraint_header, "if_factor")),
                    "if_level": _parse_scalar_value(_read_cell(row, constraint_header, "if_level")),
                    "disallow_factor": _normalize_text(
                        _read_cell(row, constraint_header, "disallow_factor")
                    ),
                    "disallow_level": _parse_scalar_value(
                        _read_cell(row, constraint_header, "disallow_level")
                    ),
                    "reason": _normalize_text(_read_cell(row, constraint_header, "reason")) or None,
                }
            )
        )

    blocking_header, blocking_rows = _sheet_rows(
        workbook,
        sheet_name="Blocking_and_Replication",
        required_columns=_BLOCKING_AND_REPLICATION_COLUMNS,
    )
    for row_index, row in blocking_rows:
        study_id = _normalize_text(_read_cell(row, blocking_header, "study_id"))
        if not study_id:
            continue
        if study_id not in raw_studies:
            raise ValueError(
                "Blocking_and_Replication row "
                f"{row_index} references unknown study_id '{study_id}'."
            )
        raw_studies[study_id]["blocking_replication"].append(
            BlockingReplicationSpec.model_validate(
                {
                    "study_id": study_id,
                    "block_type": _normalize_text(_read_cell(row, blocking_header, "block_type"))
                    or "none",
                    "block_value": _normalize_text(_read_cell(row, blocking_header, "block_value"))
                    or None,
                    "repeat_id": int(_read_cell(row, blocking_header, "repeat_id") or 1),
                    "seed": (
                        int(_read_cell(row, blocking_header, "seed"))
                        if _read_cell(row, blocking_header, "seed") not in (None, "")
                        else None
                    ),
                }
            )
        )

    generated_header, generated_rows = _sheet_rows(
        workbook,
        sheet_name="Generated_Design_Matrix",
        required_columns=_GENERATED_DESIGN_MATRIX_COLUMNS,
    )
    generated_by_study: dict[str, list[dict[str, Any]]] = {}
    for row_index, row in generated_rows:
        study_id = _normalize_text(_read_cell(row, generated_header, "study_id"))
        if not study_id:
            continue
        generated_by_study.setdefault(study_id, []).append(
            {
                "row_index": row_index,
                "trial_id": _read_cell(row, generated_header, "trial_id"),
                "cell_id": _read_cell(row, generated_header, "cell_id"),
                "factor_settings": _parse_json_object(
                    _read_cell(row, generated_header, "factor_settings_json"),
                    row_index=row_index,
                    sheet_name="Generated_Design_Matrix",
                    column_name="factor_settings_json",
                ),
                "start_section": _read_cell(row, generated_header, "start_section"),
                "end_section": _read_cell(row, generated_header, "end_section"),
                "base_artifact_id": _read_cell(row, generated_header, "base_artifact_id"),
                "resolved_params": _parse_json_object(
                    _read_cell(row, generated_header, "resolved_params_json"),
                    row_index=row_index,
                    sheet_name="Generated_Design_Matrix",
                    column_name="resolved_params_json",
                ),
                "status": _read_cell(row, generated_header, "status"),
            }
        )

    checklist_header, checklist_rows = _sheet_rows(
        workbook,
        sheet_name="Study_Rigor_Checklist",
        required_columns=_STUDY_RIGOR_CHECKLIST_COLUMNS,
    )
    checklist_by_study: dict[str, StudyRigorChecklistSpec] = {}
    for row_index, row in checklist_rows:
        study_id = _normalize_text(_read_cell(row, checklist_header, "study_id"))
        if not study_id:
            continue
        if study_id not in raw_studies:
            raise ValueError(
                f"Study_Rigor_Checklist row {row_index} references unknown study_id '{study_id}'."
            )
        if study_id in checklist_by_study:
            raise ValueError(
                "Study_Rigor_Checklist has duplicate entries for "
                f"study_id '{study_id}' (row {row_index})."
            )

        required_text_fields = {
            "missing_data_plan": _normalize_text(
                _read_cell(row, checklist_header, "missing_data_plan")
            ),
            "class_imbalance_plan": _normalize_text(
                _read_cell(row, checklist_header, "class_imbalance_plan")
            ),
            "subgroup_plan": _normalize_text(_read_cell(row, checklist_header, "subgroup_plan")),
        }
        missing_required = [name for name, value in required_text_fields.items() if not value]
        if missing_required:
            raise ValueError(
                "Study_Rigor_Checklist row "
                f"{row_index} is missing required values: {', '.join(missing_required)}"
            )

        payload: dict[str, Any] = {
            "study_id": study_id,
            "missing_data_plan": required_text_fields["missing_data_plan"],
            "class_imbalance_plan": required_text_fields["class_imbalance_plan"],
            "subgroup_plan": required_text_fields["subgroup_plan"],
            "fairness_or_applicability_notes": _normalize_text(
                _read_cell(row, checklist_header, "fairness_or_applicability_notes")
            )
            or None,
            "analysis_notes": _normalize_text(_read_cell(row, checklist_header, "analysis_notes"))
            or None,
        }
        for column_name in _CHECKLIST_YES_NO_COLUMNS:
            payload[column_name] = _parse_yes_no(
                _read_cell(row, checklist_header, column_name),
                row_index=row_index,
                sheet_name="Study_Rigor_Checklist",
                field_name=column_name,
            )
        try:
            checklist_by_study[study_id] = StudyRigorChecklistSpec.model_validate(payload)
        except ValidationError as exc:
            raise ValueError(
                f"Invalid Study_Rigor_Checklist entry for study_id '{study_id}' "
                f"(row {row_index}): {exc}"
            ) from exc

    analysis_header, analysis_rows = _sheet_rows(
        workbook,
        sheet_name="Analysis_Plan",
        required_columns=_ANALYSIS_PLAN_COLUMNS,
    )
    analysis_by_study: dict[str, AnalysisPlanSpec] = {}
    for row_index, row in analysis_rows:
        study_id = _normalize_text(_read_cell(row, analysis_header, "study_id"))
        if not study_id:
            continue
        if study_id not in raw_studies:
            raise ValueError(
                f"Analysis_Plan row {row_index} references unknown study_id '{study_id}'."
            )
        if study_id in analysis_by_study:
            raise ValueError(
                f"Analysis_Plan has duplicate entries for study_id '{study_id}' (row {row_index})."
            )

        aggregation_level = _normalize_text(_read_cell(row, analysis_header, "aggregation_level"))
        uncertainty_method = _normalize_text(_read_cell(row, analysis_header, "uncertainty_method"))
        missing_required = [
            field_name
            for field_name, value in (
                ("aggregation_level", aggregation_level),
                ("uncertainty_method", uncertainty_method),
            )
            if not value
        ]
        if missing_required:
            raise ValueError(
                f"Analysis_Plan row {row_index} is missing required values: "
                + ", ".join(missing_required)
            )

        try:
            analysis_by_study[study_id] = AnalysisPlanSpec.model_validate(
                {
                    "study_id": study_id,
                    "primary_contrast": _normalize_text(
                        _read_cell(row, analysis_header, "primary_contrast")
                    )
                    or None,
                    "secondary_contrasts": _normalize_text(
                        _read_cell(row, analysis_header, "secondary_contrasts")
                    )
                    or None,
                    "aggregation_level": aggregation_level,
                    "uncertainty_method": uncertainty_method,
                    "multiplicity_handling": _normalize_text(
                        _read_cell(row, analysis_header, "multiplicity_handling")
                    )
                    or None,
                    "interaction_reporting_policy": _normalize_text(
                        _read_cell(row, analysis_header, "interaction_reporting_policy")
                    )
                    or None,
                    "interpretation_rules": _normalize_text(
                        _read_cell(row, analysis_header, "interpretation_rules")
                    )
                    or None,
                    "notes": _normalize_text(_read_cell(row, analysis_header, "notes")) or None,
                }
            )
        except ValidationError as exc:
            raise ValueError(
                f"Invalid analysis plan '{study_id}' in Analysis_Plan row {row_index}: {exc}"
            ) from exc

    study_specs = [
        StudyDesignSpec.model_validate(raw_studies[study_id]) for study_id in sorted(raw_studies)
    ]
    checklist_specs = [checklist_by_study[study_id] for study_id in sorted(checklist_by_study)]
    analysis_plan_specs = [analysis_by_study[study_id] for study_id in sorted(analysis_by_study)]
    study_review_specs: list[Any] = []
    generated_cells: list[Any] = []
    study_trials: list[dict[str, Any]] = []
    study_experiments: list[dict[str, Any]] = []
    validation_warnings: list[str] = []

    for study in study_specs:
        if not study.enabled:
            continue
        review = _build_study_review(
            study=study,
            checklist=checklist_by_study.get(study.study_id),
            analysis_plan=analysis_by_study.get(study.study_id),
            field_presence=field_presence_by_study.get(study.study_id),
            raw_custom_rows=generated_by_study.get(study.study_id, []),
            enum_text_fn=_enum_text,
            fixed_controls_fn=_fixed_controls_map,
            repeat_plan_fn=_repeat_plan_for_study,
            disallowed_combination_fn=_is_disallowed_combination,
        )

        for warning_text in review.warnings:
            _append_warning(validation_warnings, f"Study '{study.study_id}': {warning_text}")

        blocked_reasons = list(review.errors)
        if review.execution_disposition == "blocked":
            study_review_specs.append(review)
            study_experiments.append(
                {
                    "experiment_id": study.study_id,
                    "title": study.study_name,
                    "stage": _default_stage_for_study(study),
                    "manipulated_factor": ", ".join(factor.factor_name for factor in study.factors)
                    or None,
                    "primary_metric": study.primary_metric,
                    "is_study_design": True,
                    "executable_now": False,
                    "blocked_reasons": blocked_reasons,
                    "notes": (
                        "Study blocked by scientific-rigor guardrails before execution."
                        + (f" Question: {study.question}" if study.question else "")
                        + (f" Notes: {study.notes}" if study.notes else "")
                    ),
                    "variant_templates": [],
                }
            )
            continue

        if study.study_type == StudyType.CUSTOM_MATRIX:
            cells = _build_custom_matrix_cells(
                study,
                raw_rows=generated_by_study.get(study.study_id, []),
            )
        else:
            cells = _build_factorial_cells(study)

        study_variant_templates: list[dict[str, Any]] = []
        for cell in cells:
            generated_cells.append(cell)
            trial_spec = _trial_spec_from_cell(study, cell)
            study_trials.append(trial_spec)
            study_variant_templates.append(trial_spec)

        if not study_variant_templates:
            no_cell_reason = (
                "No executable design cells were produced after applying factors, "
                "constraints, and replication rules."
            )
            blocked_reasons_with_cells = blocked_reasons + [no_cell_reason]
            review = review.model_copy(
                update={
                    "execution_disposition": "blocked",
                    "execution_eligibility_status": "blocked",
                    "error_count": int(review.error_count) + 1,
                    "errors": list(review.errors) + [no_cell_reason],
                    "missing_fields": list(review.missing_fields),
                }
            )
            study_review_specs.append(review)
            study_experiments.append(
                {
                    "experiment_id": study.study_id,
                    "title": study.study_name,
                    "stage": _default_stage_for_study(study),
                    "manipulated_factor": ", ".join(factor.factor_name for factor in study.factors)
                    or None,
                    "primary_metric": study.primary_metric,
                    "is_study_design": True,
                    "executable_now": False,
                    "blocked_reasons": blocked_reasons_with_cells,
                    "notes": (
                        "Study blocked after design expansion produced zero executable cells."
                        + (f" Question: {study.question}" if study.question else "")
                        + (f" Notes: {study.notes}" if study.notes else "")
                    ),
                    "variant_templates": [],
                }
            )
            continue

        study_review_specs.append(review)
        study_experiments.append(
            {
                "experiment_id": study.study_id,
                "title": study.study_name,
                "stage": _default_stage_for_study(study),
                "manipulated_factor": ", ".join(factor.factor_name for factor in study.factors)
                or None,
                "primary_metric": study.primary_metric,
                "is_study_design": True,
                "executable_now": True,
                "blocked_reasons": [],
                "notes": (
                    "Factorial study design compiled from workbook."
                    + (
                        " Guardrail disposition=warning."
                        if review.execution_disposition == "warning"
                        else ""
                    )
                    + (f" Question: {study.question}" if study.question else "")
                    + (f" Notes: {study.notes}" if study.notes else "")
                ),
                "variant_templates": study_variant_templates,
            }
        )

    return (
        study_specs,
        checklist_specs,
        analysis_plan_specs,
        study_review_specs,
        generated_cells,
        study_trials,
        study_experiments,
        validation_warnings,
    )

