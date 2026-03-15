from __future__ import annotations

from openpyxl.worksheet.worksheet import Worksheet

from Thesis_ML.workbook.structured_sheet_core import fill_simple_structured_sheet
from Thesis_ML.workbook.template_primitives import (
    add_dynamic_named_list,
    add_list_validation,
    col_idx,
)


def fill_study_design(
    ws: Worksheet,
    wb,
    *,
    study_design_columns: list[str],
) -> int:
    starter = [
        {
            "study_id": "S01",
            "study_name": "Example full factorial study",
            "enabled": "No",
            "study_type": "full_factorial",
            "intent": "exploratory",
            "question": "How do model and task filter settings change balanced accuracy?",
            "generalization_claim": "Within-dataset transfer across held-out sessions only.",
            "start_section": "dataset_selection",
            "end_section": "evaluation",
            "base_artifact_id": "",
            "primary_metric": "balanced_accuracy",
            "secondary_metrics": "macro_f1|accuracy",
            "cv_scheme": "within_subject_loso_session",
            "nested_cv": "No",
            "external_validation_planned": "No",
            "blocking_strategy": "none",
            "randomization_strategy": "fixed_order",
            "replication_mode": "fixed_repeats",
            "replication_strategy": "fixed_repeats",
            "num_repeats": "1",
            "random_seed_policy": "fixed",
            "stopping_rule": "Complete all planned design cells.",
            "notes": "",
        }
    ]
    last = fill_simple_structured_sheet(
        ws=ws,
        columns=study_design_columns,
        table_name="StudyDesignTable",
        title="Study Design (Factorial and Custom Matrices)",
        width_map={
            "A": 12,
            "B": 30,
            "C": 10,
            "D": 20,
            "E": 14,
            "F": 36,
            "G": 34,
            "H": 18,
            "I": 16,
            "J": 24,
            "K": 18,
            "L": 20,
            "M": 24,
            "N": 10,
            "O": 14,
            "P": 18,
            "Q": 18,
            "R": 18,
            "S": 18,
            "T": 14,
            "U": 18,
            "V": 24,
            "W": 34,
        },
        starter_rows=starter,
    )
    add_list_validation(ws, "=List_YesNo", col_idx(study_design_columns, "enabled"), 3, 1000)
    add_list_validation(
        ws,
        "=List_Study_Type",
        col_idx(study_design_columns, "study_type"),
        3,
        1000,
    )
    add_list_validation(ws, "=List_Study_Intent", col_idx(study_design_columns, "intent"), 3, 1000)
    add_list_validation(
        ws,
        "=List_Execution_Section",
        col_idx(study_design_columns, "start_section"),
        3,
        1000,
    )
    add_list_validation(
        ws,
        "=List_Execution_Section",
        col_idx(study_design_columns, "end_section"),
        3,
        1000,
    )
    add_list_validation(
        ws,
        "=List_Replication_Mode",
        col_idx(study_design_columns, "replication_mode"),
        3,
        1000,
    )
    add_list_validation(ws, "=List_YesNo", col_idx(study_design_columns, "nested_cv"), 3, 1000)
    add_list_validation(
        ws,
        "=List_YesNo",
        col_idx(study_design_columns, "external_validation_planned"),
        3,
        1000,
    )
    add_list_validation(
        ws,
        "=List_Blocking_Strategy",
        col_idx(study_design_columns, "blocking_strategy"),
        3,
        1000,
    )
    add_list_validation(
        ws,
        "=List_Randomization_Strategy",
        col_idx(study_design_columns, "randomization_strategy"),
        3,
        1000,
    )
    add_list_validation(
        ws,
        "=List_Replication_Strategy",
        col_idx(study_design_columns, "replication_strategy"),
        3,
        1000,
    )
    add_list_validation(
        ws,
        "=List_Random_Seed_Policy",
        col_idx(study_design_columns, "random_seed_policy"),
        3,
        1000,
    )
    add_dynamic_named_list(
        wb,
        "List_Study_ID",
        ws.title,
        col_idx(study_design_columns, "study_id"),
        3,
    )
    return last


def fill_factors(ws: Worksheet, *, factors_columns: list[str]) -> int:
    starter = [
        {
            "study_id": "S01",
            "factor_name": "model",
            "section_name": "",
            "parameter_path": "model",
            "factor_type": "categorical",
            "levels": "ridge|logreg",
        },
        {
            "study_id": "S01",
            "factor_name": "filter_task",
            "section_name": "",
            "parameter_path": "filter_task",
            "factor_type": "categorical",
            "levels": "passive|emo",
        },
    ]
    last = fill_simple_structured_sheet(
        ws=ws,
        columns=factors_columns,
        table_name="FactorsTable",
        title="Factor Definitions",
        width_map={
            "A": 12,
            "B": 20,
            "C": 18,
            "D": 26,
            "E": 14,
            "F": 34,
        },
        starter_rows=starter,
    )
    add_list_validation(ws, "=List_Study_ID", col_idx(factors_columns, "study_id"), 3, 1000)
    add_list_validation(ws, "=List_Factor_Type", col_idx(factors_columns, "factor_type"), 3, 1000)
    add_list_validation(
        ws,
        "=List_Execution_Section",
        col_idx(factors_columns, "section_name"),
        3,
        1000,
    )
    return last


def fill_study_rigor_checklist(
    ws: Worksheet,
    *,
    study_rigor_checklist_columns: list[str],
) -> int:
    last = fill_simple_structured_sheet(
        ws=ws,
        columns=study_rigor_checklist_columns,
        table_name="StudyRigorChecklistTable",
        title="Study Rigor Checklist",
        width_map={
            "A": 12,
            "B": 16,
            "C": 22,
            "D": 22,
            "E": 22,
            "F": 28,
            "G": 28,
            "H": 24,
            "I": 30,
            "J": 22,
            "K": 20,
            "L": 22,
            "M": 34,
        },
        starter_rows=None,
    )
    add_list_validation(
        ws,
        "=List_Study_ID",
        col_idx(study_rigor_checklist_columns, "study_id"),
        3,
        1000,
    )
    for column_name in (
        "leakage_risk_reviewed",
        "deployment_boundary_defined",
        "unit_of_analysis_defined",
        "data_hierarchy_defined",
        "reporting_checklist_completed",
        "risk_of_bias_reviewed",
        "confirmatory_lock_applied",
    ):
        add_list_validation(
            ws,
            "=List_YesNo",
            col_idx(study_rigor_checklist_columns, column_name),
            3,
            1000,
        )
    return last


def fill_analysis_plan(ws: Worksheet, *, analysis_plan_columns: list[str]) -> int:
    last = fill_simple_structured_sheet(
        ws=ws,
        columns=analysis_plan_columns,
        table_name="AnalysisPlanTable",
        title="Analysis Plan",
        width_map={
            "A": 12,
            "B": 28,
            "C": 28,
            "D": 18,
            "E": 18,
            "F": 24,
            "G": 24,
            "H": 32,
            "I": 34,
        },
        starter_rows=None,
    )
    add_list_validation(ws, "=List_Study_ID", col_idx(analysis_plan_columns, "study_id"), 3, 1000)
    add_list_validation(
        ws,
        "=List_Aggregation_Level",
        col_idx(analysis_plan_columns, "aggregation_level"),
        3,
        1000,
    )
    add_list_validation(
        ws,
        "=List_Uncertainty_Method",
        col_idx(analysis_plan_columns, "uncertainty_method"),
        3,
        1000,
    )
    add_list_validation(
        ws,
        "=List_Multiplicity_Handling",
        col_idx(analysis_plan_columns, "multiplicity_handling"),
        3,
        1000,
    )
    add_list_validation(
        ws,
        "=List_Interaction_Reporting_Policy",
        col_idx(analysis_plan_columns, "interaction_reporting_policy"),
        3,
        1000,
    )
    return last


def fill_study_review(ws: Worksheet, *, study_review_columns: list[str]) -> int:
    last = fill_simple_structured_sheet(
        ws=ws,
        columns=study_review_columns,
        table_name="StudyReviewTable",
        title="Study Review (Machine-managed pre-execution guardrails)",
        width_map={
            "A": 12,
            "B": 24,
            "C": 14,
            "D": 14,
            "E": 24,
            "F": 12,
            "G": 10,
            "H": 26,
            "I": 26,
            "J": 26,
            "K": 24,
            "L": 28,
            "M": 18,
            "N": 16,
            "O": 34,
            "P": 34,
            "Q": 30,
            "R": 12,
            "S": 12,
            "T": 12,
            "U": 20,
            "V": 24,
            "W": 20,
            "X": 10,
            "Y": 16,
            "Z": 18,
            "AA": 18,
            "AB": 18,
            "AC": 16,
            "AD": 12,
            "AE": 18,
            "AF": 20,
            "AG": 20,
            "AH": 12,
            "AI": 12,
            "AJ": 12,
            "AK": 12,
            "AL": 36,
        },
        starter_rows=None,
    )
    add_list_validation(ws, "=List_Study_ID", col_idx(study_review_columns, "study_id"), 3, 1000)
    add_list_validation(
        ws,
        "=List_Study_Review_Disposition",
        col_idx(study_review_columns, "execution_disposition"),
        3,
        1000,
    )
    add_list_validation(
        ws,
        "=List_Study_Eligibility_Status",
        col_idx(study_review_columns, "execution_eligibility_status"),
        3,
        1000,
    )
    return last


def fill_fixed_controls(ws: Worksheet, *, fixed_controls_columns: list[str]) -> int:
    last = fill_simple_structured_sheet(
        ws=ws,
        columns=fixed_controls_columns,
        table_name="DesignFixedControlsTable",
        title="Fixed Controls",
        width_map={
            "A": 12,
            "B": 28,
            "C": 28,
        },
        starter_rows=None,
    )
    add_list_validation(
        ws,
        "=List_Study_ID",
        col_idx(fixed_controls_columns, "study_id"),
        3,
        1000,
    )
    return last


def fill_constraints(ws: Worksheet, *, constraints_columns: list[str]) -> int:
    last = fill_simple_structured_sheet(
        ws=ws,
        columns=constraints_columns,
        table_name="DesignConstraintsTable",
        title="Invalid Combination Constraints",
        width_map={
            "A": 12,
            "B": 20,
            "C": 18,
            "D": 22,
            "E": 18,
            "F": 36,
        },
        starter_rows=None,
    )
    add_list_validation(ws, "=List_Study_ID", col_idx(constraints_columns, "study_id"), 3, 1000)
    return last


def fill_blocking_and_replication(
    ws: Worksheet,
    *,
    blocking_and_replication_columns: list[str],
) -> int:
    last = fill_simple_structured_sheet(
        ws=ws,
        columns=blocking_and_replication_columns,
        table_name="BlockingReplicationTable",
        title="Blocking and Replication",
        width_map={
            "A": 12,
            "B": 14,
            "C": 20,
            "D": 10,
            "E": 12,
        },
        starter_rows=None,
    )
    add_list_validation(
        ws,
        "=List_Block_Type",
        col_idx(blocking_and_replication_columns, "block_type"),
        3,
        1000,
    )
    add_list_validation(
        ws,
        "=List_Study_ID",
        col_idx(blocking_and_replication_columns, "study_id"),
        3,
        1000,
    )
    return last


def fill_generated_design_matrix(
    ws: Worksheet,
    *,
    generated_design_matrix_columns: list[str],
) -> int:
    last = fill_simple_structured_sheet(
        ws=ws,
        columns=generated_design_matrix_columns,
        table_name="GeneratedDesignMatrixTable",
        title="Generated Design Matrix (Machine-managed)",
        width_map={
            "A": 12,
            "B": 22,
            "C": 22,
            "D": 34,
            "E": 18,
            "F": 16,
            "G": 24,
            "H": 36,
            "I": 14,
        },
        starter_rows=None,
    )
    add_list_validation(
        ws,
        "=List_Study_ID",
        col_idx(generated_design_matrix_columns, "study_id"),
        3,
        1000,
    )
    add_list_validation(
        ws,
        "=List_Execution_Section",
        col_idx(generated_design_matrix_columns, "start_section"),
        3,
        1000,
    )
    add_list_validation(
        ws,
        "=List_Execution_Section",
        col_idx(generated_design_matrix_columns, "end_section"),
        3,
        1000,
    )
    add_list_validation(
        ws,
        "=List_Design_Cell_Status",
        col_idx(generated_design_matrix_columns, "status"),
        3,
        1000,
    )
    return last


def fill_effect_summaries(ws: Worksheet, *, effect_summaries_columns: list[str]) -> int:
    last = fill_simple_structured_sheet(
        ws=ws,
        columns=effect_summaries_columns,
        table_name="EffectSummariesTable",
        title="Effect Summaries (Descriptive)",
        width_map={
            "A": 12,
            "B": 22,
            "C": 22,
            "D": 18,
            "E": 24,
            "F": 20,
            "G": 20,
            "H": 20,
            "I": 18,
            "J": 34,
        },
        starter_rows=None,
    )
    add_list_validation(
        ws, "=List_Study_ID", col_idx(effect_summaries_columns, "study_id"), 3, 1000
    )
    return last

