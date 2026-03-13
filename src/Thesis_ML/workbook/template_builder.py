from __future__ import annotations

from pathlib import Path

from openpyxl import Workbook, load_workbook
from openpyxl.formatting.rule import FormulaRule
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.workbook.defined_name import DefinedName
from openpyxl.worksheet.datavalidation import DataValidation
from openpyxl.worksheet.table import Table, TableStyleInfo

from Thesis_ML.config.paths import DEFAULT_WORKBOOK_TEMPLATE
from Thesis_ML.config.schema_versions import (
    SUPPORTED_WORKBOOK_SCHEMA_VERSIONS,
    WORKBOOK_SCHEMA_METADATA_START_ROW,
    WORKBOOK_SCHEMA_VERSION,
)
from Thesis_ML.workbook.schema_metadata import (
    expected_schema_metadata,
    read_schema_metadata,
    write_schema_metadata,
)

OUT_XLSX = Path(DEFAULT_WORKBOOK_TEMPLATE)

SHEET_ORDER = [
    "README",
    "Master_Experiments",
    "Experiment_Definitions",
    "Search_Spaces",
    "Artifact_Registry",
    "Fixed_Configs",
    "Objectives",
    "Machine_Status",
    "Trial_Results",
    "Summary_Outputs",
    "Data_Selection_Design",
    "Grouping_Strategy_Map",
    "Data_Profile",
    "Run_Log",
    "Decision_Log",
    "Confirmatory_Set",
    "Thesis_Map",
    "Dictionary_Validation",
    "Dashboard",
    "Claim_Ledger",
    "AI_Usage_Log",
    "Ethics_Governance_Notes",
]

STAGE_V2 = [
    "Stage 1 - Target lock",
    "Stage 2 - Split lock",
    "Stage 3 - Model lock",
    "Stage 4 - Feature/preprocessing lock",
    "Stage 5 - Confirmatory analysis",
    "Stage 6 - Robustness analysis",
    "Stage 7 - Exploratory extension",
]

MASTER_COLUMNS = [
    "Experiment_ID",
    "Short_Title",
    "Category",
    "Evidential_Role",
    "Stage",
    "Priority",
    "Decision_Supported",
    "Exact_Question",
    "Hypothesis_or_Expectation",
    "Manipulated_Factor",
    "Held_Constant_Controls",
    "Dataset_Scope",
    "Target_Definition",
    "Split_Logic",
    "Leakage_Risk",
    "Leakage_Control",
    "Models_Compared",
    "Primary_Metric",
    "Secondary_Metrics",
    "Robustness_Checks",
    "Decision_Criterion",
    "Success_Pattern",
    "Failure_or_Warning_Pattern",
    "Threats_to_Validity",
    "Reporting_Destination",
    "Interpretation_Boundary",
    "Status",
    "Owner",
    "Outcome_Summary",
    "Decision_Taken",
    "Notes",
    "Experiment_Ready",
]

RUN_LOG_COLUMNS = [
    "Run_ID",
    "Experiment_ID",
    "Run_Date",
    "Dataset_Name",
    "Data_Subset",
    "Data_Slice_ID",
    "Grouping_Strategy_ID",
    "Code_Commit_or_Version",
    "Config_File_or_Path",
    "Random_Seed",
    "Target",
    "Split_ID_or_Fold_Definition",
    "Train_Group_Rule",
    "Test_Group_Rule",
    "Transfer_Direction",
    "Session_Coverage",
    "Task_Coverage",
    "Modality_Coverage",
    "Model",
    "Feature_Set",
    "Run_Type",
    "Affects_Frozen_Pipeline",
    "Eligible_for_Method_Decision",
    "Sample_Count",
    "Class_Counts",
    "Imbalance_Status",
    "Leakage_Check_Status",
    "Primary_Metric_Value",
    "Secondary_Metric_1",
    "Secondary_Metric_2",
    "Robustness_Output_Summary",
    "Result_Summary",
    "Preliminary_Interpretation",
    "Reviewed",
    "Used_in_Thesis",
    "Artifact_Path",
    "Notes",
]

EXPERIMENT_DEFINITIONS_COLUMNS = [
    "experiment_id",
    "enabled",
    "start_section",
    "end_section",
    "base_artifact_id",
    "target",
    "cv",
    "model",
    "subject",
    "train_subject",
    "test_subject",
    "filter_task",
    "filter_modality",
    "reuse_policy",
    "search_space_id",
]

SEARCH_SPACES_COLUMNS = [
    "search_space_id",
    "enabled",
    "optimization_mode",
    "parameter_name",
    "parameter_values",
    "parameter_scope",
    "objective_metric",
    "max_trials",
    "notes",
]

ARTIFACT_REGISTRY_COLUMNS = [
    "artifact_id",
    "artifact_type",
    "run_id",
    "status",
    "created_at",
    "path",
    "upstream_artifact_ids",
    "config_hash",
    "code_ref",
    "notes",
]

FIXED_CONFIGS_COLUMNS = [
    "config_key",
    "config_value",
    "scope",
    "locked",
    "owner",
    "last_updated",
    "notes",
]

OBJECTIVES_COLUMNS = [
    "objective_id",
    "objective_text",
    "stage",
    "linked_experiment_id",
    "primary_metric",
    "success_criterion",
    "status",
    "notes",
]

MACHINE_STATUS_COLUMNS = [
    "machine_id",
    "hostname",
    "environment_name",
    "python_version",
    "gpu",
    "status",
    "last_checked",
    "notes",
]

TRIAL_RESULTS_COLUMNS = [
    "trial_id",
    "experiment_id",
    "run_id",
    "status",
    "primary_metric_name",
    "primary_metric_value",
    "report_path",
    "metrics_path",
    "artifact_bundle",
    "notes",
]

SUMMARY_OUTPUTS_COLUMNS = [
    "summary_type",
    "summary_key",
    "primary_metric_name",
    "primary_metric_value",
    "run_id",
    "experiment_id",
    "start_section",
    "end_section",
    "model",
    "cv",
    "target",
    "xai_method",
    "report_path",
    "notes",
]

DATA_SELECTION_COLUMNS = [
    "Data_Slice_ID",
    "Slice_Name",
    "Purpose",
    "Subject_Scope",
    "Session_Scope",
    "Time_Window",
    "Task_Scope",
    "Modality_Scope",
    "Target_Type",
    "Label_Set",
    "Inclusion_Rule",
    "Exclusion_Rule",
    "Class_Balance_Policy",
    "Minimum_Samples_Rule",
    "Leakage_Risk",
    "Valid_Use_Case",
    "Thesis_Use",
    "Notes",
]

GROUPING_STRATEGY_COLUMNS = [
    "Grouping_Strategy_ID",
    "Strategy_Name",
    "Split_Family",
    "Train_Group_Unit",
    "Test_Group_Unit",
    "Grouping_Level",
    "Train_Rule",
    "Test_Rule",
    "Leakage_Safeguard",
    "Suitable_For",
    "Not_Suitable_For",
    "Interpretation_Boundary",
    "Typical_Experiments",
    "Thesis_Section",
    "Notes",
]

DECISION_COLUMNS = [
    "Decision_ID",
    "Decision_Topic",
    "Stage",
    "Candidate_Options",
    "Evidence_Experiments",
    "Decision_Criteria",
    "Chosen_Option",
    "Why_Chosen",
    "Why_Not_Others",
    "Risks_or_Tradeoffs",
    "Grounding_Section_or_Source_Packet",
    "Date_Locked",
    "Freeze_Status",
    "Thesis_Use",
]

CONFIRMATORY_COLUMNS = [
    "Experiment_ID",
    "Short_Title",
    "Final_Pipeline_Version",
    "Claim_Supported",
    "Status",
    "Eligible_to_Run",
    "Completed",
    "Required_for_Main_Claim",
    "Ready_for_Chapter_4",
    "Confirmatory_Eligible",
    "Main_Result_Summary",
    "Robustness_Status",
]

THESIS_MAP_COLUMNS = [
    "Experiment_ID",
    "Chapter",
    "Section",
    "Purpose_in_Thesis",
    "Main_or_Supporting",
    "Method_Choice_or_Method_Application_or_Discussion",
    "What_Claim_or_Decision_It_Supports",
    "Notes_for_Writing",
]

CLAIM_LEDGER_COLUMNS = [
    "Claim_ID",
    "Claim_Text",
    "Claim_Type",
    "Supported_By_Experiments",
    "Evidence_Status",
    "Writing_Location",
    "Interpretation_Limit",
    "Notes",
]

AI_USAGE_COLUMNS = [
    "Entry_ID",
    "Date",
    "Task",
    "Tool",
    "Input_Summary",
    "Output_Summary",
    "What_Was_Adopted",
    "Human_Verification_Status",
    "Thesis_Use",
    "Notes",
]

ETHICS_COLUMNS = [
    "Note_ID",
    "Experiment_ID_or_Decision_ID",
    "Topic",
    "Risk_or_Concern",
    "Mitigation_or_Response",
    "Thesis_Section",
    "Status",
    "Notes",
]

VOCABS = {
    "Category": [
        "Target-definition",
        "Split-design",
        "Model-comparison",
        "Preprocessing-feature",
        "Robustness",
        "Confirmatory core",
        "External exploratory",
    ],
    "Evidential_Role": [
        "Primary confirmatory",
        "Primary-supporting robustness",
        "Secondary decision-support",
        "Secondary decision-support / robustness",
        "Exploratory extension",
    ],
    "Stage": STAGE_V2,
    "Priority": ["Critical", "High", "Medium", "Low"],
    "Status": ["Planned", "Not started", "Running", "Completed", "On hold", "Dropped"],
    "Reporting_Destination": [
        "Chapter 3 Method choice",
        "Chapter 4 Main results",
        "Chapter 4 Supporting robustness",
        "Chapter 5 Discussion",
        "Appendix",
    ],
    "Reviewed": ["No", "Yes"],
    "Used_in_Thesis": ["No", "Yes"],
    "Freeze_Status": ["Open", "Locked"],
    "Run_Type": ["Pilot", "Decision-support", "Confirmatory", "Robustness", "Exploratory"],
    "Affects_Frozen_Pipeline": ["No", "Yes"],
    "Eligible_for_Method_Decision": ["No", "Yes"],
    "Claim_Type": ["Method-choice", "Confirmatory finding", "Robustness support", "Exploratory observation"],
    "Evidence_Status": ["open", "partial", "supported"],
    "Writing_Location": ["Chapter 3", "Chapter 4", "Chapter 5", "Appendix"],
    "Human_Verification_Status": ["Not reviewed", "Partially verified", "Verified"],
    "Ethics_Status": ["Open", "Monitoring", "Closed"],
    "Ethics_Topic": ["Data governance", "Bias/fairness", "Interpretation limits", "AI/tool transparency", "Reproducibility", "Other"],
    "Required_for_Main_Claim": ["No", "Yes"],
    "Ready_for_Chapter_4": ["No", "Yes"],
    "YesNo": ["No", "Yes"],
    "Subject_Scope": ["all_subjects", "single_subject", "cross_person_transfer"],
    "Session_Scope": ["all_sessions", "early_only", "late_only", "held_out_session", "custom_window"],
    "Task_Scope": ["pooled_all_tasks", "passive_only", "emo_only", "rating_only", "task_specific_other"],
    "Modality_Scope": ["pooled_all_modalities", "audio_only", "video_only", "audiovisual_only"],
    "Class_Balance_Policy": ["as_is", "balanced_subset", "weighted_only", "min_count_threshold"],
    "Split_Family": ["within_subject", "cross_person", "temporal", "weak_split_demo", "diagnostic"],
    "Imbalance_Status": ["unknown", "balanced", "mild_imbalance", "severe_imbalance"],
    "Leakage_Check_Status": ["not_checked", "passed", "warning", "failed"],
    "Thesis_Use_Tag": [
        "method_choice",
        "main_confirmatory",
        "supporting_robustness",
        "discussion_limitations",
        "appendix_exploratory",
    ],
    "Use_Case": [
        "method_choice",
        "confirmatory_within_person",
        "confirmatory_cross_person_transfer",
        "robustness_support",
        "weak_split_demo_only",
        "diagnostic_only",
        "exploratory_only",
    ],
    "Group_Unit": [
        "subject",
        "session",
        "task",
        "modality",
        "subject_session",
        "task_modality",
        "custom_group",
    ],
    "Grouping_Level": [
        "within_subject",
        "across_subject",
        "session_level",
        "task_level",
        "modality_level",
        "task_modality_level",
        "mixed_level",
    ],
    "Transfer_Direction": ["none_or_within_subject", "A_to_B", "B_to_A", "bidirectional", "custom"],
    "Leakage_Risk_Level": ["low", "medium", "high", "critical"],
    "Target_Type": ["coarse_affect", "binary_valence_like", "fine_emotion", "custom_target"],
    "Execution_Section": [
        "dataset_selection",
        "feature_cache_build",
        "feature_matrix_load",
        "spatial_validation",
        "model_fit",
        "interpretability",
        "evaluation",
    ],
    "Reuse_Policy": ["auto", "require_explicit_base", "disallow"],
    "Search_Optimization_Mode": ["deterministic_grid", "optuna"],
    "Search_Parameter_Scope": ["parameter", "segment", "xai"],
}

DEFINITIONS = [
    ("confirmatory", "Pre-specified, locked-pipeline evidence analysis supporting main thesis claims."),
    ("decision-support", "Method-choice analysis performed before locking confirmatory settings."),
    ("exploratory", "Hypothesis-generating extension outside confirmatory core."),
    ("data slice", "Explicit subset policy over subjects/sessions/tasks/modalities/labels used by runs."),
    ("grouping strategy", "Explicit train/test grouping logic and split family for claim-matched evaluation."),
    ("weak split demo", "Intentionally weak split used only to demonstrate inflation risk; not confirmatory evidence."),
    ("construct validity", "How well labels/features operationalize intended constructs."),
    ("internal validity", "Protection against confounds and leakage in design and execution."),
    ("statistical conclusion validity", "Reliability of metric-based inference."),
    ("external validity", "Extent findings transfer to other data contexts."),
    ("leakage", "Train-test information contamination that inflates apparent performance."),
    ("domain shift", "Distribution shift between train and test domains."),
    ("interpretability stability", "Consistency of model-behavior explanation signals across folds."),
]

COL = {
    "header_bg": "1F4E78",
    "header_fg": "FFFFFF",
    "title_bg": "D9E1F2",
    "zebra": "F8FAFC",
    "confirmatory": "E2F0D9",
    "exploratory": "FCE4D6",
    "critical": "FFF2CC",
    "dropped": "E7E6E6",
    "missing": "FBE5E7",
    "ok": "E2F0D9",
    "bad": "FBE5E7",
    "open": "FCE4D6",
}

THIN = Border(
    left=Side(style="thin", color="D0D7DE"),
    right=Side(style="thin", color="D0D7DE"),
    top=Side(style="thin", color="D0D7DE"),
    bottom=Side(style="thin", color="D0D7DE"),
)


def style_header(ws, row: int, n_cols: int) -> None:
    for c in range(1, n_cols + 1):
        cell = ws.cell(row=row, column=c)
        cell.fill = PatternFill("solid", fgColor=COL["header_bg"])
        cell.font = Font(color=COL["header_fg"], bold=True)
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = THIN


def style_body(ws, r1: int, r2: int, c1: int, c2: int, zebra: bool = True) -> None:
    fill = PatternFill("solid", fgColor=COL["zebra"])
    for r in range(r1, r2 + 1):
        for c in range(c1, c2 + 1):
            cell = ws.cell(r, c)
            cell.border = THIN
            cell.alignment = Alignment(vertical="top", wrap_text=True)
            if zebra and r % 2 == 0:
                cell.fill = fill


def set_widths(ws, widths: dict[str, float]) -> None:
    for col, width in widths.items():
        ws.column_dimensions[col].width = width


def add_table(ws, name: str, ref: str, style: str = "TableStyleMedium2") -> None:
    table = Table(displayName=name, ref=ref)
    table.tableStyleInfo = TableStyleInfo(
        name=style,
        showFirstColumn=False,
        showLastColumn=False,
        showRowStripes=True,
        showColumnStripes=False,
    )
    ws.add_table(table)


def add_list_validation(ws, formula: str, col: int, start: int, end: int, allow_blank: bool = True) -> None:
    dv = DataValidation(type="list", formula1=formula, allow_blank=allow_blank)
    ws.add_data_validation(dv)
    letter = get_column_letter(col)
    dv.add(f"{letter}{start}:{letter}{end}")


def add_named_list(wb: Workbook, list_name: str, sheet_name: str, col: int, start: int, end: int) -> None:
    letter = get_column_letter(col)
    wb.defined_names.add(DefinedName(name=list_name, attr_text=f"'{sheet_name}'!${letter}${start}:${letter}${end}"))


def add_dynamic_named_list(wb: Workbook, list_name: str, sheet_name: str, col: int, start: int) -> None:
    letter = get_column_letter(col)
    formula = (
        f"'{sheet_name}'!${letter}${start}:"
        f"INDEX('{sheet_name}'!${letter}:${letter},MATCH(\"zzz\",'{sheet_name}'!${letter}:${letter}))"
    )
    wb.defined_names.add(DefinedName(name=list_name, attr_text=formula))


def col_idx(columns: list[str], name: str) -> int:
    return columns.index(name) + 1


def build_experiments() -> list[dict[str, str]]:
    base = {
        "Priority": "Medium",
        "Decision_Supported": "Methodological decision support.",
        "Hypothesis_or_Expectation": "Evaluate under leakage-aware claim-matched split logic before interpretation.",
        "Held_Constant_Controls": "Dataset index version, train-only fitting policy, metric definitions, seed policy, artifact schema.",
        "Dataset_Scope": "Internal repeated-session BAS2 dataset (sub-001 and sub-002) unless explicitly external.",
        "Target_Definition": "coarse_affect unless this experiment manipulates target granularity.",
        "Split_Logic": "Claim-matched leakage-safe split strategy.",
        "Leakage_Risk": "Performance inflation if split logic or fit scope leaks train/test information.",
        "Leakage_Control": "Predefined split manifests, train-only fitting, frozen transfer where applicable, no test-driven tuning.",
        "Models_Compared": "ridge baseline unless model family is manipulated.",
        "Primary_Metric": "Balanced accuracy",
        "Secondary_Metrics": "Macro-F1; class-wise recall; accuracy",
        "Robustness_Checks": "Fold-level checks and stress controls where relevant.",
        "Decision_Criterion": "Pre-specified criterion recorded before interpretation.",
        "Success_Pattern": "Stable claim-consistent performance without leakage indicators.",
        "Failure_or_Warning_Pattern": "Weak-split inflation, instability, class collapse, or directionally inconsistent evidence.",
        "Threats_to_Validity": "Class imbalance, sample limits, domain shift, and model/split sensitivity.",
        "Reporting_Destination": "Chapter 3 Method choice",
        "Interpretation_Boundary": "Interpret only under stated split logic/evidence tier; no causal or localization claims.",
        "Status": "Planned",
        "Owner": "Khaled (thesis author)",
        "Outcome_Summary": "",
        "Decision_Taken": "",
        "Notes": "",
    }
    rows = [
        {"Experiment_ID": "E01", "Short_Title": "Target granularity experiment", "Category": "Target-definition", "Evidential_Role": "Secondary decision-support", "Stage": "Stage 1 - Target lock", "Priority": "High", "Decision_Supported": "Final target lock", "Exact_Question": "Should the thesis use fine-grained emotion labels, three-class coarse affect, or binary valence-like labels?", "Hypothesis_or_Expectation": "coarse_affect likely balances construct validity, class balance, and learnability.", "Manipulated_Factor": "Target definition", "Target_Definition": "fine emotion vs coarse_affect vs binary valence-like", "Split_Logic": "within_subject_loso_session for primary comparison; frozen transfer as secondary check", "Decision_Criterion": "Prefer target balancing construct validity, class balance, sample sufficiency, and learnability under claim-matched evaluation."},
        {"Experiment_ID": "E02", "Short_Title": "Task pooling experiment", "Category": "Target-definition", "Evidential_Role": "Secondary decision-support", "Stage": "Stage 1 - Target lock", "Decision_Supported": "Task pooling policy", "Exact_Question": "Should the main target be learned across all tasks together, or separately by task?", "Manipulated_Factor": "Pooling strategy by task"},
        {"Experiment_ID": "E03", "Short_Title": "Modality pooling experiment", "Category": "Target-definition", "Evidential_Role": "Secondary decision-support", "Stage": "Stage 1 - Target lock", "Decision_Supported": "Modality pooling policy", "Exact_Question": "Should audio, video, and audiovisual conditions be pooled or modeled separately?", "Manipulated_Factor": "Pooling strategy by modality"},
        {"Experiment_ID": "E04", "Short_Title": "Split-strength stress test", "Category": "Split-design", "Evidential_Role": "Secondary decision-support", "Stage": "Stage 2 - Split lock", "Priority": "High", "Decision_Supported": "Primary split policy", "Exact_Question": "How much do conclusions change under weaker split strategies compared with session-held-out evaluation?", "Manipulated_Factor": "Split logic", "Split_Logic": "record-wise random split vs weaker grouped split vs within-person leave-one-session-out", "Leakage_Risk": "High under weak splits", "Decision_Criterion": "Retain strictest split matching claim; weaker splits only as inflation/leakage demonstrations."},
        {"Experiment_ID": "E05", "Short_Title": "Cross-person transfer design", "Category": "Split-design", "Evidential_Role": "Secondary decision-support", "Stage": "Stage 2 - Split lock", "Priority": "High", "Decision_Supported": "Cross-person transfer framing", "Exact_Question": "How should cross-person generalization be framed and evaluated?", "Manipulated_Factor": "Transfer protocol", "Split_Logic": "frozen train-on-A/test-on-B and reverse direction", "Leakage_Control": "Fit only on train subject; no test subject re-fit/re-tune", "Decision_Criterion": "Keep frozen directional transfer as clean cross-case test.", "Interpretation_Boundary": "Transfer evidence under domain shift, not population-level generalization."},
        {"Experiment_ID": "E06", "Short_Title": "Linear model family comparison", "Category": "Model-comparison", "Evidential_Role": "Secondary decision-support", "Stage": "Stage 3 - Model lock", "Priority": "High", "Decision_Supported": "Model family lock", "Exact_Question": "Which linear model is the most appropriate default: ridge, logistic regression, or linear SVM?", "Manipulated_Factor": "Model family", "Models_Compared": "ridge vs logreg vs linearsvc", "Decision_Criterion": "Prefer simplest model with comparable predictive performance and better robustness/stability."},
        {"Experiment_ID": "E07", "Short_Title": "Class weighting experiment", "Category": "Model-comparison", "Evidential_Role": "Secondary decision-support", "Stage": "Stage 3 - Model lock", "Decision_Supported": "Weighting policy", "Exact_Question": "Does class weighting improve fairness across classes without creating instability?", "Manipulated_Factor": "Weighting strategy"},
        {"Experiment_ID": "E08", "Short_Title": "Hyperparameter strategy experiment", "Category": "Model-comparison", "Evidential_Role": "Secondary decision-support", "Stage": "Stage 3 - Model lock", "Decision_Supported": "Tuning policy", "Exact_Question": "Are fixed conservative hyperparameters sufficient, or does light nested tuning materially improve results?", "Manipulated_Factor": "Tuning strategy", "Decision_Criterion": "Keep fixed settings unless nested tuning yields a clear and stable gain."},
        {"Experiment_ID": "E09", "Short_Title": "Whole-brain versus ROI experiment", "Category": "Preprocessing-feature", "Evidential_Role": "Secondary decision-support", "Stage": "Stage 4 - Feature/preprocessing lock", "Decision_Supported": "Feature representation lock", "Exact_Question": "Should the primary representation remain whole-brain masked voxels or use theory-justified ROIs?", "Manipulated_Factor": "Feature space"},
        {"Experiment_ID": "E10", "Short_Title": "Dimensionality reduction experiment", "Category": "Preprocessing-feature", "Evidential_Role": "Secondary decision-support", "Stage": "Stage 4 - Feature/preprocessing lock", "Decision_Supported": "Feature reduction policy", "Exact_Question": "Does unsupervised dimensionality reduction improve performance or stability enough to justify added complexity?", "Manipulated_Factor": "Feature reduction strategy"},
        {"Experiment_ID": "E11", "Short_Title": "Scaling and normalization experiment", "Category": "Preprocessing-feature", "Evidential_Role": "Secondary decision-support", "Stage": "Stage 4 - Feature/preprocessing lock", "Decision_Supported": "Scaling policy lock", "Exact_Question": "How sensitive are the results to scaling strategy?", "Manipulated_Factor": "Scaling policy"},
        {"Experiment_ID": "E12", "Short_Title": "Permutation test experiment", "Category": "Robustness", "Evidential_Role": "Primary-supporting robustness", "Stage": "Stage 6 - Robustness analysis", "Priority": "High", "Decision_Supported": "Chance-level distinguishability support", "Exact_Question": "Are the main predictive results distinguishable from chance under the exact same split logic?", "Manipulated_Factor": "Label structure (true vs permuted)", "Split_Logic": "exact confirmatory split logic with train-only permutation", "Secondary_Metrics": "Empirical p-value; null distribution summary", "Reporting_Destination": "Chapter 4 Supporting robustness"},
        {"Experiment_ID": "E13", "Short_Title": "Trivial baseline experiment", "Category": "Robustness", "Evidential_Role": "Primary-supporting robustness", "Stage": "Stage 6 - Robustness analysis", "Priority": "High", "Decision_Supported": "Non-trivial predictive value support", "Exact_Question": "Does the final model clearly outperform trivial baselines?", "Manipulated_Factor": "Baseline type", "Models_Compared": "Final locked model vs trivial baselines", "Reporting_Destination": "Chapter 4 Supporting robustness"},
        {"Experiment_ID": "E14", "Short_Title": "Stability of explanation experiment", "Category": "Robustness", "Evidential_Role": "Primary-supporting robustness", "Stage": "Stage 6 - Robustness analysis", "Priority": "High", "Decision_Supported": "Interpretability stability support", "Exact_Question": "Are linear coefficients or importance patterns stable across held-out-session folds?", "Manipulated_Factor": "Fold identity", "Primary_Metric": "Mean pairwise coefficient correlation", "Secondary_Metrics": "Sign consistency; top-k overlap", "Reporting_Destination": "Chapter 4 Supporting robustness", "Interpretation_Boundary": "Model-behavior evidence only; not direct neural localization."},
        {"Experiment_ID": "E15", "Short_Title": "Sensitivity to subset exclusion", "Category": "Robustness", "Evidential_Role": "Secondary decision-support / robustness", "Stage": "Stage 6 - Robustness analysis", "Decision_Supported": "Sensitivity framing", "Exact_Question": "Do the main conclusions depend excessively on one subset, such as one task or modality?", "Manipulated_Factor": "Subset inclusion", "Reporting_Destination": "Chapter 5 Discussion"},
        {"Experiment_ID": "E16", "Short_Title": "Final within-person confirmatory analysis", "Category": "Confirmatory core", "Evidential_Role": "Primary confirmatory", "Stage": "Stage 5 - Confirmatory analysis", "Priority": "Critical", "Decision_Supported": "Primary confirmatory claim", "Exact_Question": "Does the final locked pipeline show meaningful within-person held-out-session generalization?", "Manipulated_Factor": "None; final frozen pipeline", "Target_Definition": "Locked target after Stage 1 target lock", "Split_Logic": "within_subject_loso_session", "Models_Compared": "Final locked model", "Decision_Criterion": "Primary evidential core of thesis.", "Reporting_Destination": "Chapter 4 Main results"},
        {"Experiment_ID": "E17", "Short_Title": "Final cross-person transfer analysis", "Category": "Confirmatory core", "Evidential_Role": "Primary confirmatory", "Stage": "Stage 5 - Confirmatory analysis", "Priority": "Critical", "Decision_Supported": "Secondary confirmatory transfer claim", "Exact_Question": "Does the final locked pipeline transfer meaningfully across individuals under frozen directional transfer?", "Manipulated_Factor": "None; final frozen pipeline", "Target_Definition": "Locked target after Stage 1 target lock", "Split_Logic": "frozen_cross_person_transfer both directions", "Models_Compared": "Final locked model", "Decision_Criterion": "Interpret as cross-case transfer evidence under domain shift.", "Reporting_Destination": "Chapter 4 Main results", "Interpretation_Boundary": "Directional transfer evidence only; not population-level generalization."},
        {"Experiment_ID": "E18", "Short_Title": "External open-source split replication", "Category": "External exploratory", "Evidential_Role": "Exploratory extension", "Stage": "Stage 7 - Exploratory extension", "Priority": "Low", "Decision_Supported": "External split-consistency exploration", "Exact_Question": "Does the same leakage-aware split logic materially affect conclusions on a compatible external open-source dataset?", "Manipulated_Factor": "Dataset source", "Dataset_Scope": "External open-source dataset if available and approved", "Reporting_Destination": "Appendix"},
        {"Experiment_ID": "E19", "Short_Title": "External open-source model portability", "Category": "External exploratory", "Evidential_Role": "Exploratory extension", "Stage": "Stage 7 - Exploratory extension", "Priority": "Low", "Decision_Supported": "External model portability exploration", "Exact_Question": "Does the relative ranking of reasonable model choices remain similar on external data?", "Manipulated_Factor": "Dataset source with same small model set", "Dataset_Scope": "External open-source dataset if available and approved", "Models_Compared": "ridge vs logreg vs linearsvc", "Reporting_Destination": "Appendix"},
    ]
    out: list[dict[str, str]] = []
    for row in rows:
        item = dict(base)
        item.update(row)
        out.append(item)
    return out


def fill_readme_sheet(ws) -> None:
    ws.merge_cells("A1:I1")
    ws["A1"] = "Thesis Experiment Program Workbook (v2)"
    ws["A1"].font = Font(size=16, bold=True)
    ws["A1"].fill = PatternFill("solid", fgColor=COL["title_bg"])
    ws["A1"].alignment = Alignment(horizontal="left")

    blocks = [
        ("Purpose", "Scientific control system for thesis experiment governance: pre-interpretation design, lock tracking, leakage-aware execution traceability, and chapter-ready evidence mapping."),
        ("Governance layers", "Experiment governance documents what each experiment is allowed to conclude. Data-slice governance documents what subset policy and grouping strategy were used in each run and what claims they can support."),
        ("Evidence tiers", "Primary confirmatory; Primary-supporting robustness; Secondary decision-support; Exploratory extension. Each tier has explicit interpretation boundaries."),
        ("Stage system", "Stage 1 Target lock -> Stage 2 Split lock -> Stage 3 Model lock -> Stage 4 Feature/preprocessing lock -> Stage 5 Confirmatory analysis -> Stage 6 Robustness analysis -> Stage 7 Exploratory extension."),
        ("Freeze policy", "Confirmatory eligibility requires D01-D07 locked in Decision_Log. Chapter 4 readiness additionally requires required confirmatory/supporting items completed and marked Ready_for_Chapter_4=YES."),
        ("Data_Selection_Design", "Defines allowed data slices (subject/session/task/modality/target scope, inclusion/exclusion, class-balance policy, leakage risk, and thesis-use boundary)."),
        ("Grouping_Strategy_Map", "Defines split family, train/test grouping units, leakage safeguards, and interpretation boundaries for each grouping strategy identifier."),
        ("Data_Profile", "Structured worksheet for manual or imported descriptive counts by subject/session/task/modality/target/class and by Data_Slice_ID, with formula-based sparsity/imbalance flags."),
        ("Claim boundaries", "Within-person held-out-session claims and cross-person transfer claims are different scientific claims and must not be interpreted interchangeably."),
        ("Weak-split policy", "Weak split results are allowed only as inflation demonstrations and diagnostic contrast. They are not confirmatory evidence."),
        ("Governance additions", "Claim_Ledger enforces claim discipline. AI_Usage_Log supports tool transparency and human verification traceability. Ethics_Governance_Notes supports risk/mitigation accountability."),
        ("Workflow", "Master_Experiments + Data_Selection_Design + Grouping_Strategy_Map -> Run_Log -> Decision_Log -> Confirmatory_Set -> Claim_Ledger/AI_Usage_Log/Ethics_Governance_Notes -> Thesis_Map -> Dashboard."),
    ]
    row = 3
    for title, text in blocks:
        ws[f"A{row}"] = title
        ws[f"A{row}"].font = Font(bold=True)
        ws[f"A{row}"].fill = PatternFill("solid", fgColor="EEF3FB")
        ws.merge_cells(start_row=row, start_column=2, end_row=row + 1, end_column=9)
        ws.cell(row=row, column=2, value=text).alignment = Alignment(wrap_text=True, vertical="top")
        for rr in (row, row + 1):
            for cc in range(1, 10):
                ws.cell(rr, cc).border = THIN
        row += 3

    summary_row = row
    ws[f"A{summary_row}"] = (
        "This workbook is aligned to thesis method-choice/method-application workflow, "
        "data-slice/grouping governance, and leakage-aware reporting requirements."
    )
    ws.merge_cells(start_row=summary_row, start_column=1, end_row=summary_row + 1, end_column=9)
    ws[f"A{summary_row}"].fill = PatternFill("solid", fgColor="FFF8E1")
    ws[f"A{summary_row}"].alignment = Alignment(wrap_text=True, vertical="top")
    ws[f"A{summary_row}"].border = THIN
    for cc in range(1, 10):
        ws.cell(summary_row + 1, cc).border = THIN

    set_widths(ws, {"A": 22, "B": 24, "C": 24, "D": 24, "E": 24, "F": 24, "G": 24, "H": 24, "I": 24})
    ws.freeze_panes = "A2"


def fill_master_sheet(ws) -> int:
    for i, h in enumerate(MASTER_COLUMNS, start=1):
        ws.cell(1, i, h)
    style_header(ws, 1, len(MASTER_COLUMNS))

    rows = build_experiments()
    for r, row in enumerate(rows, start=2):
        for c, name in enumerate(MASTER_COLUMNS[:-1], start=1):
            ws.cell(r, c, row.get(name, ""))
        ws.cell(
            r,
            32,
            f'=IF($AA{r}="Dropped","N/A",IF(AND($H{r}<>"",$J{r}<>"",$K{r}<>"",$M{r}<>"",$N{r}<>"",$P{r}<>"",$R{r}<>"",$U{r}<>"",$V{r}<>"",$W{r}<>"",$Z{r}<>"",$Y{r}<>""),"READY","INCOMPLETE"))',
        )

    last = 1 + len(rows)
    style_body(ws, 2, last, 1, len(MASTER_COLUMNS))
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = f"A1:{get_column_letter(len(MASTER_COLUMNS))}{last}"

    set_widths(
        ws,
        {
            "A": 13, "B": 30, "C": 20, "D": 32, "E": 28, "F": 10, "G": 24, "H": 46, "I": 34, "J": 24,
            "K": 42, "L": 30, "M": 30, "N": 34, "O": 24, "P": 36, "Q": 24, "R": 18, "S": 26, "T": 36,
            "U": 36, "V": 34, "W": 36, "X": 34, "Y": 30, "Z": 36, "AA": 14, "AB": 18, "AC": 32,
            "AD": 28, "AE": 32, "AF": 16,
        },
    )

    add_list_validation(ws, "=List_Category", 3, 2, 500, allow_blank=False)
    add_list_validation(ws, "=List_Evidential_Role", 4, 2, 500, allow_blank=False)
    add_list_validation(ws, "=List_Stage", 5, 2, 500, allow_blank=False)
    add_list_validation(ws, "=List_Priority", 6, 2, 500, allow_blank=False)
    add_list_validation(ws, "=List_Reporting_Destination", 25, 2, 500, allow_blank=False)
    add_list_validation(ws, "=List_Status", 27, 2, 500, allow_blank=False)

    rng = f"A2:AF{max(last, 200)}"
    ws.conditional_formatting.add(rng, FormulaRule(formula=['$D2="Primary confirmatory"'], fill=PatternFill("solid", fgColor=COL["confirmatory"])))
    ws.conditional_formatting.add(rng, FormulaRule(formula=['ISNUMBER(SEARCH("Exploratory",$D2))'], fill=PatternFill("solid", fgColor=COL["exploratory"])))
    ws.conditional_formatting.add(rng, FormulaRule(formula=['$F2="Critical"'], fill=PatternFill("solid", fgColor=COL["critical"])))
    ws.conditional_formatting.add(rng, FormulaRule(formula=['$AA2="Dropped"'], fill=PatternFill("solid", fgColor=COL["dropped"]), font=Font(color="6E6E6E", italic=True)))
    ws.conditional_formatting.add(
        rng,
        FormulaRule(
            formula=['OR($H2="",$J2="",$K2="",$M2="",$N2="",$P2="",$R2="",$U2="",$V2="",$W2="",$Z2="",$Y2="")'],
            fill=PatternFill("solid", fgColor=COL["missing"]),
        ),
    )
    return last


def fill_experiment_definitions_sheet(ws) -> int:
    for i, h in enumerate(EXPERIMENT_DEFINITIONS_COLUMNS, start=1):
        ws.cell(1, i, h)
    style_header(ws, 1, len(EXPERIMENT_DEFINITIONS_COLUMNS))

    seed_rows = [
        {
            "experiment_id": "E16",
            "enabled": "No",
            "start_section": "dataset_selection",
            "end_section": "evaluation",
            "base_artifact_id": "",
            "target": "coarse_affect",
            "cv": "within_subject_loso_session",
            "model": "ridge",
            "subject": "sub-001",
            "train_subject": "",
            "test_subject": "",
            "filter_task": "",
            "filter_modality": "",
            "reuse_policy": "auto",
            "search_space_id": "",
        },
        {
            "experiment_id": "E17",
            "enabled": "No",
            "start_section": "dataset_selection",
            "end_section": "evaluation",
            "base_artifact_id": "",
            "target": "coarse_affect",
            "cv": "frozen_cross_person_transfer",
            "model": "ridge",
            "subject": "",
            "train_subject": "sub-001",
            "test_subject": "sub-002",
            "filter_task": "",
            "filter_modality": "",
            "reuse_policy": "auto",
            "search_space_id": "",
        },
    ]
    for r, row in enumerate(seed_rows, start=2):
        for c, name in enumerate(EXPERIMENT_DEFINITIONS_COLUMNS, start=1):
            ws.cell(r, c, row.get(name, ""))

    last = 81
    style_body(ws, 2, last, 1, len(EXPERIMENT_DEFINITIONS_COLUMNS))
    add_table(ws, "ExperimentDefinitionsTable", f"A1:O{last}", style="TableStyleMedium2")
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = f"A1:O{last}"

    add_list_validation(ws, "=List_Experiment_ID", col_idx(EXPERIMENT_DEFINITIONS_COLUMNS, "experiment_id"), 2, 1000)
    add_list_validation(ws, "=List_YesNo", col_idx(EXPERIMENT_DEFINITIONS_COLUMNS, "enabled"), 2, 1000)
    add_list_validation(ws, "=List_Execution_Section", col_idx(EXPERIMENT_DEFINITIONS_COLUMNS, "start_section"), 2, 1000)
    add_list_validation(ws, "=List_Execution_Section", col_idx(EXPERIMENT_DEFINITIONS_COLUMNS, "end_section"), 2, 1000)
    add_list_validation(ws, "=List_Reuse_Policy", col_idx(EXPERIMENT_DEFINITIONS_COLUMNS, "reuse_policy"), 2, 1000)
    add_list_validation(ws, "=List_Search_Space_ID", col_idx(EXPERIMENT_DEFINITIONS_COLUMNS, "search_space_id"), 2, 1000)

    set_widths(
        ws,
        {
            "A": 14,
            "B": 10,
            "C": 20,
            "D": 20,
            "E": 28,
            "F": 18,
            "G": 28,
            "H": 14,
            "I": 12,
            "J": 14,
            "K": 14,
            "L": 16,
            "M": 18,
            "N": 18,
            "O": 16,
        },
    )
    return last


def fill_search_spaces_sheet(ws, wb: Workbook) -> int:
    seed_rows = [
        {
            "search_space_id": "SS01",
            "enabled": "No",
            "optimization_mode": "deterministic_grid",
            "parameter_name": "model",
            "parameter_values": "ridge|logreg|linearsvc",
            "parameter_scope": "parameter",
            "objective_metric": "balanced_accuracy",
            "max_trials": "",
            "notes": "Example model-family search.",
        },
        {
            "search_space_id": "SS01",
            "enabled": "No",
            "optimization_mode": "deterministic_grid",
            "parameter_name": "start_section",
            "parameter_values": "dataset_selection|feature_matrix_load",
            "parameter_scope": "segment",
            "objective_metric": "balanced_accuracy",
            "max_trials": "",
            "notes": "Example section-start search.",
        },
    ]
    last = _fill_simple_structured_sheet(
        ws=ws,
        columns=SEARCH_SPACES_COLUMNS,
        table_name="SearchSpacesTable",
        title="Search Space Definitions",
        width_map={
            "A": 16,
            "B": 10,
            "C": 20,
            "D": 24,
            "E": 34,
            "F": 14,
            "G": 20,
            "H": 12,
            "I": 32,
        },
        starter_rows=seed_rows,
    )
    add_list_validation(ws, "=List_YesNo", col_idx(SEARCH_SPACES_COLUMNS, "enabled"), 3, 1000)
    add_list_validation(
        ws,
        "=List_Search_Optimization_Mode",
        col_idx(SEARCH_SPACES_COLUMNS, "optimization_mode"),
        3,
        1000,
    )
    add_list_validation(
        ws,
        "=List_Search_Parameter_Scope",
        col_idx(SEARCH_SPACES_COLUMNS, "parameter_scope"),
        3,
        1000,
    )
    add_dynamic_named_list(
        wb,
        "List_Search_Space_ID",
        ws.title,
        col_idx(SEARCH_SPACES_COLUMNS, "search_space_id"),
        3,
    )
    return last


def _fill_simple_structured_sheet(
    ws,
    columns: list[str],
    table_name: str,
    title: str,
    width_map: dict[str, float],
    starter_rows: list[dict[str, str]] | None = None,
) -> int:
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=len(columns))
    ws.cell(1, 1, title)
    ws.cell(1, 1).font = Font(size=12, bold=True)
    ws.cell(1, 1).fill = PatternFill("solid", fgColor=COL["title_bg"])
    ws.cell(1, 1).alignment = Alignment(horizontal="left")
    for i, h in enumerate(columns, start=1):
        ws.cell(2, i, h)
    style_header(ws, 2, len(columns))

    if starter_rows:
        for r, row in enumerate(starter_rows, start=3):
            for c, name in enumerate(columns, start=1):
                ws.cell(r, c, row.get(name, ""))

    last = 61
    style_body(ws, 3, last, 1, len(columns))
    end_col = get_column_letter(len(columns))
    add_table(ws, table_name, f"A2:{end_col}{last}", style="TableStyleMedium6")
    ws.freeze_panes = "A3"
    ws.auto_filter.ref = f"A2:{end_col}{last}"
    set_widths(ws, width_map)
    return last


def fill_artifact_registry_sheet(ws) -> int:
    starter = [
        {
            "artifact_id": "",
            "artifact_type": "",
            "run_id": "",
            "status": "",
            "created_at": "",
            "path": "",
            "upstream_artifact_ids": "",
            "config_hash": "",
            "code_ref": "",
            "notes": "Optional workbook mirror of machine registry for audit snapshots.",
        }
    ]
    return _fill_simple_structured_sheet(
        ws=ws,
        columns=ARTIFACT_REGISTRY_COLUMNS,
        table_name="WorkbookArtifactRegistryTable",
        title="Workbook Artifact Registry Snapshot",
        width_map={
            "A": 32,
            "B": 20,
            "C": 24,
            "D": 14,
            "E": 24,
            "F": 40,
            "G": 32,
            "H": 24,
            "I": 20,
            "J": 34,
        },
        starter_rows=starter,
    )


def fill_fixed_configs_sheet(ws) -> int:
    starter = [
        {
            "config_key": "default_target",
            "config_value": "coarse_affect",
            "scope": "global",
            "locked": "No",
            "owner": "Khaled",
            "last_updated": "",
            "notes": "Machine-readable defaults for execution templates.",
        },
        {
            "config_key": "default_model",
            "config_value": "ridge",
            "scope": "global",
            "locked": "No",
            "owner": "Khaled",
            "last_updated": "",
            "notes": "",
        },
    ]
    last = _fill_simple_structured_sheet(
        ws=ws,
        columns=FIXED_CONFIGS_COLUMNS,
        table_name="FixedConfigsTable",
        title="Fixed Configurations and Locks",
        width_map={
            "A": 28,
            "B": 28,
            "C": 14,
            "D": 10,
            "E": 16,
            "F": 16,
            "G": 36,
        },
        starter_rows=starter,
    )
    add_list_validation(ws, "=List_YesNo", col_idx(FIXED_CONFIGS_COLUMNS, "locked"), 3, 1000)
    return last


def fill_objectives_sheet(ws) -> int:
    starter = [
        {
            "objective_id": "OBJ01",
            "objective_text": "Lock target definition under leakage-aware evaluation.",
            "stage": "Stage 1 - Target lock",
            "linked_experiment_id": "E01",
            "primary_metric": "balanced_accuracy",
            "success_criterion": "Decision D01 locked with pre-registered rationale.",
            "status": "Planned",
            "notes": "",
        }
    ]
    last = _fill_simple_structured_sheet(
        ws=ws,
        columns=OBJECTIVES_COLUMNS,
        table_name="ObjectivesTable",
        title="Program Objectives",
        width_map={
            "A": 12,
            "B": 44,
            "C": 28,
            "D": 18,
            "E": 20,
            "F": 34,
            "G": 14,
            "H": 30,
        },
        starter_rows=starter,
    )
    add_list_validation(ws, "=List_Stage", col_idx(OBJECTIVES_COLUMNS, "stage"), 3, 1000)
    add_list_validation(ws, "=List_Experiment_ID", col_idx(OBJECTIVES_COLUMNS, "linked_experiment_id"), 3, 1000)
    add_list_validation(ws, "=List_Status", col_idx(OBJECTIVES_COLUMNS, "status"), 3, 1000)
    return last


def fill_machine_status_sheet(ws) -> int:
    starter = [
        {
            "machine_id": "M01",
            "hostname": "",
            "environment_name": "thesis-dev",
            "python_version": "",
            "gpu": "",
            "status": "Monitoring",
            "last_checked": "",
            "notes": "Track execution environment snapshots used for thesis runs.",
        }
    ]
    last = _fill_simple_structured_sheet(
        ws=ws,
        columns=MACHINE_STATUS_COLUMNS,
        table_name="MachineStatusTable",
        title="Machine Status and Environment",
        width_map={
            "A": 12,
            "B": 20,
            "C": 22,
            "D": 18,
            "E": 20,
            "F": 14,
            "G": 16,
            "H": 34,
        },
        starter_rows=starter,
    )
    add_list_validation(ws, "=List_Ethics_Status", col_idx(MACHINE_STATUS_COLUMNS, "status"), 3, 1000)
    return last


def fill_trial_results_sheet(ws) -> int:
    last = _fill_simple_structured_sheet(
        ws=ws,
        columns=TRIAL_RESULTS_COLUMNS,
        table_name="TrialResultsTable",
        title="Trial Results (Manual Import or Future Sync)",
        width_map={
            "A": 22,
            "B": 14,
            "C": 24,
            "D": 14,
            "E": 22,
            "F": 18,
            "G": 34,
            "H": 34,
            "I": 28,
            "J": 30,
        },
        starter_rows=None,
    )
    add_list_validation(ws, "=List_Experiment_ID", col_idx(TRIAL_RESULTS_COLUMNS, "experiment_id"), 3, 1000)
    add_list_validation(ws, "=List_Status", col_idx(TRIAL_RESULTS_COLUMNS, "status"), 3, 1000)
    return last


def fill_summary_outputs_sheet(ws) -> int:
    return _fill_simple_structured_sheet(
        ws=ws,
        columns=SUMMARY_OUTPUTS_COLUMNS,
        table_name="SummaryOutputsTable",
        title="Machine Summary Outputs (Best Runs and Patterns)",
        width_map={
            "A": 18,
            "B": 22,
            "C": 20,
            "D": 18,
            "E": 26,
            "F": 14,
            "G": 18,
            "H": 18,
            "I": 14,
            "J": 24,
            "K": 18,
            "L": 20,
            "M": 34,
            "N": 36,
        },
        starter_rows=None,
    )


def fill_data_selection_design_sheet(ws, wb: Workbook) -> int:
    for i, h in enumerate(DATA_SELECTION_COLUMNS, start=1):
        ws.cell(1, i, h)
    style_header(ws, 1, len(DATA_SELECTION_COLUMNS))

    rows = [
        {
            "Data_Slice_ID": "DS01",
            "Slice_Name": "All subjects, all sessions, pooled task, pooled modality, coarse_affect",
            "Purpose": "Global baseline for target/split/model method-choice comparisons.",
            "Subject_Scope": "all_subjects",
            "Session_Scope": "all_sessions",
            "Time_Window": "full_study_window",
            "Task_Scope": "pooled_all_tasks",
            "Modality_Scope": "pooled_all_modalities",
            "Target_Type": "coarse_affect",
            "Label_Set": "coarse_affect",
            "Inclusion_Rule": "All QC-passed repeated-session beta maps with valid labels.",
            "Exclusion_Rule": "Drop failed-preprocessing rows and missing targets.",
            "Class_Balance_Policy": "as_is",
            "Minimum_Samples_Rule": "Prefer >=20 per class before claim-level interpretation.",
            "Leakage_Risk": "medium",
            "Valid_Use_Case": "method_choice",
            "Thesis_Use": "method_choice",
            "Notes": "",
        },
        {
            "Data_Slice_ID": "DS02",
            "Slice_Name": "Subject-specific pooled-task pooled-modality",
            "Purpose": "Primary within-person held-out-session claim support.",
            "Subject_Scope": "single_subject",
            "Session_Scope": "held_out_session",
            "Time_Window": "full_study_window",
            "Task_Scope": "pooled_all_tasks",
            "Modality_Scope": "pooled_all_modalities",
            "Target_Type": "coarse_affect",
            "Label_Set": "coarse_affect",
            "Inclusion_Rule": "One subject at a time with all eligible sessions.",
            "Exclusion_Rule": "No cross-subject pooling inside fold-level fit.",
            "Class_Balance_Policy": "weighted_only",
            "Minimum_Samples_Rule": "Per-class floor checked per subject.",
            "Leakage_Risk": "low",
            "Valid_Use_Case": "confirmatory_within_person",
            "Thesis_Use": "main_confirmatory",
            "Notes": "Matches held-out-session inference boundary.",
        },
        {
            "Data_Slice_ID": "DS03",
            "Slice_Name": "Task-specific pooled-modality",
            "Purpose": "Task pooling vs task-specific method-choice and diagnostics.",
            "Subject_Scope": "all_subjects",
            "Session_Scope": "all_sessions",
            "Time_Window": "full_study_window",
            "Task_Scope": "task_specific_other",
            "Modality_Scope": "pooled_all_modalities",
            "Target_Type": "coarse_affect",
            "Label_Set": "coarse_affect by task",
            "Inclusion_Rule": "Restrict each run to one task family.",
            "Exclusion_Rule": "Exclude off-task records for that run.",
            "Class_Balance_Policy": "as_is",
            "Minimum_Samples_Rule": "Minimum counts checked per task-specific slice.",
            "Leakage_Risk": "medium",
            "Valid_Use_Case": "method_choice",
            "Thesis_Use": "method_choice",
            "Notes": "",
        },
        {
            "Data_Slice_ID": "DS04",
            "Slice_Name": "Modality-specific pooled-task",
            "Purpose": "Modality pooling vs modality-specific method-choice checks.",
            "Subject_Scope": "all_subjects",
            "Session_Scope": "all_sessions",
            "Time_Window": "full_study_window",
            "Task_Scope": "pooled_all_tasks",
            "Modality_Scope": "audio_only",
            "Target_Type": "coarse_affect",
            "Label_Set": "coarse_affect",
            "Inclusion_Rule": "One modality family per run (audio/video/audiovisual variants).",
            "Exclusion_Rule": "Exclude other modalities in each modality-specific run.",
            "Class_Balance_Policy": "as_is",
            "Minimum_Samples_Rule": "Check sample sufficiency per modality variant.",
            "Leakage_Risk": "medium",
            "Valid_Use_Case": "method_choice",
            "Thesis_Use": "method_choice",
            "Notes": "Seed modality is audio_only; clone row for other modalities.",
        },
        {
            "Data_Slice_ID": "DS05",
            "Slice_Name": "Subject-specific task x modality subset",
            "Purpose": "Fine-grained diagnostic sensitivity slices.",
            "Subject_Scope": "single_subject",
            "Session_Scope": "all_sessions",
            "Time_Window": "full_study_window",
            "Task_Scope": "task_specific_other",
            "Modality_Scope": "audiovisual_only",
            "Target_Type": "coarse_affect",
            "Label_Set": "coarse_affect",
            "Inclusion_Rule": "Single subject + selected task + selected modality combination.",
            "Exclusion_Rule": "Exclude all records outside the chosen intersection.",
            "Class_Balance_Policy": "min_count_threshold",
            "Minimum_Samples_Rule": "Hard floor required before reporting.",
            "Leakage_Risk": "high",
            "Valid_Use_Case": "diagnostic_only",
            "Thesis_Use": "discussion_limitations",
            "Notes": "",
        },
        {
            "Data_Slice_ID": "DS06",
            "Slice_Name": "Early sessions only",
            "Purpose": "Temporal drift checks and early-to-late transfer design.",
            "Subject_Scope": "all_subjects",
            "Session_Scope": "early_only",
            "Time_Window": "early_window",
            "Task_Scope": "pooled_all_tasks",
            "Modality_Scope": "pooled_all_modalities",
            "Target_Type": "coarse_affect",
            "Label_Set": "coarse_affect",
            "Inclusion_Rule": "Use sessions pre-registered as early window.",
            "Exclusion_Rule": "Exclude late-window sessions.",
            "Class_Balance_Policy": "as_is",
            "Minimum_Samples_Rule": "Check class floor by temporal window.",
            "Leakage_Risk": "medium",
            "Valid_Use_Case": "robustness_support",
            "Thesis_Use": "supporting_robustness",
            "Notes": "",
        },
        {
            "Data_Slice_ID": "DS07",
            "Slice_Name": "Late sessions only",
            "Purpose": "Temporal drift checks and early-to-late transfer design.",
            "Subject_Scope": "all_subjects",
            "Session_Scope": "late_only",
            "Time_Window": "late_window",
            "Task_Scope": "pooled_all_tasks",
            "Modality_Scope": "pooled_all_modalities",
            "Target_Type": "coarse_affect",
            "Label_Set": "coarse_affect",
            "Inclusion_Rule": "Use sessions pre-registered as late window.",
            "Exclusion_Rule": "Exclude early-window sessions.",
            "Class_Balance_Policy": "as_is",
            "Minimum_Samples_Rule": "Check class floor by temporal window.",
            "Leakage_Risk": "medium",
            "Valid_Use_Case": "robustness_support",
            "Thesis_Use": "supporting_robustness",
            "Notes": "",
        },
        {
            "Data_Slice_ID": "DS08",
            "Slice_Name": "Balanced subset only",
            "Purpose": "Imbalance-aware sensitivity diagnostics.",
            "Subject_Scope": "all_subjects",
            "Session_Scope": "all_sessions",
            "Time_Window": "full_study_window",
            "Task_Scope": "pooled_all_tasks",
            "Modality_Scope": "pooled_all_modalities",
            "Target_Type": "coarse_affect",
            "Label_Set": "coarse_affect",
            "Inclusion_Rule": "Only include rows passing balancing policy.",
            "Exclusion_Rule": "Exclude classes below predefined minimum count floor.",
            "Class_Balance_Policy": "balanced_subset",
            "Minimum_Samples_Rule": "Set explicit class floor in analysis config.",
            "Leakage_Risk": "low",
            "Valid_Use_Case": "robustness_support",
            "Thesis_Use": "supporting_robustness",
            "Notes": "Do not replace confirmatory as-is evidence with this slice.",
        },
    ]

    for r, row in enumerate(rows, start=2):
        for c, name in enumerate(DATA_SELECTION_COLUMNS, start=1):
            ws.cell(r, c, row.get(name, ""))

    last = 41
    style_body(ws, 2, last, 1, len(DATA_SELECTION_COLUMNS))
    add_table(ws, "DataSelectionTable", f"A1:R{last}", style="TableStyleMedium3")
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = f"A1:R{last}"

    add_list_validation(ws, "=List_Subject_Scope", col_idx(DATA_SELECTION_COLUMNS, "Subject_Scope"), 2, 1000)
    add_list_validation(ws, "=List_Session_Scope", col_idx(DATA_SELECTION_COLUMNS, "Session_Scope"), 2, 1000)
    add_list_validation(ws, "=List_Task_Scope", col_idx(DATA_SELECTION_COLUMNS, "Task_Scope"), 2, 1000)
    add_list_validation(ws, "=List_Modality_Scope", col_idx(DATA_SELECTION_COLUMNS, "Modality_Scope"), 2, 1000)
    add_list_validation(ws, "=List_Target_Type", col_idx(DATA_SELECTION_COLUMNS, "Target_Type"), 2, 1000)
    add_list_validation(ws, "=List_Class_Balance_Policy", col_idx(DATA_SELECTION_COLUMNS, "Class_Balance_Policy"), 2, 1000)
    add_list_validation(ws, "=List_Leakage_Risk_Level", col_idx(DATA_SELECTION_COLUMNS, "Leakage_Risk"), 2, 1000)
    add_list_validation(ws, "=List_Use_Case", col_idx(DATA_SELECTION_COLUMNS, "Valid_Use_Case"), 2, 1000)
    add_list_validation(ws, "=List_Thesis_Use_Tag", col_idx(DATA_SELECTION_COLUMNS, "Thesis_Use"), 2, 1000)

    set_widths(
        ws,
        {
            "A": 13,
            "B": 42,
            "C": 34,
            "D": 18,
            "E": 16,
            "F": 18,
            "G": 20,
            "H": 22,
            "I": 16,
            "J": 22,
            "K": 34,
            "L": 32,
            "M": 20,
            "N": 24,
            "O": 14,
            "P": 22,
            "Q": 22,
            "R": 30,
        },
    )

    add_dynamic_named_list(
        wb,
        "List_Data_Slice_ID",
        ws.title,
        col_idx(DATA_SELECTION_COLUMNS, "Data_Slice_ID"),
        2,
    )
    return last


def fill_grouping_strategy_map_sheet(ws, wb: Workbook) -> int:
    for i, h in enumerate(GROUPING_STRATEGY_COLUMNS, start=1):
        ws.cell(1, i, h)
    style_header(ws, 1, len(GROUPING_STRATEGY_COLUMNS))

    rows = [
        {
            "Grouping_Strategy_ID": "GS01",
            "Strategy_Name": "within-subject LOSO-session",
            "Split_Family": "within_subject",
            "Train_Group_Unit": "subject_session",
            "Test_Group_Unit": "session",
            "Grouping_Level": "within_subject",
            "Train_Rule": "Per subject: train on all but one session.",
            "Test_Rule": "Per subject: test on held-out session.",
            "Leakage_Safeguard": "No held-out session rows in any train-side fit/tune stage.",
            "Suitable_For": "confirmatory_within_person",
            "Not_Suitable_For": "confirmatory_cross_person_transfer",
            "Interpretation_Boundary": "Supports within-person held-out-session claim only.",
            "Typical_Experiments": "E04,E16",
            "Thesis_Section": "Chapter 4 Main results",
            "Notes": "",
        },
        {
            "Grouping_Strategy_ID": "GS02",
            "Strategy_Name": "pooled-task within-subject LOSO-session",
            "Split_Family": "within_subject",
            "Train_Group_Unit": "subject_session",
            "Test_Group_Unit": "session",
            "Grouping_Level": "within_subject",
            "Train_Rule": "Within each subject, pool tasks and hold out one session.",
            "Test_Rule": "Evaluate on held-out session with pooled task scope.",
            "Leakage_Safeguard": "No test-session information in pooled train transforms.",
            "Suitable_For": "confirmatory_within_person",
            "Not_Suitable_For": "confirmatory_cross_person_transfer",
            "Interpretation_Boundary": "Within-person conclusion under pooled task policy.",
            "Typical_Experiments": "E02,E16",
            "Thesis_Section": "Chapter 3 Method choice",
            "Notes": "",
        },
        {
            "Grouping_Strategy_ID": "GS03",
            "Strategy_Name": "task-specific within-subject LOSO-session",
            "Split_Family": "within_subject",
            "Train_Group_Unit": "subject_session",
            "Test_Group_Unit": "session",
            "Grouping_Level": "task_level",
            "Train_Rule": "Within each subject/task, hold out one session.",
            "Test_Rule": "Test on same-task held-out session.",
            "Leakage_Safeguard": "Task-specific split manifests and train-only preprocessing.",
            "Suitable_For": "method_choice",
            "Not_Suitable_For": "confirmatory_cross_person_transfer",
            "Interpretation_Boundary": "Task-conditioned within-person evidence only.",
            "Typical_Experiments": "E02,E15",
            "Thesis_Section": "Chapter 3 Method choice",
            "Notes": "",
        },
        {
            "Grouping_Strategy_ID": "GS04",
            "Strategy_Name": "modality-specific within-subject LOSO-session",
            "Split_Family": "within_subject",
            "Train_Group_Unit": "subject_session",
            "Test_Group_Unit": "session",
            "Grouping_Level": "modality_level",
            "Train_Rule": "Within each subject/modality, hold out one session.",
            "Test_Rule": "Test on same-modality held-out session.",
            "Leakage_Safeguard": "Modality-specific fold manifests with strict train-only fit.",
            "Suitable_For": "method_choice",
            "Not_Suitable_For": "confirmatory_cross_person_transfer",
            "Interpretation_Boundary": "Modality-conditioned within-person evidence only.",
            "Typical_Experiments": "E03,E15",
            "Thesis_Section": "Chapter 3 Method choice",
            "Notes": "",
        },
        {
            "Grouping_Strategy_ID": "GS05",
            "Strategy_Name": "early-train late-test",
            "Split_Family": "temporal",
            "Train_Group_Unit": "session",
            "Test_Group_Unit": "session",
            "Grouping_Level": "session_level",
            "Train_Rule": "Train on pre-defined early sessions.",
            "Test_Rule": "Test on disjoint late sessions.",
            "Leakage_Safeguard": "Fixed temporal boundary and no late-session tuning.",
            "Suitable_For": "robustness_support",
            "Not_Suitable_For": "confirmatory_within_person",
            "Interpretation_Boundary": "Temporal transfer sensitivity, not the primary confirmatory claim.",
            "Typical_Experiments": "E04,E15",
            "Thesis_Section": "Chapter 4 Supporting robustness",
            "Notes": "",
        },
        {
            "Grouping_Strategy_ID": "GS06",
            "Strategy_Name": "frozen cross-person transfer A->B",
            "Split_Family": "cross_person",
            "Train_Group_Unit": "subject",
            "Test_Group_Unit": "subject",
            "Grouping_Level": "across_subject",
            "Train_Rule": "Fit on subject A only (frozen pipeline).",
            "Test_Rule": "Evaluate on subject B without refit/re-tune.",
            "Leakage_Safeguard": "No test-subject fitting; strict direction lock.",
            "Suitable_For": "confirmatory_cross_person_transfer",
            "Not_Suitable_For": "confirmatory_within_person",
            "Interpretation_Boundary": "Directional transfer under domain shift only.",
            "Typical_Experiments": "E05,E17",
            "Thesis_Section": "Chapter 4 Main results",
            "Notes": "",
        },
        {
            "Grouping_Strategy_ID": "GS07",
            "Strategy_Name": "frozen cross-person transfer B->A",
            "Split_Family": "cross_person",
            "Train_Group_Unit": "subject",
            "Test_Group_Unit": "subject",
            "Grouping_Level": "across_subject",
            "Train_Rule": "Fit on subject B only (frozen pipeline).",
            "Test_Rule": "Evaluate on subject A without refit/re-tune.",
            "Leakage_Safeguard": "No test-subject fitting; strict direction lock.",
            "Suitable_For": "confirmatory_cross_person_transfer",
            "Not_Suitable_For": "confirmatory_within_person",
            "Interpretation_Boundary": "Directional transfer under domain shift only.",
            "Typical_Experiments": "E05,E17",
            "Thesis_Section": "Chapter 4 Main results",
            "Notes": "",
        },
        {
            "Grouping_Strategy_ID": "GS08",
            "Strategy_Name": "weak split inflation demo",
            "Split_Family": "weak_split_demo",
            "Train_Group_Unit": "custom_group",
            "Test_Group_Unit": "custom_group",
            "Grouping_Level": "mixed_level",
            "Train_Rule": "Use intentionally weaker split variant for contrast.",
            "Test_Rule": "Evaluate inflation magnitude vs strict split.",
            "Leakage_Safeguard": "Tag as weak split and exclude from confirmatory claims.",
            "Suitable_For": "weak_split_demo_only",
            "Not_Suitable_For": "confirmatory_within_person",
            "Interpretation_Boundary": "Inflation demonstration only; non-confirmatory.",
            "Typical_Experiments": "E04",
            "Thesis_Section": "Chapter 3 Method choice",
            "Notes": "Permitted only as cautionary demonstration.",
        },
        {
            "Grouping_Strategy_ID": "GS09",
            "Strategy_Name": "task x modality diagnostic split",
            "Split_Family": "diagnostic",
            "Train_Group_Unit": "task_modality",
            "Test_Group_Unit": "task_modality",
            "Grouping_Level": "task_modality_level",
            "Train_Rule": "Train within selected task/modality intersections.",
            "Test_Rule": "Test on matched or held-out intersections per protocol.",
            "Leakage_Safeguard": "Pre-registered intersection mapping and no adaptive relabeling.",
            "Suitable_For": "diagnostic_only",
            "Not_Suitable_For": "confirmatory_within_person",
            "Interpretation_Boundary": "Diagnostic sensitivity only; not core evidence.",
            "Typical_Experiments": "E15",
            "Thesis_Section": "Chapter 5 Discussion",
            "Notes": "",
        },
    ]

    for r, row in enumerate(rows, start=2):
        for c, name in enumerate(GROUPING_STRATEGY_COLUMNS, start=1):
            ws.cell(r, c, row.get(name, ""))

    last = 41
    style_body(ws, 2, last, 1, len(GROUPING_STRATEGY_COLUMNS))
    add_table(ws, "GroupingStrategyTable", f"A1:O{last}", style="TableStyleMedium4")
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = f"A1:O{last}"

    add_list_validation(ws, "=List_Split_Family", col_idx(GROUPING_STRATEGY_COLUMNS, "Split_Family"), 2, 1000)
    add_list_validation(ws, "=List_Group_Unit", col_idx(GROUPING_STRATEGY_COLUMNS, "Train_Group_Unit"), 2, 1000)
    add_list_validation(ws, "=List_Group_Unit", col_idx(GROUPING_STRATEGY_COLUMNS, "Test_Group_Unit"), 2, 1000)
    add_list_validation(ws, "=List_Grouping_Level", col_idx(GROUPING_STRATEGY_COLUMNS, "Grouping_Level"), 2, 1000)
    add_list_validation(ws, "=List_Use_Case", col_idx(GROUPING_STRATEGY_COLUMNS, "Suitable_For"), 2, 1000)
    add_list_validation(ws, "=List_Use_Case", col_idx(GROUPING_STRATEGY_COLUMNS, "Not_Suitable_For"), 2, 1000)
    add_list_validation(ws, "=List_Reporting_Destination", col_idx(GROUPING_STRATEGY_COLUMNS, "Thesis_Section"), 2, 1000)

    set_widths(
        ws,
        {
            "A": 16,
            "B": 36,
            "C": 16,
            "D": 18,
            "E": 18,
            "F": 18,
            "G": 30,
            "H": 30,
            "I": 32,
            "J": 20,
            "K": 22,
            "L": 36,
            "M": 20,
            "N": 24,
            "O": 26,
        },
    )

    add_dynamic_named_list(
        wb,
        "List_Grouping_Strategy_ID",
        ws.title,
        col_idx(GROUPING_STRATEGY_COLUMNS, "Grouping_Strategy_ID"),
        2,
    )
    return last


def fill_data_profile_sheet(ws) -> int:
    ws.merge_cells("A1:J1")
    ws["A1"] = "Data Profile and Imbalance Diagnostics"
    ws["A1"].font = Font(size=14, bold=True)
    ws["A1"].fill = PatternFill("solid", fgColor=COL["title_bg"])
    ws["A1"].alignment = Alignment(horizontal="left")

    ws["A3"] = "How to use"
    ws["A3"].font = Font(bold=True)
    ws["A3"].fill = PatternFill("solid", fgColor="EEF3FB")
    ws.merge_cells("B3:J4")
    ws["B3"] = (
        "Paste exported counts into Count columns (or maintain manually). "
        "Formulas compute shares, imbalance flags, and sparsity flags."
    )
    ws["B3"].alignment = Alignment(wrap_text=True, vertical="top")
    for rr in range(3, 5):
        for cc in range(1, 11):
            ws.cell(rr, cc).border = THIN

    ws["A6"] = "Threshold controls"
    ws["A6"].font = Font(bold=True)
    ws["A6"].fill = PatternFill("solid", fgColor="EEF3FB")
    ws["A7"] = "Sparse count threshold"
    ws["A8"] = "Mild imbalance ratio (max/min)"
    ws["A9"] = "Severe imbalance ratio (max/min)"
    ws["B7"] = 20
    ws["B8"] = 2
    ws["B9"] = 3
    for rr in range(7, 10):
        ws.cell(rr, 1).border = THIN
        ws.cell(rr, 2).border = THIN
        if rr % 2 == 0:
            ws.cell(rr, 1).fill = PatternFill("solid", fgColor=COL["zebra"])
            ws.cell(rr, 2).fill = PatternFill("solid", fgColor=COL["zebra"])

    ws["D6"] = "Profile summary"
    ws["D6"].font = Font(bold=True)
    ws["D6"].fill = PatternFill("solid", fgColor="EEF3FB")
    summary_labels = [
        ("Total_subject_samples", '=SUM(B13:B20)'),
        ("Total_session_samples", '=SUM(B25:B34)'),
        ("Total_task_samples", '=SUM(B39:B45)'),
        ("Total_modality_samples", '=SUM(B50:B55)'),
        ("Total_class_samples", '=SUM(B60:B65)'),
        ("Class_imbalance_ratio_max_min", '=IFERROR(MAX(B60:B65)/MIN(B60:B65),"")'),
        (
            "Class_imbalance_flag",
            '=IF(B60="","",IF(OR(MIN(B60:B65)=0,G12>$B$9),"SEVERE_IMBALANCE",IF(G12>$B$8,"MILD_IMBALANCE","BALANCED")))',
        ),
    ]
    for i, (label, formula) in enumerate(summary_labels, start=7):
        ws.cell(i, 4, label)
        ws.cell(i, 7, formula)
        ws.cell(i, 4).border = THIN
        ws.cell(i, 7).border = THIN
        if i % 2 == 0:
            ws.cell(i, 4).fill = PatternFill("solid", fgColor=COL["zebra"])
            ws.cell(i, 7).fill = PatternFill("solid", fgColor=COL["zebra"])

    def write_count_block(
        start_row: int,
        title: str,
        key_header: str,
        keys: list[str],
        table_name: str,
    ) -> int:
        ws.merge_cells(start_row=start_row, start_column=1, end_row=start_row, end_column=5)
        ws.cell(start_row, 1, title)
        ws.cell(start_row, 1).font = Font(size=11, bold=True)
        ws.cell(start_row, 1).fill = PatternFill("solid", fgColor="EEF3FB")
        ws.cell(start_row, 1).border = THIN

        header_row = start_row + 1
        headers = [key_header, "Count", "Share_of_Block", "Flag", "Notes"]
        for c, h in enumerate(headers, start=1):
            ws.cell(header_row, c, h)
        style_header(ws, header_row, len(headers))

        data_start = header_row + 1
        data_end = data_start + len(keys) - 1
        for r, key in enumerate(keys, start=data_start):
            ws.cell(r, 1, key)
            ws.cell(r, 3, f'=IFERROR(B{r}/SUM($B${data_start}:$B${data_end}),0)')
            ws.cell(r, 4, f'=IF(B{r}="","",IF(B{r}<$B$7,"SPARSE","OK"))')
        style_body(ws, data_start, data_end, 1, 5)
        add_table(ws, table_name, f"A{header_row}:E{data_end}", style="TableStyleMedium6")
        return data_end

    write_count_block(
        11,
        "Counts by subject",
        "Subject_ID",
        ["sub-001", "sub-002", "sub-003", "sub-004", "sub-005", "sub-006", "sub-007", "sub-008"],
        "ProfileSubjectTable",
    )
    write_count_block(
        23,
        "Counts by session",
        "Session_ID",
        ["ses-01", "ses-02", "ses-03", "ses-04", "ses-05", "ses-06", "ses-07", "ses-08", "ses-09", "ses-10"],
        "ProfileSessionTable",
    )
    write_count_block(
        37,
        "Counts by task",
        "Task_ID",
        ["pooled_all_tasks", "passive_only", "emo_only", "rating_only", "task_specific_other", "custom_task_1", "custom_task_2"],
        "ProfileTaskTable",
    )
    write_count_block(
        48,
        "Counts by modality",
        "Modality_ID",
        ["pooled_all_modalities", "audio_only", "video_only", "audiovisual_only", "custom_modality_1", "custom_modality_2"],
        "ProfileModalityTable",
    )
    write_count_block(
        58,
        "Counts by target/class",
        "Target_or_Class_ID",
        ["coarse_affect_neg", "coarse_affect_neu", "coarse_affect_pos", "binary_neg", "binary_pos", "custom_class_1"],
        "ProfileTargetClassTable",
    )

    slice_start = 69
    ws.merge_cells(start_row=slice_start, start_column=1, end_row=slice_start, end_column=5)
    ws.cell(slice_start, 1, "Counts by Data_Slice_ID")
    ws.cell(slice_start, 1).font = Font(size=11, bold=True)
    ws.cell(slice_start, 1).fill = PatternFill("solid", fgColor="EEF3FB")
    ws.cell(slice_start, 1).border = THIN
    for c, h in enumerate(["Data_Slice_ID", "Count", "Share_of_Block", "Flag", "Notes"], start=1):
        ws.cell(slice_start + 1, c, h)
    style_header(ws, slice_start + 1, 5)

    data_start = slice_start + 2
    data_end = data_start + 9
    for r in range(data_start, data_end + 1):
        source_row = r - data_start + 2
        ws.cell(r, 1, f'=IFERROR(Data_Selection_Design!$A{source_row},"")')
        ws.cell(r, 3, f'=IFERROR(B{r}/SUM($B${data_start}:$B${data_end}),0)')
        ws.cell(r, 4, f'=IF(B{r}="","",IF(B{r}<$B$7,"SPARSE","OK"))')
    style_body(ws, data_start, data_end, 1, 5)
    add_table(ws, "ProfileDataSliceTable", f"A{slice_start + 1}:E{data_end}", style="TableStyleMedium6")

    class_flag_range = "$B$60:$B$65"
    for r in range(60, 66):
        ws.cell(
            r,
            4,
            f'=IF(B{r}="","",IF(OR(MIN({class_flag_range})=0,MAX({class_flag_range})/MIN({class_flag_range})>$B$9),'
            f'"SEVERE_IMBALANCE",IF(MAX({class_flag_range})/MIN({class_flag_range})>$B$8,"MILD_IMBALANCE","BALANCED")))',
        )

    set_widths(ws, {"A": 26, "B": 14, "C": 16, "D": 20, "E": 38, "F": 4, "G": 24, "H": 14, "I": 12, "J": 12})
    ws.freeze_panes = "A12"
    return data_end


def fill_run_log_sheet(ws) -> int:
    for i, h in enumerate(RUN_LOG_COLUMNS, start=1):
        ws.cell(1, i, h)
    style_header(ws, 1, len(RUN_LOG_COLUMNS))
    last = 41
    style_body(ws, 2, last, 1, len(RUN_LOG_COLUMNS))
    end_col = get_column_letter(len(RUN_LOG_COLUMNS))
    add_table(ws, "RunLogTable", f"A1:{end_col}{last}", style="TableStyleMedium9")
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = f"A1:{end_col}{last}"

    add_list_validation(ws, "=List_Data_Slice_ID", col_idx(RUN_LOG_COLUMNS, "Data_Slice_ID"), 2, 1000)
    add_list_validation(ws, "=List_Grouping_Strategy_ID", col_idx(RUN_LOG_COLUMNS, "Grouping_Strategy_ID"), 2, 1000)
    add_list_validation(ws, "=List_Transfer_Direction", col_idx(RUN_LOG_COLUMNS, "Transfer_Direction"), 2, 1000)
    add_list_validation(ws, "=List_Session_Scope", col_idx(RUN_LOG_COLUMNS, "Session_Coverage"), 2, 1000)
    add_list_validation(ws, "=List_Task_Scope", col_idx(RUN_LOG_COLUMNS, "Task_Coverage"), 2, 1000)
    add_list_validation(ws, "=List_Modality_Scope", col_idx(RUN_LOG_COLUMNS, "Modality_Coverage"), 2, 1000)
    add_list_validation(ws, "=List_Run_Type", col_idx(RUN_LOG_COLUMNS, "Run_Type"), 2, 1000)
    add_list_validation(ws, "=List_Affects_Frozen_Pipeline", col_idx(RUN_LOG_COLUMNS, "Affects_Frozen_Pipeline"), 2, 1000)
    add_list_validation(ws, "=List_Eligible_for_Method_Decision", col_idx(RUN_LOG_COLUMNS, "Eligible_for_Method_Decision"), 2, 1000)
    add_list_validation(ws, "=List_Imbalance_Status", col_idx(RUN_LOG_COLUMNS, "Imbalance_Status"), 2, 1000)
    add_list_validation(ws, "=List_Leakage_Check_Status", col_idx(RUN_LOG_COLUMNS, "Leakage_Check_Status"), 2, 1000)
    add_list_validation(ws, "=List_Reviewed", col_idx(RUN_LOG_COLUMNS, "Reviewed"), 2, 1000)
    add_list_validation(ws, "=List_Used_in_Thesis", col_idx(RUN_LOG_COLUMNS, "Used_in_Thesis"), 2, 1000)

    set_widths(
        ws,
        {
            "A": 18,
            "B": 13,
            "C": 12,
            "D": 20,
            "E": 22,
            "F": 12,
            "G": 16,
            "H": 22,
            "I": 30,
            "J": 12,
            "K": 16,
            "L": 24,
            "M": 22,
            "N": 22,
            "O": 18,
            "P": 18,
            "Q": 18,
            "R": 20,
            "S": 16,
            "T": 22,
            "U": 18,
            "V": 18,
            "W": 22,
            "X": 12,
            "Y": 20,
            "Z": 18,
            "AA": 20,
            "AB": 18,
            "AC": 18,
            "AD": 18,
            "AE": 30,
            "AF": 28,
            "AG": 34,
            "AH": 12,
            "AI": 14,
            "AJ": 36,
            "AK": 28,
        }
    )
    return last


def fill_decision_log_sheet(ws) -> int:
    for i, h in enumerate(DECISION_COLUMNS, start=1):
        ws.cell(1, i, h)
    style_header(ws, 1, len(DECISION_COLUMNS))

    rows = [
        ("D01", "Final target choice", "Stage 1 - Target lock", "fine emotion; coarse_affect; binary valence-like", "E01,E02,E03", "Construct validity + class balance + learnability under strict split", "To be locked after method-choice evidence review", "", "", "Tradeoff between construct validity and learnability", "To be linked after background/method revision", "", "Open", "Chapter 3 method lock feeding Chapter 4 confirmatory analyses"),
        ("D02", "Final split design", "Stage 2 - Split lock", "within_subject_loso_session; weaker alternatives for stress testing", "E04,E05", "Prefer strictest split matching claim while documenting weak-split inflation", "To be locked after split-strength evidence review", "", "", "Weak split inflation risk", "To be linked after method section split rationale update", "", "Open", "Defines primary inference logic and leakage controls"),
        ("D03", "Final model family", "Stage 3 - Model lock", "ridge; logreg; linearsvc", "E06", "Simplicity + robustness + comparable performance", "To be locked after model comparison", "", "", "Complexity vs stability tradeoff", "To be linked after model-choice subsection revision", "", "Open", "Sets final confirmatory model family"),
        ("D04", "Final weighting policy", "Stage 3 - Model lock", "No class weighting; balanced weighting", "E07", "Minority-class fairness gain without instability", "To be locked after weighting sensitivity", "", "", "Fairness-stability tension", "To be linked after class-imbalance discussion draft", "", "Open", "Controls fairness assumptions in final pipeline"),
        ("D05", "Final tuning policy", "Stage 3 - Model lock", "Fixed explicit settings; light nested tuning", "E08", "Keep fixed unless clear stable gain from tuning", "To be locked after tuning strategy review", "", "", "Overfitting and complexity risk with tuning", "To be linked after reproducibility subsection update", "", "Open", "Determines model configuration policy"),
        ("D06", "Final feature representation", "Stage 4 - Feature/preprocessing lock", "Whole-brain masked voxels; theory-justified ROI", "E09,E10", "Validity + stability + complexity tradeoff", "To be locked after feature-space evidence review", "", "", "Representation choice impacts validity and interpretability", "To be linked after feature-method section update", "", "Open", "Locks final feature representation"),
        ("D07", "Final scaling policy", "Stage 4 - Feature/preprocessing lock", "No scaling; standard/robust scaling (train-only)", "E11", "Train-only scaling with best stability and simplicity", "To be locked after scaling sensitivity", "", "", "Scaling can alter class boundaries", "To be linked after preprocessing subsection revision", "", "Open", "Locks preprocessing policy"),
        ("D08", "Use of external data", "Stage 7 - Exploratory extension", "No external data; external replication/portability", "E18,E19", "External work remains exploratory and non-core", "To be decided after confirmatory core completion", "", "", "Comparability and external validity limits", "To be linked after discussion/limitations revision", "", "Open", "Controls exploratory external scope and appendix use"),
    ]

    for r, row in enumerate(rows, start=2):
        for c, val in enumerate(row, start=1):
            ws.cell(r, c, val)

    last = 1 + len(rows)
    style_body(ws, 2, last, 1, len(DECISION_COLUMNS))
    add_table(ws, "DecisionLogTable", f"A1:N{last}", style="TableStyleMedium4")
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = f"A1:N{last}"
    add_list_validation(ws, "=List_Freeze_Status", 13, 2, 500, allow_blank=False)
    ws.conditional_formatting.add(f"M2:M{max(last,200)}", FormulaRule(formula=['$M2="Locked"'], fill=PatternFill("solid", fgColor=COL["ok"])))
    ws.conditional_formatting.add(f"M2:M{max(last,200)}", FormulaRule(formula=['$M2="Open"'], fill=PatternFill("solid", fgColor=COL["open"])))
    set_widths(
        ws,
        {"A": 12, "B": 24, "C": 30, "D": 36, "E": 18, "F": 34, "G": 34, "H": 22, "I": 22, "J": 28, "K": 34, "L": 12, "M": 14, "N": 30},
    )
    return last


def fill_confirmatory_sheet(ws) -> int:
    for i, h in enumerate(CONFIRMATORY_COLUMNS, start=1):
        ws.cell(1, i, h)
    style_header(ws, 1, len(CONFIRMATORY_COLUMNS))

    rows = [
        ("E12", "Permutation test experiment", "To be fixed after freeze lock", "Chance-distinguishability support", "YES"),
        ("E13", "Trivial baseline experiment", "To be fixed after freeze lock", "Non-trivial predictive value support", "YES"),
        ("E14", "Stability of explanation experiment", "To be fixed after freeze lock", "Model-behavior robustness support", "YES"),
        ("E16", "Final within-person confirmatory analysis", "To be fixed after freeze lock", "Primary within-person claim", "YES"),
        ("E17", "Final cross-person transfer analysis", "To be fixed after freeze lock", "Secondary transfer claim", "YES"),
    ]
    for r, row in enumerate(rows, start=2):
        ws.cell(r, 1, row[0])
        ws.cell(r, 2, row[1])
        ws.cell(r, 3, row[2])
        ws.cell(r, 4, row[3])
        ws.cell(r, 5, f'=IFERROR(INDEX(Master_Experiments!$AA:$AA,MATCH($A{r},Master_Experiments!$A:$A,0)),"Planned")')
        ws.cell(
            r,
            10,
            '=IF(AND(COUNTIFS(Decision_Log!$A:$A,"D01",Decision_Log!$M:$M,"Locked")>0,COUNTIFS(Decision_Log!$A:$A,"D02",Decision_Log!$M:$M,"Locked")>0,COUNTIFS(Decision_Log!$A:$A,"D03",Decision_Log!$M:$M,"Locked")>0,COUNTIFS(Decision_Log!$A:$A,"D04",Decision_Log!$M:$M,"Locked")>0,COUNTIFS(Decision_Log!$A:$A,"D05",Decision_Log!$M:$M,"Locked")>0,COUNTIFS(Decision_Log!$A:$A,"D06",Decision_Log!$M:$M,"Locked")>0,COUNTIFS(Decision_Log!$A:$A,"D07",Decision_Log!$M:$M,"Locked")>0),"YES","NO")',
        )
        ws.cell(r, 6, f'=IF(AND($J{r}="YES",IFERROR(INDEX(Master_Experiments!$AF:$AF,MATCH($A{r},Master_Experiments!$A:$A,0)),"INCOMPLETE")="READY"),"YES","NO")')
        ws.cell(r, 7, f'=IF(IFERROR(INDEX(Master_Experiments!$AA:$AA,MATCH($A{r},Master_Experiments!$A:$A,0)),"Planned")="Completed","YES","NO")')
        ws.cell(r, 8, row[4])
        ws.cell(r, 9, "NO")
        ws.cell(r, 11, f'=IFERROR(INDEX(Master_Experiments!$AC:$AC,MATCH($A{r},Master_Experiments!$A:$A,0)),"")')
        ws.cell(r, 12, "Planned")

    last = 1 + len(rows)
    style_body(ws, 2, last, 1, len(CONFIRMATORY_COLUMNS))
    add_table(ws, "ConfirmatoryTable", f"A1:L{last}", style="TableStyleMedium10")
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = f"A1:L{last}"
    add_list_validation(ws, "=List_Status", 5, 2, 500)
    add_list_validation(ws, "=List_Required_for_Main_Claim", 8, 2, 500)
    add_list_validation(ws, "=List_Ready_for_Chapter_4", 9, 2, 500)
    add_list_validation(ws, "=List_Status", 12, 2, 500)
    set_widths(
        ws,
        {"A": 13, "B": 34, "C": 28, "D": 34, "E": 14, "F": 18, "G": 12, "H": 20, "I": 18, "J": 22, "K": 32, "L": 20},
    )
    ws.conditional_formatting.add(f"J2:J{max(last,200)}", FormulaRule(formula=['$J2="YES"'], fill=PatternFill("solid", fgColor=COL["ok"])))
    ws.conditional_formatting.add(f"F2:F{max(last,200)}", FormulaRule(formula=['$F2="YES"'], fill=PatternFill("solid", fgColor=COL["ok"])))
    ws.conditional_formatting.add(f"G2:G{max(last,200)}", FormulaRule(formula=['$G2="YES"'], fill=PatternFill("solid", fgColor=COL["ok"])))
    ws.conditional_formatting.add(f"I2:I{max(last,200)}", FormulaRule(formula=['$I2="YES"'], fill=PatternFill("solid", fgColor=COL["ok"])))
    return last


def fill_thesis_map_sheet(ws) -> int:
    for i, h in enumerate(THESIS_MAP_COLUMNS, start=1):
        ws.cell(1, i, h)
    style_header(ws, 1, len(THESIS_MAP_COLUMNS))
    rows = [
        ("E01", "Chapter 3", "3.3 Target definition", "Target granularity method choice", "Supporting", "Method choice", "Locks target definition", "Reference Decision D01"),
        ("E02", "Chapter 3", "3.3 Task strategy", "Task pooling method choice", "Supporting", "Method choice", "Task pooling policy", ""),
        ("E03", "Chapter 3", "3.3 Modality strategy", "Modality pooling method choice", "Supporting", "Method choice", "Modality pooling policy", ""),
        ("E04", "Chapter 3", "3.4 Split design", "Split-strength stress evidence", "Supporting", "Method choice", "Justifies strict split", ""),
        ("E05", "Chapter 3", "3.4 Transfer protocol", "Cross-person framing", "Supporting", "Method choice", "Directional transfer design", ""),
        ("E06", "Chapter 3", "3.5 Model family", "Model family selection", "Supporting", "Method choice", "Final model family lock", ""),
        ("E07", "Chapter 3", "3.5 Weighting policy", "Class weighting decision", "Supporting", "Method choice", "Weighting policy lock", ""),
        ("E08", "Chapter 3", "3.5 Tuning policy", "Hyperparameter policy", "Supporting", "Method choice", "Fixed vs tuning decision", ""),
        ("E09", "Chapter 3", "3.2 Feature space", "Whole-brain vs ROI decision", "Supporting", "Method choice", "Feature representation lock", ""),
        ("E10", "Chapter 3", "3.2 Dimensionality", "Reduction strategy decision", "Supporting", "Method choice", "Complexity/stability tradeoff", ""),
        ("E11", "Chapter 3", "3.2 Scaling", "Scaling sensitivity decision", "Supporting", "Method choice", "Scaling policy lock", ""),
        ("E12", "Chapter 4", "4.3 Robustness", "Permutation support", "Supporting", "Method application", "Beyond-chance support", ""),
        ("E13", "Chapter 4", "4.3 Robustness", "Trivial baseline support", "Supporting", "Method application", "Non-trivial value support", ""),
        ("E14", "Chapter 4", "4.3 Robustness", "Interpretability stability support", "Supporting", "Method application", "Model-behavior robustness", "No localization claim"),
        ("E15", "Chapter 5", "5.2 Limitations", "Subset sensitivity discussion", "Supporting", "Discussion", "Sensitivity/limitations support", ""),
        ("E16", "Chapter 4", "4.1 Main results", "Within-person confirmatory evidence", "Main", "Method application", "Primary thesis claim", ""),
        ("E17", "Chapter 4", "4.2 Transfer results", "Cross-person transfer evidence", "Main", "Method application", "Secondary transfer claim", ""),
        ("E18", "Appendix", "A. External exploratory", "External split replication", "Supporting", "Discussion", "Exploratory external consistency", ""),
        ("E19", "Appendix", "A. External exploratory", "External model portability", "Supporting", "Discussion", "Exploratory portability", ""),
    ]
    for r, row in enumerate(rows, start=2):
        for c, val in enumerate(row, start=1):
            ws.cell(r, c, val)
    last = 1 + len(rows)
    style_body(ws, 2, last, 1, len(THESIS_MAP_COLUMNS))
    add_table(ws, "ThesisMapTable", f"A1:H{last}", style="TableStyleMedium6")
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = f"A1:H{last}"
    set_widths(ws, {"A": 13, "B": 12, "C": 24, "D": 30, "E": 14, "F": 44, "G": 34, "H": 30})
    return last


def fill_claim_ledger_sheet(ws) -> int:
    for i, h in enumerate(CLAIM_LEDGER_COLUMNS, start=1):
        ws.cell(1, i, h)
    style_header(ws, 1, len(CLAIM_LEDGER_COLUMNS))
    starter = [
        ("C01", "Primary within-person held-out-session decoding demonstrates meaningful generalization under locked pipeline.", "Confirmatory finding", "E16,E12,E13,E14", "open", "Chapter 4", "Interpret only within claim-matched split and current dataset scope.", ""),
        ("C02", "Frozen directional cross-person transfer indicates cross-case portability under domain shift.", "Confirmatory finding", "E17,E12,E13", "open", "Chapter 4", "Not a population-level generalization claim.", ""),
        ("C03", "Method lock decisions reduce leakage risk and interpretation drift.", "Method-choice", "E01,E04,E05,E06,E09,E11", "partial", "Chapter 3", "Governance claim; not a performance claim.", ""),
    ]
    for r, row in enumerate(starter, start=2):
        for c, val in enumerate(row, start=1):
            ws.cell(r, c, val)
    last = 31
    style_body(ws, 2, last, 1, len(CLAIM_LEDGER_COLUMNS))
    add_table(ws, "ClaimLedgerTable", f"A1:H{last}", style="TableStyleMedium8")
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = f"A1:H{last}"
    add_list_validation(ws, "=List_Claim_Type", 3, 2, 1000)
    add_list_validation(ws, "=List_Evidence_Status", 5, 2, 1000)
    add_list_validation(ws, "=List_Writing_Location", 6, 2, 1000)
    set_widths(ws, {"A": 12, "B": 48, "C": 20, "D": 24, "E": 14, "F": 14, "G": 38, "H": 28})
    return last


def fill_ai_usage_sheet(ws) -> int:
    for i, h in enumerate(AI_USAGE_COLUMNS, start=1):
        ws.cell(1, i, h)
    style_header(ws, 1, len(AI_USAGE_COLUMNS))
    ws.cell(2, 1, "AI001")
    ws.cell(2, 4, "GPT-5.3-Codex")
    ws.cell(2, 5, "Workbook package revision and governance alignment")
    ws.cell(2, 8, "Not reviewed")
    ws.cell(2, 9, "No")
    last = 41
    style_body(ws, 2, last, 1, len(AI_USAGE_COLUMNS))
    add_table(ws, "AIUsageTable", f"A1:J{last}", style="TableStyleMedium5")
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = f"A1:J{last}"
    add_list_validation(ws, "=List_Human_Verification_Status", 8, 2, 1000)
    add_list_validation(ws, "=List_Used_in_Thesis", 9, 2, 1000)
    set_widths(ws, {"A": 12, "B": 12, "C": 24, "D": 18, "E": 32, "F": 32, "G": 30, "H": 24, "I": 14, "J": 28})
    return last


def fill_ethics_sheet(ws) -> int:
    for i, h in enumerate(ETHICS_COLUMNS, start=1):
        ws.cell(1, i, h)
    style_header(ws, 1, len(ETHICS_COLUMNS))
    starter = [
        ("EN01", "E16", "Interpretation limits", "Risk of over-claiming confirmatory evidence beyond split/domain scope", "Enforce interpretation boundaries in Claim_Ledger and Chapter 5 limitations", "Chapter 5", "Open", ""),
        ("EN02", "D08", "AI/tool transparency", "Insufficient traceability of AI-assisted drafting decisions", "Maintain AI_Usage_Log with human verification status before thesis inclusion", "Chapter 2/Appendix", "Open", ""),
    ]
    for r, row in enumerate(starter, start=2):
        for c, val in enumerate(row, start=1):
            ws.cell(r, c, val)
    last = 31
    style_body(ws, 2, last, 1, len(ETHICS_COLUMNS))
    add_table(ws, "EthicsTable", f"A1:H{last}", style="TableStyleMedium7")
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = f"A1:H{last}"
    add_list_validation(ws, "=List_Ethics_Topic", 3, 2, 1000)
    add_list_validation(ws, "=List_Ethics_Status", 7, 2, 1000)
    set_widths(ws, {"A": 12, "B": 22, "C": 22, "D": 34, "E": 34, "F": 20, "G": 12, "H": 26})
    return last


def fill_dictionary_sheet(ws, wb: Workbook) -> None:
    ws["A1"] = "Controlled vocabularies (drives dropdown validation)"
    ws["A1"].font = Font(size=12, bold=True)
    ws["A1"].fill = PatternFill("solid", fgColor=COL["title_bg"])

    names = [
        "Category",
        "Evidential_Role",
        "Stage",
        "Priority",
        "Status",
        "Reporting_Destination",
        "Reviewed",
        "Used_in_Thesis",
        "Freeze_Status",
        "Run_Type",
        "Affects_Frozen_Pipeline",
        "Eligible_for_Method_Decision",
        "Claim_Type",
        "Evidence_Status",
        "Writing_Location",
        "Human_Verification_Status",
        "Ethics_Status",
        "Ethics_Topic",
        "Required_for_Main_Claim",
        "Ready_for_Chapter_4",
        "YesNo",
        "Subject_Scope",
        "Session_Scope",
        "Task_Scope",
        "Modality_Scope",
        "Class_Balance_Policy",
        "Split_Family",
        "Imbalance_Status",
        "Leakage_Check_Status",
        "Thesis_Use_Tag",
        "Use_Case",
        "Group_Unit",
        "Grouping_Level",
        "Transfer_Direction",
        "Leakage_Risk_Level",
        "Target_Type",
        "Execution_Section",
        "Reuse_Policy",
        "Search_Optimization_Mode",
        "Search_Parameter_Scope",
    ]
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=len(names))
    for c, name in enumerate(names, start=1):
        ws.cell(2, c, name)
        ws.cell(2, c).font = Font(color=COL["header_fg"], bold=True)
        ws.cell(2, c).fill = PatternFill("solid", fgColor=COL["header_bg"])
        items = VOCABS[name]
        for r, item in enumerate(items, start=3):
            ws.cell(r, c, item)
            ws.cell(r, c).border = THIN
            if r % 2 == 0:
                ws.cell(r, c).fill = PatternFill("solid", fgColor=COL["zebra"])
        add_named_list(wb, f"List_{name}", ws.title, c, 3, 2 + len(items))

    def_col = len(names) + 1
    term_col = def_col
    definition_col = def_col + 1
    ws.cell(1, term_col, "Term definitions")
    ws.cell(1, term_col).font = Font(size=12, bold=True)
    ws.cell(1, term_col).fill = PatternFill("solid", fgColor=COL["title_bg"])
    ws.merge_cells(start_row=1, start_column=term_col, end_row=1, end_column=definition_col)
    ws.cell(2, term_col, "Term")
    ws.cell(2, definition_col, "Definition")
    for c in (term_col, definition_col):
        ws.cell(2, c).font = Font(color=COL["header_fg"], bold=True)
        ws.cell(2, c).fill = PatternFill("solid", fgColor=COL["header_bg"])
    for r, (term, definition) in enumerate(DEFINITIONS, start=3):
        ws.cell(r, term_col, term)
        ws.cell(r, definition_col, definition)
        ws.cell(r, term_col).border = THIN
        ws.cell(r, definition_col).border = THIN
        ws.cell(r, definition_col).alignment = Alignment(wrap_text=True, vertical="top")
        if r % 2 == 0:
            ws.cell(r, term_col).fill = PatternFill("solid", fgColor=COL["zebra"])
            ws.cell(r, definition_col).fill = PatternFill("solid", fgColor=COL["zebra"])

    for c in range(1, len(names) + 1):
        ws.column_dimensions[get_column_letter(c)].width = 22
    ws.column_dimensions[get_column_letter(term_col)].width = 24
    ws.column_dimensions[get_column_letter(definition_col)].width = 84
    ws.freeze_panes = "A3"


def fill_dashboard_sheet(ws) -> None:
    ws.merge_cells("A1:N1")
    ws["A1"] = "Thesis Experiment Program Dashboard (v2)"
    ws["A1"].font = Font(size=15, bold=True)
    ws["A1"].fill = PatternFill("solid", fgColor=COL["title_bg"])

    ws["A3"] = "Program state summary"
    ws["A3"].font = Font(size=12, bold=True)
    ws["A3"].fill = PatternFill("solid", fgColor="EEF3FB")
    ws.merge_cells("A3:B3")

    labels = [
        "Total number of experiments",
        "Completed experiments",
        "Critical experiments completed",
        "Open decisions",
        "Locked decisions",
        "Target locked?",
        "Split locked?",
        "Model locked?",
        "Feature/preprocessing locked?",
        "Confirmatory eligibility achieved?",
        "Confirmatory ready for Chapter 4?",
        "Experiments mapped to Chapter 4 main results",
        "Experiments mapped only to Appendix",
        "Claims with Evidence_Status = supported",
        "Claims with Evidence_Status = partial",
        "Claims with Evidence_Status = open",
        "AI log entries",
        "Open ethics/governance notes",
        "Runs flagged with imbalance warning",
        "Runs flagged with leakage warning/fail",
        "Distinct data slices used in Run_Log",
        "Distinct grouping strategies used in Run_Log",
        "Runs with Data_Slice_ID recorded",
        "Runs with Grouping_Strategy_ID recorded",
    ]
    formulas = [
        '=COUNTA(Master_Experiments!$A$2:$A$500)',
        '=COUNTIF(Master_Experiments!$AA$2:$AA$500,"Completed")',
        '=COUNTIFS(Master_Experiments!$F$2:$F$500,"Critical",Master_Experiments!$AA$2:$AA$500,"Completed")',
        '=COUNTIF(Decision_Log!$M$2:$M$500,"Open")',
        '=COUNTIF(Decision_Log!$M$2:$M$500,"Locked")',
        '=IF(COUNTIFS(Decision_Log!$A:$A,"D01",Decision_Log!$M:$M,"Locked")>0,"YES","NO")',
        '=IF(COUNTIFS(Decision_Log!$A:$A,"D02",Decision_Log!$M:$M,"Locked")>0,"YES","NO")',
        '=IF(AND(COUNTIFS(Decision_Log!$A:$A,"D03",Decision_Log!$M:$M,"Locked")>0,COUNTIFS(Decision_Log!$A:$A,"D04",Decision_Log!$M:$M,"Locked")>0,COUNTIFS(Decision_Log!$A:$A,"D05",Decision_Log!$M:$M,"Locked")>0),"YES","NO")',
        '=IF(AND(COUNTIFS(Decision_Log!$A:$A,"D06",Decision_Log!$M:$M,"Locked")>0,COUNTIFS(Decision_Log!$A:$A,"D07",Decision_Log!$M:$M,"Locked")>0),"YES","NO")',
        '=IF(COUNTIFS(Confirmatory_Set!$J:$J,"YES")=COUNTIFS(Confirmatory_Set!$A:$A,"<>"),"YES","NO")',
        '=IF(AND(B13="YES",COUNTIFS(Confirmatory_Set!$H:$H,"YES",Confirmatory_Set!$G:$G,"YES",Confirmatory_Set!$I:$I,"YES")=COUNTIFS(Confirmatory_Set!$H:$H,"YES")),"YES","NO")',
        '=COUNTIFS(Thesis_Map!$B$2:$B$500,"Chapter 4",Thesis_Map!$E$2:$E$500,"Main")',
        '=COUNTIF(Thesis_Map!$B$2:$B$500,"Appendix")',
        '=COUNTIF(Claim_Ledger!$E$2:$E$500,"supported")',
        '=COUNTIF(Claim_Ledger!$E$2:$E$500,"partial")',
        '=COUNTIF(Claim_Ledger!$E$2:$E$500,"open")',
        '=COUNTA(AI_Usage_Log!$A$2:$A$500)',
        '=COUNTIF(Ethics_Governance_Notes!$G$2:$G$500,"Open")',
        '=COUNTIFS(Run_Log!$Z$2:$Z$2000,"mild_imbalance")+COUNTIFS(Run_Log!$Z$2:$Z$2000,"severe_imbalance")',
        '=COUNTIFS(Run_Log!$AA$2:$AA$2000,"warning")+COUNTIFS(Run_Log!$AA$2:$AA$2000,"failed")',
        '=SUMPRODUCT((Run_Log!$F$2:$F$2000<>"")/COUNTIF(Run_Log!$F$2:$F$2000,Run_Log!$F$2:$F$2000&""))',
        '=SUMPRODUCT((Run_Log!$G$2:$G$2000<>"")/COUNTIF(Run_Log!$G$2:$G$2000,Run_Log!$G$2:$G$2000&""))',
        '=COUNTIF(Run_Log!$F$2:$F$2000,"<>")',
        '=COUNTIF(Run_Log!$G$2:$G$2000,"<>")',
    ]

    for i, (label, formula) in enumerate(zip(labels, formulas, strict=True), start=4):
        ws.cell(i, 1, label)
        ws.cell(i, 2, formula)
        ws.cell(i, 1).border = THIN
        ws.cell(i, 2).border = THIN
        if i % 2 == 0:
            ws.cell(i, 1).fill = PatternFill("solid", fgColor=COL["zebra"])
            ws.cell(i, 2).fill = PatternFill("solid", fgColor=COL["zebra"])

    ws["D3"] = "Experiments by category"
    ws["D3"].font = Font(size=12, bold=True)
    ws["D3"].fill = PatternFill("solid", fgColor="EEF3FB")
    ws.merge_cells("D3:E3")
    for i, _ in enumerate(VOCABS["Category"], start=4):
        ws.cell(i, 4, f"=Dictionary_Validation!A{i-1}")
        ws.cell(i, 5, f"=COUNTIF(Master_Experiments!$C:$C,D{i})")
        ws.cell(i, 4).border = THIN
        ws.cell(i, 5).border = THIN

    ws["G3"] = "Experiments by evidential role"
    ws["G3"].font = Font(size=12, bold=True)
    ws["G3"].fill = PatternFill("solid", fgColor="EEF3FB")
    ws.merge_cells("G3:H3")
    for i, _ in enumerate(VOCABS["Evidential_Role"], start=4):
        ws.cell(i, 7, f"=Dictionary_Validation!B{i-1}")
        ws.cell(i, 8, f"=COUNTIF(Master_Experiments!$D:$D,G{i})")
        ws.cell(i, 7).border = THIN
        ws.cell(i, 8).border = THIN
        ws.cell(i, 7).alignment = Alignment(wrap_text=True)

    ws["J3"] = "Experiments by data slice"
    ws["J3"].font = Font(size=12, bold=True)
    ws["J3"].fill = PatternFill("solid", fgColor="EEF3FB")
    ws.merge_cells("J3:K3")
    for i in range(4, 16):
        src_row = i - 2
        ws.cell(i, 10, f'=IFERROR(Data_Selection_Design!$A{src_row},"")')
        ws.cell(
            i,
            11,
            f'=IF(J{i}="","",SUMPRODUCT((Run_Log!$F$2:$F$2000=J{i})*(Run_Log!$B$2:$B$2000<>"")/'
            f'IFERROR(COUNTIFS(Run_Log!$F$2:$F$2000,J{i},Run_Log!$B$2:$B$2000,Run_Log!$B$2:$B$2000),1)))',
        )
        ws.cell(i, 10).border = THIN
        ws.cell(i, 11).border = THIN

    ws["M3"] = "Experiments by grouping strategy"
    ws["M3"].font = Font(size=12, bold=True)
    ws["M3"].fill = PatternFill("solid", fgColor="EEF3FB")
    ws.merge_cells("M3:N3")
    for i in range(4, 16):
        src_row = i - 2
        ws.cell(i, 13, f'=IFERROR(Grouping_Strategy_Map!$A{src_row},"")')
        ws.cell(
            i,
            14,
            f'=IF(M{i}="","",SUMPRODUCT((Run_Log!$G$2:$G$2000=M{i})*(Run_Log!$B$2:$B$2000<>"")/'
            f'IFERROR(COUNTIFS(Run_Log!$G$2:$G$2000,M{i},Run_Log!$B$2:$B$2000,Run_Log!$B$2:$B$2000),1)))',
        )
        ws.cell(i, 13).border = THIN
        ws.cell(i, 14).border = THIN

    ws["D17"] = "Quick comparison counts (run-level)"
    ws["D17"].font = Font(size=12, bold=True)
    ws["D17"].fill = PatternFill("solid", fgColor="EEF3FB")
    ws.merge_cells("D17:E17")
    comp_labels = [
        "Pooled task runs",
        "Task-specific runs",
        "Pooled modality runs",
        "Modality-specific runs",
        "Within-person runs",
        "Cross-person runs",
        "Temporal split runs",
    ]
    comp_formulas = [
        '=COUNTIF(Run_Log!$Q$2:$Q$2000,"pooled_all_tasks")',
        '=COUNTIFS(Run_Log!$Q$2:$Q$2000,"<>",Run_Log!$Q$2:$Q$2000,"<>pooled_all_tasks")',
        '=COUNTIF(Run_Log!$R$2:$R$2000,"pooled_all_modalities")',
        '=COUNTIFS(Run_Log!$R$2:$R$2000,"<>",Run_Log!$R$2:$R$2000,"<>pooled_all_modalities")',
        '=SUMPRODUCT((Grouping_Strategy_Map!$C$2:$C$200="within_subject")*COUNTIF(Run_Log!$G$2:$G$2000,Grouping_Strategy_Map!$A$2:$A$200))',
        '=SUMPRODUCT((Grouping_Strategy_Map!$C$2:$C$200="cross_person")*COUNTIF(Run_Log!$G$2:$G$2000,Grouping_Strategy_Map!$A$2:$A$200))',
        '=SUMPRODUCT((Grouping_Strategy_Map!$C$2:$C$200="temporal")*COUNTIF(Run_Log!$G$2:$G$2000,Grouping_Strategy_Map!$A$2:$A$200))',
    ]
    for i, (label, formula) in enumerate(zip(comp_labels, comp_formulas, strict=True), start=18):
        ws.cell(i, 4, label)
        ws.cell(i, 5, formula)
        ws.cell(i, 4).border = THIN
        ws.cell(i, 5).border = THIN
        if i % 2 == 0:
            ws.cell(i, 4).fill = PatternFill("solid", fgColor=COL["zebra"])
            ws.cell(i, 5).fill = PatternFill("solid", fgColor=COL["zebra"])

    set_widths(
        ws,
        {
            "A": 46,
            "B": 16,
            "C": 3,
            "D": 30,
            "E": 12,
            "F": 3,
            "G": 36,
            "H": 12,
            "I": 3,
            "J": 24,
            "K": 12,
            "L": 3,
            "M": 28,
            "N": 12,
        },
    )
    ws.freeze_panes = "A4"
    ws.conditional_formatting.add("B9:B14", FormulaRule(formula=['B9="YES"'], fill=PatternFill("solid", fgColor=COL["ok"])))
    ws.conditional_formatting.add("B9:B14", FormulaRule(formula=['B9="NO"'], fill=PatternFill("solid", fgColor=COL["bad"])))


def build_workbook() -> Workbook:
    wb = Workbook()
    wb.remove(wb.active)
    for name in SHEET_ORDER:
        wb.create_sheet(name)

    fill_readme_sheet(wb["README"])
    fill_master_sheet(wb["Master_Experiments"])
    add_dynamic_named_list(wb, "List_Experiment_ID", "Master_Experiments", col_idx(MASTER_COLUMNS, "Experiment_ID"), 2)
    fill_experiment_definitions_sheet(wb["Experiment_Definitions"])
    fill_search_spaces_sheet(wb["Search_Spaces"], wb)
    fill_artifact_registry_sheet(wb["Artifact_Registry"])
    fill_fixed_configs_sheet(wb["Fixed_Configs"])
    fill_objectives_sheet(wb["Objectives"])
    fill_machine_status_sheet(wb["Machine_Status"])
    fill_trial_results_sheet(wb["Trial_Results"])
    fill_summary_outputs_sheet(wb["Summary_Outputs"])
    fill_data_selection_design_sheet(wb["Data_Selection_Design"], wb)
    fill_grouping_strategy_map_sheet(wb["Grouping_Strategy_Map"], wb)
    fill_data_profile_sheet(wb["Data_Profile"])
    fill_run_log_sheet(wb["Run_Log"])
    fill_decision_log_sheet(wb["Decision_Log"])
    fill_confirmatory_sheet(wb["Confirmatory_Set"])
    fill_thesis_map_sheet(wb["Thesis_Map"])
    fill_dictionary_sheet(wb["Dictionary_Validation"], wb)
    fill_dashboard_sheet(wb["Dashboard"])
    fill_claim_ledger_sheet(wb["Claim_Ledger"])
    fill_ai_usage_sheet(wb["AI_Usage_Log"])
    fill_ethics_sheet(wb["Ethics_Governance_Notes"])

    wb.calculation.fullCalcOnLoad = True
    write_schema_metadata(wb["README"])
    generator_row = WORKBOOK_SCHEMA_METADATA_START_ROW + 7
    wb["README"][f"A{generator_row}"] = "Generated by create_thesis_experiment_workbook.py (v2)"
    wb["README"][f"A{generator_row}"].font = Font(italic=True, color="4B5563")
    return wb


def validate(path: Path) -> dict[str, str]:
    wb = load_workbook(path)
    sheets = [ws.title for ws in wb.worksheets]
    sheet_ok = sheets == SHEET_ORDER
    missing_sheets = [s for s in SHEET_ORDER if s not in sheets]
    legacy_required = [
        "README",
        "Master_Experiments",
        "Experiment_Definitions",
        "Search_Spaces",
        "Artifact_Registry",
        "Fixed_Configs",
        "Objectives",
        "Machine_Status",
        "Trial_Results",
        "Summary_Outputs",
        "Run_Log",
        "Decision_Log",
        "Confirmatory_Set",
        "Thesis_Map",
        "Dictionary_Validation",
        "Dashboard",
        "Claim_Ledger",
        "AI_Usage_Log",
        "Ethics_Governance_Notes",
    ]
    legacy_sheets_present = all(s in sheets for s in legacy_required)
    new_sheets = [
        "Experiment_Definitions",
        "Search_Spaces",
        "Artifact_Registry",
        "Fixed_Configs",
        "Objectives",
        "Machine_Status",
        "Trial_Results",
        "Summary_Outputs",
        "Data_Selection_Design",
        "Grouping_Strategy_Map",
        "Data_Profile",
    ]
    new_sheets_present = all(s in sheets for s in new_sheets)

    master = wb["Master_Experiments"]
    experiment_definitions = wb["Experiment_Definitions"]
    readme = wb["README"]
    confirm = wb["Confirmatory_Set"]
    dash = wb["Dashboard"]
    dictionary = wb["Dictionary_Validation"]
    run_log = wb["Run_Log"]

    dv_count = sum(
        len(wb[name].data_validations.dataValidation)
        for name in [
            "Master_Experiments",
            "Experiment_Definitions",
            "Search_Spaces",
            "Data_Selection_Design",
            "Grouping_Strategy_Map",
            "Run_Log",
            "Decision_Log",
            "Confirmatory_Set",
            "Claim_Ledger",
            "AI_Usage_Log",
            "Ethics_Governance_Notes",
        ]
    )
    stage_values = {master[f"E{r}"].value for r in range(2, master.max_row + 1)}
    stage_vocab = set(STAGE_V2)
    stage_consistent = stage_values.issubset(stage_vocab)
    run_log_headers = [run_log.cell(1, c).value for c in range(1, len(RUN_LOG_COLUMNS) + 1)]
    run_log_new_cols = [
        "Data_Slice_ID",
        "Grouping_Strategy_ID",
        "Train_Group_Rule",
        "Test_Group_Rule",
        "Transfer_Direction",
        "Sample_Count",
        "Class_Counts",
        "Imbalance_Status",
        "Leakage_Check_Status",
        "Session_Coverage",
        "Task_Coverage",
        "Modality_Coverage",
    ]
    run_log_columns_ok = all(col in run_log_headers for col in run_log_new_cols)
    experiment_definitions_headers = [
        experiment_definitions.cell(1, c).value
        for c in range(1, len(EXPERIMENT_DEFINITIONS_COLUMNS) + 1)
    ]
    experiment_definitions_columns_ok = (
        experiment_definitions_headers == EXPERIMENT_DEFINITIONS_COLUMNS
    )

    required_named_lists = [
        "List_Experiment_ID",
        "List_Data_Slice_ID",
        "List_Grouping_Strategy_ID",
        "List_Subject_Scope",
        "List_Session_Scope",
        "List_Task_Scope",
        "List_Modality_Scope",
        "List_Class_Balance_Policy",
        "List_Split_Family",
        "List_Imbalance_Status",
        "List_Leakage_Check_Status",
        "List_Transfer_Direction",
        "List_Execution_Section",
        "List_Reuse_Policy",
        "List_Search_Optimization_Mode",
        "List_Search_Parameter_Scope",
        "List_Search_Space_ID",
    ]
    defined_names = {name for name in wb.defined_names.keys()}
    named_lists_ok = all(name in defined_names for name in required_named_lists)
    missing_named_lists = [name for name in required_named_lists if name not in defined_names]

    confirmatory_formulas_ok = all(
        isinstance(confirm[cell].value, str) and confirm[cell].value.startswith("=")
        for cell in ["F2", "G2", "J2"]
    ) and isinstance(master["AF2"].value, str) and master["AF2"].value.startswith("=")
    dashboard_core_formulas_ok = all(
        isinstance(dash[cell].value, str) and dash[cell].value.startswith("=")
        for cell in ["B13", "B14", "B22", "B23", "B24", "B25", "K4", "N4", "E18"]
    )
    schema_metadata = read_schema_metadata(readme)
    required_schema_metadata = expected_schema_metadata()
    schema_metadata_keys_present = all(
        key in schema_metadata for key in required_schema_metadata
    )
    workbook_schema_version_value = schema_metadata.get(
        "workbook_schema_version", WORKBOOK_SCHEMA_VERSION
    )
    workbook_schema_supported = (
        workbook_schema_version_value in SUPPORTED_WORKBOOK_SCHEMA_VERSIONS
    )

    return {
        "sheet_order_ok": str(sheet_ok),
        "missing_sheets": ", ".join(missing_sheets) if missing_sheets else "None",
        "legacy_sheets_present": str(legacy_sheets_present),
        "new_sheets_present": str(new_sheets_present),
        "sheet_count": str(len(sheets)),
        "data_validations_found": str(dv_count),
        "experiment_definitions_columns_ok": str(experiment_definitions_columns_ok),
        "run_log_new_columns_present": str(run_log_columns_ok),
        "required_named_lists_present": str(named_lists_ok),
        "missing_named_lists": ", ".join(missing_named_lists) if missing_named_lists else "None",
        "experiment_ready_formula_present": str(isinstance(master["AF2"].value, str) and master["AF2"].value.startswith("=")),
        "confirmatory_formula_present": str(confirmatory_formulas_ok),
        "dashboard_formula_present": str(dashboard_core_formulas_ok),
        "stage_vocab_consistent": str(stage_consistent),
        "stage_vocab_rows": str(len([v for v in stage_values if v is not None])),
        "dictionary_stage_head": str(dictionary["C3"].value),
        "schema_metadata_keys_present": str(schema_metadata_keys_present),
        "workbook_schema_version": workbook_schema_version_value,
        "workbook_schema_supported": str(workbook_schema_supported),
    }


def main() -> None:
    wb = build_workbook()
    wb.save(OUT_XLSX)
    summary = validate(OUT_XLSX)
    print("Created workbook:", OUT_XLSX.resolve())
    print("Sheet order valid:", summary["sheet_order_ok"])
    print("Missing required sheets:", summary["missing_sheets"])
    print("Legacy required sheets present:", summary["legacy_sheets_present"])
    print("New sheets present:", summary["new_sheets_present"])
    print("Sheet count:", summary["sheet_count"])
    print("Data validations found:", summary["data_validations_found"])
    print("Experiment_Definitions columns valid:", summary["experiment_definitions_columns_ok"])
    print("Run_Log new columns present:", summary["run_log_new_columns_present"])
    print("Required named lists present:", summary["required_named_lists_present"])
    print("Missing named lists:", summary["missing_named_lists"])
    print("Experiment_Ready formula present:", summary["experiment_ready_formula_present"])
    print("Confirmatory formulas present:", summary["confirmatory_formula_present"])
    print("Dashboard formulas present:", summary["dashboard_formula_present"])
    print("Stage vocabulary consistent:", summary["stage_vocab_consistent"])
    print("Stage rows detected:", summary["stage_vocab_rows"])
    print("Schema metadata keys present:", summary["schema_metadata_keys_present"])
    print("Workbook schema version:", summary["workbook_schema_version"])
    print("Workbook schema supported:", summary["workbook_schema_supported"])


if __name__ == "__main__":
    main()
