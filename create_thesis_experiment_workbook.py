from __future__ import annotations

from pathlib import Path

from openpyxl import Workbook, load_workbook
from openpyxl.formatting.rule import FormulaRule
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.workbook.defined_name import DefinedName
from openpyxl.worksheet.datavalidation import DataValidation
from openpyxl.worksheet.table import Table, TableStyleInfo


OUT_XLSX = Path("thesis_experiment_program.xlsx")

SHEET_ORDER = [
    "README",
    "Master_Experiments",
    "Run_Log",
    "Decision_Log",
    "Confirmatory_Set",
    "Thesis_Map",
    "Dictionary_Validation",
    "Dashboard",
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
    "Code_Commit_or_Version",
    "Config_File_or_Path",
    "Random_Seed",
    "Target",
    "Split_ID_or_Fold_Definition",
    "Model",
    "Feature_Set",
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

DECISION_COLUMNS = [
    "Decision_ID",
    "Decision_Topic",
    "Stage",
    "Candidate_Options",
    "Evidence_Experiments",
    "Chosen_Option",
    "Why_Chosen",
    "Why_Not_Others",
    "Risks_or_Tradeoffs",
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
    "Main_Result_Summary",
    "Robustness_Status",
    "Ready_for_Chapter_4",
    "Confirmatory_Eligible",
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
    "Stage": [
        "Stage 0 - Program design",
        "Stage 1 - Method choice",
        "Stage 2 - Robustness support",
        "Stage 3 - Confirmatory core",
        "Stage 4 - External exploratory",
        "Stage 5 - Thesis write-up",
    ],
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
}

DEFINITIONS = [
    ("confirmatory", "Primary evidence analysis with pre-specified pipeline and decision rules."),
    ("decision-support", "Method-choice evidence used before confirmatory lock."),
    ("exploratory", "Hypothesis-generating or extension analysis outside confirmatory core."),
    ("construct validity", "How well operational labels/features represent the intended construct."),
    ("internal validity", "Protection against confounds and leakage in experiment design."),
    ("statistical conclusion validity", "Reliability of inference from metrics and comparisons."),
    ("external validity", "How far findings transfer to other settings/data."),
    ("leakage", "Information contamination across train-test boundaries that inflates performance."),
    ("domain shift", "Distribution mismatch between training and testing domains."),
    ("interpretability stability", "Consistency of model explanation signals across folds."),
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
    zebra_fill = PatternFill("solid", fgColor=COL["zebra"])
    for r in range(r1, r2 + 1):
        for c in range(c1, c2 + 1):
            cell = ws.cell(r, c)
            cell.alignment = Alignment(vertical="top", wrap_text=True)
            cell.border = THIN
            if zebra and r % 2 == 0:
                cell.fill = zebra_fill


def set_widths(ws, widths: dict[str, float]) -> None:
    for col, w in widths.items():
        ws.column_dimensions[col].width = w


def add_table(ws, name: str, ref: str, style: str) -> None:
    t = Table(displayName=name, ref=ref)
    t.tableStyleInfo = TableStyleInfo(
        name=style,
        showFirstColumn=False,
        showLastColumn=False,
        showRowStripes=True,
        showColumnStripes=False,
    )
    ws.add_table(t)


def add_list_validation(ws, formula: str, col: int, r1: int, r2: int, allow_blank: bool = True) -> None:
    dv = DataValidation(type="list", formula1=formula, allow_blank=allow_blank)
    ws.add_data_validation(dv)
    letter = get_column_letter(col)
    dv.add(f"{letter}{r1}:{letter}{r2}")


def add_named_list(wb: Workbook, list_name: str, sheet_name: str, col: int, start: int, end: int) -> None:
    letter = get_column_letter(col)
    rng = f"'{sheet_name}'!${letter}${start}:${letter}${end}"
    wb.defined_names.add(DefinedName(name=list_name, attr_text=rng))


def build_experiments() -> list[dict[str, str]]:
    base = {
        "Stage": "Stage 1 - Method choice",
        "Priority": "Medium",
        "Decision_Supported": "Method-choice decision support.",
        "Hypothesis_or_Expectation": "Evaluate under leakage-aware splits before interpretation.",
        "Held_Constant_Controls": "Data index version, preprocessing order, metric definitions, seed policy.",
        "Dataset_Scope": "Internal repeated-session BAS2 dataset (sub-001, sub-002) unless explicitly external.",
        "Target_Definition": "coarse_affect unless this experiment manipulates target definition.",
        "Split_Logic": "Claim-matched leakage-safe split (within_subject_loso_session unless manipulated).",
        "Leakage_Risk": "Performance inflation if split logic or fit scope leaks train-test information.",
        "Leakage_Control": "Use explicit split manifests and train-only fitting; frozen transfer for cross-person runs.",
        "Models_Compared": "ridge baseline unless model family is manipulated.",
        "Primary_Metric": "Balanced accuracy",
        "Secondary_Metrics": "Macro-F1; class-wise recall; accuracy",
        "Robustness_Checks": "Fold-level stability checks and leakage stress checks where relevant.",
        "Decision_Criterion": "Pre-specified criterion set before interpretation.",
        "Success_Pattern": "Stable claim-matched behavior with no leakage indicators.",
        "Failure_or_Warning_Pattern": "Weak-split inflation, instability, or minority-class collapse.",
        "Threats_to_Validity": "Small sample size, class imbalance, domain shift, and model sensitivity.",
        "Reporting_Destination": "Chapter 3 Method choice",
        "Interpretation_Boundary": "Interpret only within the specified split logic and evidence tier.",
        "Status": "Planned",
        "Owner": "Khaled (thesis author)",
        "Outcome_Summary": "",
        "Decision_Taken": "",
        "Notes": "",
    }
    rows = [
        {"Experiment_ID": "E01", "Short_Title": "Target granularity experiment", "Category": "Target-definition", "Evidential_Role": "Secondary decision-support", "Priority": "High", "Decision_Supported": "Final target lock", "Exact_Question": "Should the thesis use fine-grained emotion labels, three-class coarse affect, or binary valence-like labels?", "Hypothesis_or_Expectation": "coarse_affect likely balances construct validity, class balance, and learnability.", "Manipulated_Factor": "Target definition", "Target_Definition": "fine emotion vs coarse_affect vs binary valence-like", "Split_Logic": "within_subject_loso_session (primary) with frozen transfer as secondary stress context.", "Models_Compared": "ridge fixed baseline.", "Decision_Criterion": "Prefer target balancing construct validity, class balance, sample sufficiency, and learnability under claim-matched evaluation."},
        {"Experiment_ID": "E02", "Short_Title": "Task pooling experiment", "Category": "Target-definition", "Evidential_Role": "Secondary decision-support", "Decision_Supported": "Task pooling policy", "Exact_Question": "Should the main target be learned across all tasks together, or separately by task?", "Manipulated_Factor": "Pooling strategy by task"},
        {"Experiment_ID": "E03", "Short_Title": "Modality pooling experiment", "Category": "Target-definition", "Evidential_Role": "Secondary decision-support", "Decision_Supported": "Modality pooling policy", "Exact_Question": "Should audio, video, and audiovisual conditions be pooled or modeled separately?", "Manipulated_Factor": "Pooling strategy by modality"},
        {"Experiment_ID": "E04", "Short_Title": "Split-strength stress test", "Category": "Split-design", "Evidential_Role": "Secondary decision-support", "Priority": "High", "Decision_Supported": "Primary split policy", "Exact_Question": "How much do conclusions change under weaker split strategies compared with session-held-out evaluation?", "Manipulated_Factor": "Split logic", "Split_Logic": "record-wise random split vs weaker grouped split vs within-person leave-one-session-out", "Leakage_Risk": "High under weak splits.", "Decision_Criterion": "Retain strictest split matching claim; weaker splits only as inflation demonstrations."},
        {"Experiment_ID": "E05", "Short_Title": "Cross-person transfer design", "Category": "Split-design", "Evidential_Role": "Secondary decision-support", "Priority": "High", "Decision_Supported": "Cross-person framing", "Exact_Question": "How should cross-person generalization be framed and evaluated?", "Manipulated_Factor": "Transfer protocol", "Split_Logic": "frozen train-on-A/test-on-B and reverse direction", "Leakage_Control": "Fit once on train subject only; no re-fit/re-tune on test subject.", "Decision_Criterion": "Keep frozen directional transfer as clean cross-case test.", "Interpretation_Boundary": "Transfer evidence under domain shift, not population generalization."},
        {"Experiment_ID": "E06", "Short_Title": "Linear model family comparison", "Category": "Model-comparison", "Evidential_Role": "Secondary decision-support", "Priority": "High", "Decision_Supported": "Final model family lock", "Exact_Question": "Which linear model is most appropriate: ridge, logistic regression, or linear SVM?", "Manipulated_Factor": "Model family", "Models_Compared": "ridge vs logreg vs linearsvc", "Decision_Criterion": "Prefer simplest model with comparable performance and better robustness/stability."},
        {"Experiment_ID": "E07", "Short_Title": "Class weighting experiment", "Category": "Model-comparison", "Evidential_Role": "Secondary decision-support", "Decision_Supported": "Weighting policy", "Exact_Question": "Does class weighting improve fairness across classes without instability?", "Manipulated_Factor": "Weighting strategy"},
        {"Experiment_ID": "E08", "Short_Title": "Hyperparameter strategy experiment", "Category": "Model-comparison", "Evidential_Role": "Secondary decision-support", "Decision_Supported": "Tuning policy", "Exact_Question": "Are fixed conservative hyperparameters sufficient, or does light nested tuning materially improve results?", "Manipulated_Factor": "Tuning strategy", "Decision_Criterion": "Keep fixed settings unless nested tuning gives clear, stable gain."},
        {"Experiment_ID": "E09", "Short_Title": "Whole-brain versus ROI experiment", "Category": "Preprocessing-feature", "Evidential_Role": "Secondary decision-support", "Decision_Supported": "Feature representation lock", "Exact_Question": "Should primary representation remain whole-brain masked voxels or use theory-justified ROIs?", "Manipulated_Factor": "Feature space"},
        {"Experiment_ID": "E10", "Short_Title": "Dimensionality reduction experiment", "Category": "Preprocessing-feature", "Evidential_Role": "Secondary decision-support", "Decision_Supported": "Feature reduction policy", "Exact_Question": "Does unsupervised dimensionality reduction improve performance or stability enough to justify complexity?", "Manipulated_Factor": "Feature reduction strategy"},
        {"Experiment_ID": "E11", "Short_Title": "Scaling and normalization experiment", "Category": "Preprocessing-feature", "Evidential_Role": "Secondary decision-support", "Decision_Supported": "Scaling policy lock", "Exact_Question": "How sensitive are results to scaling strategy?", "Manipulated_Factor": "Scaling policy"},
        {"Experiment_ID": "E12", "Short_Title": "Permutation test experiment", "Category": "Robustness", "Evidential_Role": "Primary-supporting robustness", "Stage": "Stage 2 - Robustness support", "Priority": "High", "Decision_Supported": "Chance-level distinguishability", "Exact_Question": "Are main predictive results distinguishable from chance under exact same split logic?", "Manipulated_Factor": "Label structure (true vs permuted)", "Split_Logic": "Exact confirmatory split logic with train-only permutation.", "Secondary_Metrics": "Empirical p-value; null distribution summary", "Reporting_Destination": "Chapter 4 Supporting robustness"},
        {"Experiment_ID": "E13", "Short_Title": "Trivial baseline experiment", "Category": "Robustness", "Evidential_Role": "Primary-supporting robustness", "Stage": "Stage 2 - Robustness support", "Priority": "High", "Decision_Supported": "Non-trivial signal check", "Exact_Question": "Does the final model clearly outperform trivial baselines?", "Manipulated_Factor": "Baseline type", "Models_Compared": "Final locked model vs trivial baselines", "Reporting_Destination": "Chapter 4 Supporting robustness"},
        {"Experiment_ID": "E14", "Short_Title": "Stability of explanation experiment", "Category": "Robustness", "Evidential_Role": "Primary-supporting robustness", "Stage": "Stage 2 - Robustness support", "Priority": "High", "Decision_Supported": "Interpretability stability support", "Exact_Question": "Are linear coefficients or importance patterns stable across held-out-session folds?", "Manipulated_Factor": "Fold identity", "Primary_Metric": "Mean pairwise coefficient correlation", "Secondary_Metrics": "Sign consistency; top-k overlap", "Reporting_Destination": "Chapter 4 Supporting robustness", "Interpretation_Boundary": "Model-behavior evidence only; no localization claims."},
        {"Experiment_ID": "E15", "Short_Title": "Sensitivity to subset exclusion", "Category": "Robustness", "Evidential_Role": "Secondary decision-support / robustness", "Stage": "Stage 2 - Robustness support", "Decision_Supported": "Sensitivity framing", "Exact_Question": "Do main conclusions depend excessively on one subset such as one task or modality?", "Manipulated_Factor": "Subset inclusion", "Reporting_Destination": "Chapter 5 Discussion"},
        {"Experiment_ID": "E16", "Short_Title": "Final within-person confirmatory analysis", "Category": "Confirmatory core", "Evidential_Role": "Primary confirmatory", "Stage": "Stage 3 - Confirmatory core", "Priority": "Critical", "Decision_Supported": "Primary evidential core", "Exact_Question": "Does the final locked pipeline show meaningful within-person held-out-session generalization?", "Manipulated_Factor": "None; final frozen pipeline", "Target_Definition": "Locked target after Stage 1 target lock", "Split_Logic": "within_subject_loso_session", "Models_Compared": "Final locked model", "Decision_Criterion": "Primary evidential core of thesis.", "Reporting_Destination": "Chapter 4 Main results"},
        {"Experiment_ID": "E17", "Short_Title": "Final cross-person transfer analysis", "Category": "Confirmatory core", "Evidential_Role": "Primary confirmatory", "Stage": "Stage 3 - Confirmatory core", "Priority": "Critical", "Decision_Supported": "Secondary confirmatory transfer evidence", "Exact_Question": "Does final locked pipeline transfer meaningfully across individuals under frozen directional transfer?", "Manipulated_Factor": "None; final frozen pipeline", "Target_Definition": "Locked target after Stage 1 target lock", "Split_Logic": "frozen_cross_person_transfer (both directions)", "Models_Compared": "Final locked model", "Decision_Criterion": "Interpret as cross-case transfer evidence under domain shift.", "Reporting_Destination": "Chapter 4 Main results", "Interpretation_Boundary": "Not population-level generalization."},
        {"Experiment_ID": "E18", "Short_Title": "External open-source split replication", "Category": "External exploratory", "Evidential_Role": "Exploratory extension", "Stage": "Stage 4 - External exploratory", "Priority": "Low", "Decision_Supported": "External split-consistency check", "Exact_Question": "Does same leakage-aware split logic materially affect conclusions on compatible external open-source data?", "Manipulated_Factor": "Dataset source", "Dataset_Scope": "External open-source dataset if available/approved", "Reporting_Destination": "Appendix"},
        {"Experiment_ID": "E19", "Short_Title": "External open-source model portability", "Category": "External exploratory", "Evidential_Role": "Exploratory extension", "Stage": "Stage 4 - External exploratory", "Priority": "Low", "Decision_Supported": "External model-ranking portability check", "Exact_Question": "Does relative ranking of reasonable model choices remain similar on external data?", "Manipulated_Factor": "Dataset source with same small model set", "Dataset_Scope": "External open-source dataset if available/approved", "Models_Compared": "ridge vs logreg vs linearsvc", "Reporting_Destination": "Appendix"},
    ]
    out: list[dict[str, str]] = []
    for r in rows:
        m = dict(base)
        m.update(r)
        out.append(m)
    return out


def fill_readme(ws) -> None:
    ws.merge_cells("A1:H1")
    ws["A1"] = "Thesis Experiment Program Workbook"
    ws["A1"].font = Font(size=16, bold=True)
    ws["A1"].fill = PatternFill("solid", fgColor=COL["title_bg"])
    ws["A1"].alignment = Alignment(horizontal="left")
    blocks = [
        ("Purpose", "Scientific control workbook for pre-interpretation experiment specification, traceable decision locking, leakage-aware execution, and thesis reporting alignment."),
        ("Rules of use", "One exact question per experiment. One manipulated factor unless justified. Define split logic and leakage controls before running. No interpretation before criteria are specified."),
        ("Evidential tiers", "Primary confirmatory; Primary-supporting robustness; Secondary decision-support; Exploratory extension."),
        ("Freeze policy", "Confirmatory claims require locked decisions (target, split, model, weighting, tuning, feature representation, scaling) in Decision_Log."),
        ("Color legend", "Green: confirmatory/locked/YES. Orange: exploratory/open. Yellow: critical priority. Gray: dropped. Red: missing required setup."),
        ("Workflow", "Master_Experiments -> Run_Log -> Decision_Log -> Confirmatory_Set -> Thesis_Map -> Dashboard."),
    ]
    row = 3
    for title, text in blocks:
        ws[f"A{row}"] = title
        ws[f"A{row}"].font = Font(bold=True)
        ws[f"A{row}"].fill = PatternFill("solid", fgColor="EEF3FB")
        ws.merge_cells(start_row=row, start_column=2, end_row=row + 1, end_column=8)
        ws.cell(row=row, column=2, value=text).alignment = Alignment(wrap_text=True, vertical="top")
        for rr in (row, row + 1):
            for cc in range(1, 9):
                ws.cell(rr, cc).border = THIN
        row += 3
    ws.merge_cells("A22:H23")
    ws["A22"] = "Use this workbook as the single source of truth for thesis experiment planning, locking, and chapter mapping."
    ws["A22"].fill = PatternFill("solid", fgColor="FFF8E1")
    ws["A22"].alignment = Alignment(wrap_text=True, vertical="top")
    set_widths(ws, {c: 26 for c in "ABCDEFGH"})
    ws.column_dimensions["A"].width = 20
    ws.freeze_panes = "A2"


def fill_master(ws) -> int:
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
            f'=IF($AA{r}="Dropped","N/A",IF(AND($H{r}<>"",$J{r}<>"",$K{r}<>"",$M{r}<>"",$N{r}<>"",$P{r}<>"",$Q{r}<>"",$Y{r}<>""),"READY","INCOMPLETE"))',
        )
    last = 1 + len(rows)
    style_body(ws, 2, last, 1, len(MASTER_COLUMNS))
    ws.auto_filter.ref = f"A1:{get_column_letter(len(MASTER_COLUMNS))}{last}"
    ws.freeze_panes = "A2"
    set_widths(
        ws,
        {
            "A": 13, "B": 30, "C": 20, "D": 32, "E": 22, "F": 10, "G": 24, "H": 46, "I": 34, "J": 24,
            "K": 40, "L": 28, "M": 28, "N": 34, "O": 24, "P": 36, "Q": 24, "R": 18, "S": 24, "T": 36,
            "U": 34, "V": 34, "W": 34, "X": 34, "Y": 30, "Z": 36, "AA": 14, "AB": 18, "AC": 32,
            "AD": 28, "AE": 32, "AF": 16,
        },
    )
    add_list_validation(ws, "=List_Category", 3, 2, 400, False)
    add_list_validation(ws, "=List_Evidential_Role", 4, 2, 400, False)
    add_list_validation(ws, "=List_Stage", 5, 2, 400, False)
    add_list_validation(ws, "=List_Priority", 6, 2, 400, False)
    add_list_validation(ws, "=List_Reporting_Destination", 25, 2, 400, False)
    add_list_validation(ws, "=List_Status", 27, 2, 400, False)
    rng = f"A2:AF{max(last, 200)}"
    ws.conditional_formatting.add(rng, FormulaRule(formula=['$D2="Primary confirmatory"'], fill=PatternFill("solid", fgColor=COL["confirmatory"])))
    ws.conditional_formatting.add(rng, FormulaRule(formula=['ISNUMBER(SEARCH("Exploratory",$D2))'], fill=PatternFill("solid", fgColor=COL["exploratory"])))
    ws.conditional_formatting.add(rng, FormulaRule(formula=['$F2="Critical"'], fill=PatternFill("solid", fgColor=COL["critical"])))
    ws.conditional_formatting.add(rng, FormulaRule(formula=['$AA2="Dropped"'], fill=PatternFill("solid", fgColor=COL["dropped"]), font=Font(italic=True, color="6E6E6E")))
    ws.conditional_formatting.add(rng, FormulaRule(formula=['OR($H2="",$J2="",$K2="",$M2="",$N2="",$P2="",$Q2="",$Y2="")'], fill=PatternFill("solid", fgColor=COL["missing"])))
    return last


def fill_run_log(ws) -> int:
    for i, h in enumerate(RUN_LOG_COLUMNS, start=1):
        ws.cell(1, i, h)
    style_header(ws, 1, len(RUN_LOG_COLUMNS))
    last = 31
    style_body(ws, 2, last, 1, len(RUN_LOG_COLUMNS))
    add_table(ws, "RunLogTable", f"A1:V{last}", "TableStyleMedium9")
    ws.freeze_panes = "A2"
    add_list_validation(ws, "=List_Reviewed", 19, 2, 1000)
    add_list_validation(ws, "=List_Used_in_Thesis", 20, 2, 1000)
    set_widths(ws, {"A": 18, "B": 13, "C": 12, "D": 20, "E": 24, "F": 22, "G": 30, "H": 12, "I": 16, "J": 24, "K": 16, "L": 22, "M": 18, "N": 18, "O": 18, "P": 30, "Q": 28, "R": 34, "S": 12, "T": 14, "U": 36, "V": 28})
    return last


def fill_decision_log(ws) -> int:
    for i, h in enumerate(DECISION_COLUMNS, start=1):
        ws.cell(1, i, h)
    style_header(ws, 1, len(DECISION_COLUMNS))
    rows = [
        ("D01", "Final target choice", "Stage 1 - Method choice", "fine emotion; coarse_affect; binary valence-like", "E01,E02,E03", "To be locked after evidence review", "", "", "Construct validity vs learnability tradeoff.", "", "Open", "Locks confirmatory target."),
        ("D02", "Final split design", "Stage 1 - Method choice", "within_subject_loso_session; weaker alternatives", "E04,E05", "To be locked after split stress test", "", "", "Weak split may inflate performance.", "", "Open", "Locks claim-matched split logic."),
        ("D03", "Final model family", "Stage 1 - Method choice", "ridge; logreg; linearsvc", "E06", "To be locked after model comparison", "", "", "Performance-stability-complexity tradeoff.", "", "Open", "Locks confirmatory model."),
        ("D04", "Final weighting policy", "Stage 1 - Method choice", "No weighting; class weighting", "E07", "To be locked after fairness/stability check", "", "", "Minority fairness vs instability tradeoff.", "", "Open", "Locks class weighting."),
        ("D05", "Final tuning policy", "Stage 1 - Method choice", "Fixed settings; light nested tuning", "E08", "To be locked after tuning check", "", "", "Added tuning complexity and overfit risk.", "", "Open", "Locks configuration policy."),
        ("D06", "Final feature representation", "Stage 1 - Method choice", "Whole-brain; ROI", "E09,E10", "To be locked after representation evidence", "", "", "Feature-space choice affects validity and stability.", "", "Open", "Locks feature space."),
        ("D07", "Final scaling policy", "Stage 1 - Method choice", "No scaling; standard/robust scaling", "E11", "To be locked after scaling sensitivity check", "", "", "Scaling can shift classifier behavior.", "", "Open", "Locks preprocessing."),
        ("D08", "Use of external data", "Stage 4 - External exploratory", "No external; external replication/portability", "E18,E19", "To be decided after confirmatory completion", "", "", "Comparability and validity limits for external data.", "", "Open", "Controls exploratory scope."),
    ]
    for r, row in enumerate(rows, start=2):
        for c, v in enumerate(row, start=1):
            ws.cell(r, c, v)
    last = 1 + len(rows)
    style_body(ws, 2, last, 1, len(DECISION_COLUMNS))
    add_table(ws, "DecisionLogTable", f"A1:L{last}", "TableStyleMedium4")
    ws.freeze_panes = "A2"
    add_list_validation(ws, "=List_Freeze_Status", 11, 2, 500, False)
    ws.conditional_formatting.add(f"K2:K{max(last,200)}", FormulaRule(formula=['$K2="Locked"'], fill=PatternFill("solid", fgColor=COL["ok"])))
    ws.conditional_formatting.add(f"K2:K{max(last,200)}", FormulaRule(formula=['$K2="Open"'], fill=PatternFill("solid", fgColor=COL["open"])))
    set_widths(ws, {"A": 12, "B": 24, "C": 24, "D": 32, "E": 18, "F": 34, "G": 24, "H": 22, "I": 30, "J": 12, "K": 14, "L": 30})
    return last


def fill_confirmatory(ws) -> int:
    for i, h in enumerate(CONFIRMATORY_COLUMNS, start=1):
        ws.cell(1, i, h)
    style_header(ws, 1, len(CONFIRMATORY_COLUMNS))
    rows = [
        ("E12", "Permutation test experiment", "To be fixed after freeze lock", "Chance-distinguishability support"),
        ("E13", "Trivial baseline experiment", "To be fixed after freeze lock", "Non-trivial model value support"),
        ("E14", "Stability of explanation experiment", "To be fixed after freeze lock", "Model-behavior stability support"),
        ("E16", "Final within-person confirmatory analysis", "To be fixed after freeze lock", "Primary within-person claim"),
        ("E17", "Final cross-person transfer analysis", "To be fixed after freeze lock", "Directional transfer claim"),
    ]
    for r, row in enumerate(rows, start=2):
        ws.cell(r, 1, row[0])
        ws.cell(r, 2, row[1])
        ws.cell(r, 3, row[2])
        ws.cell(r, 4, row[3])
        ws.cell(r, 5, f'=IFERROR(INDEX(Master_Experiments!$AA:$AA,MATCH($A{r},Master_Experiments!$A:$A,0)),"Planned")')
        ws.cell(r, 6, f'=IFERROR(INDEX(Master_Experiments!$AC:$AC,MATCH($A{r},Master_Experiments!$A:$A,0)),"")')
        ws.cell(r, 7, "Planned")
        ws.cell(r, 8, f'=IF(AND($E{r}="Completed",$I{r}="YES"),"YES","NO")')
        ws.cell(r, 9, '=IF(AND(COUNTIFS(Decision_Log!$A:$A,"D01",Decision_Log!$K:$K,"Locked")>0,COUNTIFS(Decision_Log!$A:$A,"D02",Decision_Log!$K:$K,"Locked")>0,COUNTIFS(Decision_Log!$A:$A,"D03",Decision_Log!$K:$K,"Locked")>0,COUNTIFS(Decision_Log!$A:$A,"D04",Decision_Log!$K:$K,"Locked")>0,COUNTIFS(Decision_Log!$A:$A,"D05",Decision_Log!$K:$K,"Locked")>0,COUNTIFS(Decision_Log!$A:$A,"D06",Decision_Log!$K:$K,"Locked")>0,COUNTIFS(Decision_Log!$A:$A,"D07",Decision_Log!$K:$K,"Locked")>0),"YES","NO")')
    last = 1 + len(rows)
    style_body(ws, 2, last, 1, len(CONFIRMATORY_COLUMNS))
    add_table(ws, "ConfirmatoryTable", f"A1:I{last}", "TableStyleMedium10")
    ws.freeze_panes = "A2"
    set_widths(ws, {"A": 13, "B": 32, "C": 28, "D": 34, "E": 14, "F": 30, "G": 20, "H": 20, "I": 22})
    ws.conditional_formatting.add(f"I2:I{max(last,200)}", FormulaRule(formula=['$I2="YES"'], fill=PatternFill("solid", fgColor=COL["ok"])))
    ws.conditional_formatting.add(f"I2:I{max(last,200)}", FormulaRule(formula=['$I2="NO"'], fill=PatternFill("solid", fgColor=COL["bad"])))
    return last


def fill_thesis_map(ws) -> int:
    for i, h in enumerate(THESIS_MAP_COLUMNS, start=1):
        ws.cell(1, i, h)
    style_header(ws, 1, len(THESIS_MAP_COLUMNS))
    rows = [
        ("E01", "Chapter 3", "3.3 Target definition", "Target granularity method choice", "Supporting", "Method choice", "Locks target definition", "Feeds D01"),
        ("E02", "Chapter 3", "3.3 Task strategy", "Task pooling choice", "Supporting", "Method choice", "Task pooling decision", ""),
        ("E03", "Chapter 3", "3.3 Modality strategy", "Modality pooling choice", "Supporting", "Method choice", "Modality pooling decision", ""),
        ("E04", "Chapter 3", "3.4 Split design", "Split-strength stress evidence", "Supporting", "Method choice", "Strict split justification", ""),
        ("E05", "Chapter 3", "3.4 Transfer protocol", "Cross-person framing", "Supporting", "Method choice", "Frozen directional transfer protocol", ""),
        ("E06", "Chapter 3", "3.5 Model family", "Model comparison evidence", "Supporting", "Method choice", "Final model family lock", ""),
        ("E07", "Chapter 3", "3.5 Weighting policy", "Class-weighting evidence", "Supporting", "Method choice", "Weighting decision", ""),
        ("E08", "Chapter 3", "3.5 Tuning policy", "Tuning strategy evidence", "Supporting", "Method choice", "Fixed vs nested tuning", ""),
        ("E09", "Chapter 3", "3.2 Feature space", "Whole-brain vs ROI decision", "Supporting", "Method choice", "Feature representation lock", ""),
        ("E10", "Chapter 3", "3.2 Dimensionality", "Reduction strategy decision", "Supporting", "Method choice", "Complexity vs stability tradeoff", ""),
        ("E11", "Chapter 3", "3.2 Scaling", "Scaling sensitivity decision", "Supporting", "Method choice", "Scaling lock", ""),
        ("E12", "Chapter 4", "4.3 Robustness", "Permutation support", "Supporting", "Method application", "Beyond-chance evidence", ""),
        ("E13", "Chapter 4", "4.3 Robustness", "Trivial baseline support", "Supporting", "Method application", "Non-trivial predictive value", ""),
        ("E14", "Chapter 4", "4.3 Robustness", "Interpretability stability support", "Supporting", "Method application", "Model-behavior robustness", "No localization claim"),
        ("E15", "Chapter 5", "5.2 Limitations", "Subset sensitivity discussion", "Supporting", "Discussion", "Bounds dependence on subsets", ""),
        ("E16", "Chapter 4", "4.1 Main results", "Within-person confirmatory evidence", "Main", "Method application", "Primary thesis claim", ""),
        ("E17", "Chapter 4", "4.2 Transfer results", "Cross-person transfer evidence", "Main", "Method application", "Secondary transfer claim", ""),
        ("E18", "Appendix", "A. External exploratory", "External split replication", "Supporting", "Discussion", "Exploratory external consistency", ""),
        ("E19", "Appendix", "A. External exploratory", "External model portability", "Supporting", "Discussion", "Exploratory portability check", ""),
    ]
    for r, row in enumerate(rows, start=2):
        for c, v in enumerate(row, start=1):
            ws.cell(r, c, v)
    last = 1 + len(rows)
    style_body(ws, 2, last, 1, len(THESIS_MAP_COLUMNS))
    add_table(ws, "ThesisMapTable", f"A1:H{last}", "TableStyleMedium6")
    ws.freeze_panes = "A2"
    set_widths(ws, {"A": 13, "B": 12, "C": 24, "D": 28, "E": 14, "F": 44, "G": 34, "H": 28})
    return last


def fill_dictionary(ws, wb: Workbook) -> None:
    ws["A1"] = "Controlled vocabularies (drives dropdown validation)"
    ws["A1"].font = Font(size=12, bold=True)
    ws["A1"].fill = PatternFill("solid", fgColor=COL["title_bg"])
    ws.merge_cells("A1:I1")
    names = ["Category", "Evidential_Role", "Stage", "Priority", "Status", "Reporting_Destination", "Reviewed", "Used_in_Thesis", "Freeze_Status"]
    for c, name in enumerate(names, start=1):
        ws.cell(2, c, name)
        ws.cell(2, c).font = Font(color=COL["header_fg"], bold=True)
        ws.cell(2, c).fill = PatternFill("solid", fgColor=COL["header_bg"])
        items = VOCABS[name]
        for i, item in enumerate(items, start=3):
            ws.cell(i, c, item)
            ws.cell(i, c).border = THIN
            if i % 2 == 0:
                ws.cell(i, c).fill = PatternFill("solid", fgColor=COL["zebra"])
        add_named_list(wb, f"List_{name}", ws.title, c, 3, 2 + len(items))
    ws["K1"] = "Term definitions"
    ws["K1"].font = Font(size=12, bold=True)
    ws["K1"].fill = PatternFill("solid", fgColor=COL["title_bg"])
    ws.merge_cells("K1:L1")
    ws["K2"] = "Term"
    ws["L2"] = "Definition"
    for cell in ("K2", "L2"):
        ws[cell].font = Font(color=COL["header_fg"], bold=True)
        ws[cell].fill = PatternFill("solid", fgColor=COL["header_bg"])
    for r, (term, definition) in enumerate(DEFINITIONS, start=3):
        ws.cell(r, 11, term)
        ws.cell(r, 12, definition)
        ws.cell(r, 11).border = THIN
        ws.cell(r, 12).border = THIN
        ws.cell(r, 12).alignment = Alignment(wrap_text=True, vertical="top")
        if r % 2 == 0:
            ws.cell(r, 11).fill = PatternFill("solid", fgColor=COL["zebra"])
            ws.cell(r, 12).fill = PatternFill("solid", fgColor=COL["zebra"])
    set_widths(ws, {"A": 24, "B": 30, "C": 24, "D": 12, "E": 16, "F": 28, "G": 10, "H": 14, "I": 14, "K": 24, "L": 80})
    ws.freeze_panes = "A3"


def fill_dashboard(ws) -> None:
    ws.merge_cells("A1:H1")
    ws["A1"] = "Dashboard"
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
        "Feature pipeline locked?",
        "Confirmatory ready?",
        "Experiments mapped to Chapter 4 main results",
        "Experiments mapped only to Appendix",
    ]
    formulas = [
        '=COUNTA(Master_Experiments!$A$2:$A$200)',
        '=COUNTIF(Master_Experiments!$AA$2:$AA$200,"Completed")',
        '=COUNTIFS(Master_Experiments!$F$2:$F$200,"Critical",Master_Experiments!$AA$2:$AA$200,"Completed")',
        '=COUNTIF(Decision_Log!$K$2:$K$200,"Open")',
        '=COUNTIF(Decision_Log!$K$2:$K$200,"Locked")',
        '=IF(COUNTIFS(Decision_Log!$A:$A,"D01",Decision_Log!$K:$K,"Locked")>0,"YES","NO")',
        '=IF(COUNTIFS(Decision_Log!$A:$A,"D02",Decision_Log!$K:$K,"Locked")>0,"YES","NO")',
        '=IF(COUNTIFS(Decision_Log!$A:$A,"D03",Decision_Log!$K:$K,"Locked")>0,"YES","NO")',
        '=IF(AND(COUNTIFS(Decision_Log!$A:$A,"D06",Decision_Log!$K:$K,"Locked")>0,COUNTIFS(Decision_Log!$A:$A,"D07",Decision_Log!$K:$K,"Locked")>0),"YES","NO")',
        '=IF(AND(B9="YES",B10="YES",B11="YES",B12="YES",COUNTIFS(Confirmatory_Set!$I:$I,"YES")>=5),"YES","NO")',
        '=COUNTIFS(Thesis_Map!$B$2:$B$200,"Chapter 4",Thesis_Map!$E$2:$E$200,"Main")',
        '=COUNTIF(Thesis_Map!$B$2:$B$200,"Appendix")',
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
    set_widths(ws, {"A": 44, "B": 16, "C": 3, "D": 26, "E": 10, "F": 3, "G": 34, "H": 10})
    ws.freeze_panes = "A4"
    ws.conditional_formatting.add("B9:B13", FormulaRule(formula=['B9="YES"'], fill=PatternFill("solid", fgColor=COL["ok"])))
    ws.conditional_formatting.add("B9:B13", FormulaRule(formula=['B9="NO"'], fill=PatternFill("solid", fgColor=COL["bad"])))


def build_workbook() -> Workbook:
    wb = Workbook()
    wb.remove(wb.active)
    for name in SHEET_ORDER:
        wb.create_sheet(name)
    fill_readme(wb["README"])
    fill_master(wb["Master_Experiments"])
    fill_run_log(wb["Run_Log"])
    fill_decision_log(wb["Decision_Log"])
    fill_confirmatory(wb["Confirmatory_Set"])
    fill_thesis_map(wb["Thesis_Map"])
    fill_dictionary(wb["Dictionary_Validation"], wb)
    fill_dashboard(wb["Dashboard"])
    wb["README"]["A25"] = "Generated by create_thesis_experiment_workbook.py"
    wb["README"]["A25"].font = Font(italic=True, color="4B5563")
    return wb


def validate(path: Path) -> dict[str, str]:
    wb = load_workbook(path)
    sheet_ok = [ws.title for ws in wb.worksheets] == SHEET_ORDER
    m = wb["Master_Experiments"]
    c = wb["Confirmatory_Set"]
    dv_count = sum(len(w.data_validations.dataValidation) for w in [m, wb["Run_Log"], wb["Decision_Log"]])
    return {
        "sheet_order_ok": str(sheet_ok),
        "sheet_order": ", ".join(ws.title for ws in wb.worksheets),
        "master_rows": str(m.max_row),
        "run_log_rows": str(wb["Run_Log"].max_row),
        "decision_rows": str(wb["Decision_Log"].max_row),
        "data_validations": str(dv_count),
        "experiment_ready_formula": str(bool(m["AF2"].value)),
        "confirmatory_formula": str(bool(c["I2"].value)),
    }


def main() -> None:
    wb = build_workbook()
    wb.save(OUT_XLSX)
    summary = validate(OUT_XLSX)
    print("Created workbook:", OUT_XLSX.resolve())
    print("Sheet order:", summary["sheet_order"])
    print("Sheet order valid:", summary["sheet_order_ok"])
    print("Master rows:", summary["master_rows"])
    print("Run_Log rows:", summary["run_log_rows"])
    print("Decision_Log rows:", summary["decision_rows"])
    print("Data validations found:", summary["data_validations"])
    print("Experiment_Ready formula present:", summary["experiment_ready_formula"])
    print("Confirmatory_Eligible formula present:", summary["confirmatory_formula"])


if __name__ == "__main__":
    main()
