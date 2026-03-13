from __future__ import annotations

from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.worksheet import Worksheet

from Thesis_ML.workbook.template_primitives import (
    COL,
    add_list_validation,
    add_table,
    col_idx,
    set_widths,
    style_body,
    style_header,
)


def fill_simple_structured_sheet(
    ws: Worksheet,
    *,
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
    for col_idx_value, header in enumerate(columns, start=1):
        ws.cell(2, col_idx_value, header)
    style_header(ws, 2, len(columns))

    if starter_rows:
        for row_idx, row in enumerate(starter_rows, start=3):
            for col_idx_value, name in enumerate(columns, start=1):
                ws.cell(row_idx, col_idx_value, row.get(name, ""))

    last = 61
    style_body(ws, 3, last, 1, len(columns))
    end_col = get_column_letter(len(columns))
    add_table(ws, table_name, f"A2:{end_col}{last}", style="TableStyleMedium6")
    ws.freeze_panes = "A3"
    ws.auto_filter.ref = f"A2:{end_col}{last}"
    set_widths(ws, width_map)
    return last


def fill_artifact_registry(ws: Worksheet, *, artifact_registry_columns: list[str]) -> int:
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
    return fill_simple_structured_sheet(
        ws=ws,
        columns=artifact_registry_columns,
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


def fill_fixed_configs(ws: Worksheet, *, fixed_configs_columns: list[str]) -> int:
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
    last = fill_simple_structured_sheet(
        ws=ws,
        columns=fixed_configs_columns,
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
    add_list_validation(ws, "=List_YesNo", col_idx(fixed_configs_columns, "locked"), 3, 1000)
    return last


def fill_objectives(ws: Worksheet, *, objectives_columns: list[str]) -> int:
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
    last = fill_simple_structured_sheet(
        ws=ws,
        columns=objectives_columns,
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
    add_list_validation(ws, "=List_Stage", col_idx(objectives_columns, "stage"), 3, 1000)
    add_list_validation(
        ws,
        "=List_Experiment_ID",
        col_idx(objectives_columns, "linked_experiment_id"),
        3,
        1000,
    )
    add_list_validation(ws, "=List_Status", col_idx(objectives_columns, "status"), 3, 1000)
    return last


def fill_machine_status(ws: Worksheet, *, machine_status_columns: list[str]) -> int:
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
    last = fill_simple_structured_sheet(
        ws=ws,
        columns=machine_status_columns,
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
    add_list_validation(
        ws,
        "=List_Ethics_Status",
        col_idx(machine_status_columns, "status"),
        3,
        1000,
    )
    return last


def fill_trial_results(ws: Worksheet, *, trial_results_columns: list[str]) -> int:
    last = fill_simple_structured_sheet(
        ws=ws,
        columns=trial_results_columns,
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
    add_list_validation(ws, "=List_Experiment_ID", col_idx(trial_results_columns, "experiment_id"), 3, 1000)
    add_list_validation(ws, "=List_Status", col_idx(trial_results_columns, "status"), 3, 1000)
    return last


def fill_summary_outputs(ws: Worksheet, *, summary_outputs_columns: list[str]) -> int:
    return fill_simple_structured_sheet(
        ws=ws,
        columns=summary_outputs_columns,
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


__all__ = [
    "fill_artifact_registry",
    "fill_fixed_configs",
    "fill_machine_status",
    "fill_objectives",
    "fill_summary_outputs",
    "fill_trial_results",
]
