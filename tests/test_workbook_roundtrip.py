from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest
from openpyxl import load_workbook

from Thesis_ML.orchestration.decision_support import run_workbook_decision_support_campaign
from Thesis_ML.orchestration.workbook_compiler import compile_workbook_file

_WORKBOOK_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "create_thesis_experiment_workbook.py"
)
_WORKBOOK_SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "create_thesis_experiment_workbook",
    _WORKBOOK_SCRIPT_PATH,
)
if _WORKBOOK_SCRIPT_SPEC is None or _WORKBOOK_SCRIPT_SPEC.loader is None:
    raise RuntimeError(f"Unable to load workbook script from {_WORKBOOK_SCRIPT_PATH}")
_workbook_script = importlib.util.module_from_spec(_WORKBOOK_SCRIPT_SPEC)
_WORKBOOK_SCRIPT_SPEC.loader.exec_module(_workbook_script)


def _make_workbook(path: Path) -> None:
    workbook = _workbook_script.build_workbook()
    workbook.save(path)


def _set_executable_row(path: Path) -> None:
    workbook = load_workbook(path)
    ws = workbook["Experiment_Definitions"]
    headers = [ws.cell(1, col).value for col in range(1, ws.max_column + 1)]
    col = {str(name): idx + 1 for idx, name in enumerate(headers)}

    ws.cell(2, col["experiment_id"], "E16")
    ws.cell(2, col["enabled"], "Yes")
    ws.cell(2, col["start_section"], "dataset_selection")
    ws.cell(2, col["end_section"], "evaluation")
    ws.cell(2, col["base_artifact_id"], "")
    ws.cell(2, col["target"], "coarse_affect")
    ws.cell(2, col["cv"], "within_subject_loso_session")
    ws.cell(2, col["model"], "ridge")
    ws.cell(2, col["subject"], "sub-001")
    ws.cell(2, col["train_subject"], "")
    ws.cell(2, col["test_subject"], "")
    ws.cell(2, col["filter_task"], "")
    ws.cell(2, col["filter_modality"], "")
    ws.cell(2, col["reuse_policy"], "auto")
    workbook.save(path)


def _write_index_csv(path: Path) -> None:
    df = pd.DataFrame(
        [
            {"sample_id": "s1", "subject": "sub-001", "task": "passive", "modality": "audio"},
            {"sample_id": "s2", "subject": "sub-001", "task": "emo", "modality": "video"},
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _stub_run_experiment(**kwargs: object) -> dict[str, object]:
    run_id = str(kwargs["run_id"])
    reports_root = Path(kwargs["reports_root"])
    report_dir = reports_root / run_id
    report_dir.mkdir(parents=True, exist_ok=True)

    config_path = report_dir / "config.json"
    metrics_path = report_dir / "metrics.json"
    fold_metrics_path = report_dir / "fold_metrics.csv"
    fold_splits_path = report_dir / "fold_splits.csv"
    predictions_path = report_dir / "predictions.csv"
    spatial_path = report_dir / "spatial_compatibility_report.json"
    interpretability_path = report_dir / "interpretability_summary.json"

    config_path.write_text('{"ok": true}\n', encoding="utf-8")
    metrics_path.write_text(
        json.dumps(
            {
                "accuracy": 0.62,
                "balanced_accuracy": 0.59,
                "macro_f1": 0.58,
                "n_folds": 2,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    fold_metrics_path.write_text("fold,score\n0,0.59\n", encoding="utf-8")
    fold_splits_path.write_text("fold,train,test\n0,a,b\n", encoding="utf-8")
    predictions_path.write_text("y_true,y_pred\nneg,neg\n", encoding="utf-8")
    spatial_path.write_text('{"status":"passed","passed":true,"n_groups_checked":1}\n', encoding="utf-8")
    interpretability_path.write_text('{"status":"not_applicable"}\n', encoding="utf-8")

    return {
        "run_id": run_id,
        "report_dir": str(report_dir),
        "config_path": str(config_path),
        "metrics_path": str(metrics_path),
        "fold_metrics_path": str(fold_metrics_path),
        "fold_splits_path": str(fold_splits_path),
        "predictions_path": str(predictions_path),
        "spatial_compatibility_report_path": str(spatial_path),
        "interpretability_summary_path": str(interpretability_path),
        "artifact_ids": {
            "metrics_bundle": f"metrics_{run_id}",
            "interpretability_bundle": f"interp_{run_id}",
            "experiment_report": f"report_{run_id}",
        },
        "metrics": {
            "accuracy": 0.62,
            "balanced_accuracy": 0.59,
            "macro_f1": 0.58,
            "n_folds": 2,
        },
    }


def _sheet_header_map(ws, header_row: int) -> dict[str, int]:
    return {
        str(ws.cell(header_row, col).value): col
        for col in range(1, ws.max_column + 1)
        if ws.cell(header_row, col).value
    }


def test_workbook_roundtrip_compile_execute_writeback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workbook_path = tmp_path / "thesis_experiment_program.xlsx"
    _make_workbook(workbook_path)
    _set_executable_row(workbook_path)
    index_csv = tmp_path / "dataset_index.csv"
    _write_index_csv(index_csv)

    manifest = compile_workbook_file(workbook_path)
    assert len(manifest.trial_specs) == 1

    from Thesis_ML.orchestration import decision_support as orchestrator

    monkeypatch.setattr(orchestrator, "run_experiment", _stub_run_experiment)

    result = run_workbook_decision_support_campaign(
        workbook_path=workbook_path,
        index_csv=index_csv,
        data_root=tmp_path / "Data",
        cache_dir=tmp_path / "cache",
        output_root=tmp_path / "artifacts" / "decision_support",
        experiment_id=None,
        stage=None,
        run_all=True,
        seed=42,
        n_permutations=0,
        dry_run=False,
        write_back_to_workbook=True,
        append_workbook_run_log=True,
        workbook_output_dir=tmp_path / "workbook_outputs",
    )

    output_workbook_path = Path(str(result["workbook_output_path"]))
    assert output_workbook_path.exists()
    assert output_workbook_path != workbook_path
    assert output_workbook_path.name.startswith("thesis_experiment_program__results_")

    output_wb = load_workbook(output_workbook_path)

    machine_ws = output_wb["Machine_Status"]
    machine_cols = _sheet_header_map(machine_ws, 2)
    machine_rows = [
        row
        for row in range(3, machine_ws.max_row + 1)
        if str(machine_ws.cell(row, machine_cols["machine_id"]).value or "").startswith("campaign_")
    ]
    assert machine_rows
    machine_status_value = machine_ws.cell(
        machine_rows[-1], machine_cols["status"]
    ).value
    assert machine_status_value in {"Open", "Monitoring", "Closed"}

    trial_ws = output_wb["Trial_Results"]
    trial_cols = _sheet_header_map(trial_ws, 2)
    trial_rows = [
        row
        for row in range(3, trial_ws.max_row + 1)
        if str(trial_ws.cell(row, trial_cols["experiment_id"]).value or "") == "E16"
    ]
    assert trial_rows
    latest_trial_row = trial_rows[-1]
    assert trial_ws.cell(latest_trial_row, trial_cols["status"]).value == "completed"
    assert trial_ws.cell(latest_trial_row, trial_cols["report_path"]).value
    assert trial_ws.cell(latest_trial_row, trial_cols["metrics_path"]).value

    run_log_ws = output_wb["Run_Log"]
    run_log_cols = _sheet_header_map(run_log_ws, 1)
    run_log_rows = [
        row
        for row in range(2, run_log_ws.max_row + 1)
        if str(run_log_ws.cell(row, run_log_cols["Experiment_ID"]).value or "") == "E16"
    ]
    assert run_log_rows
    latest_run_log_row = run_log_rows[-1]
    assert run_log_ws.cell(latest_run_log_row, run_log_cols["Result_Summary"]).value == "completed"
    assert run_log_ws.cell(latest_run_log_row, run_log_cols["Artifact_Path"]).value

    source_wb = load_workbook(workbook_path)
    source_trial_ws = source_wb["Trial_Results"]
    source_trial_cols = _sheet_header_map(source_trial_ws, 2)
    source_rows_with_e16 = [
        row
        for row in range(3, source_trial_ws.max_row + 1)
        if str(source_trial_ws.cell(row, source_trial_cols["experiment_id"]).value or "") == "E16"
    ]
    assert source_rows_with_e16 == []

