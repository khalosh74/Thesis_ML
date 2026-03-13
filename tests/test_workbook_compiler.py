from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
from openpyxl import load_workbook

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


def test_compile_workbook_file_success(tmp_path: Path) -> None:
    workbook_path = tmp_path / "thesis_experiment_program.xlsx"
    _make_workbook(workbook_path)
    _set_executable_row(workbook_path)

    manifest = compile_workbook_file(workbook_path)

    assert manifest.schema_version == "workbook-v1"
    assert len(manifest.experiments) == 1
    assert manifest.experiments[0].experiment_id == "E16"
    assert len(manifest.trial_specs) == 1
    trial = manifest.trial_specs[0]
    assert trial.start_section == "dataset_selection"
    assert trial.end_section == "evaluation"
    assert trial.params["target"] == "coarse_affect"
    assert trial.params["cv"] == "within_subject_loso_session"
    assert trial.params["model"] == "ridge"


def test_compile_workbook_missing_required_columns_raises(tmp_path: Path) -> None:
    workbook_path = tmp_path / "thesis_experiment_program.xlsx"
    _make_workbook(workbook_path)
    workbook = load_workbook(workbook_path)
    ws = workbook["Experiment_Definitions"]
    ws["C1"] = "start_section_broken"
    workbook.save(workbook_path)

    with pytest.raises(ValueError, match="missing required columns"):
        compile_workbook_file(workbook_path)


def test_compile_workbook_invalid_section_name_raises(tmp_path: Path) -> None:
    workbook_path = tmp_path / "thesis_experiment_program.xlsx"
    _make_workbook(workbook_path)
    _set_executable_row(workbook_path)
    workbook = load_workbook(workbook_path)
    ws = workbook["Experiment_Definitions"]
    ws["C2"] = "invalid_section_name"
    workbook.save(workbook_path)

    with pytest.raises(ValueError, match="Invalid start_section"):
        compile_workbook_file(workbook_path)


def test_compile_workbook_invalid_base_artifact_usage_raises(tmp_path: Path) -> None:
    workbook_path = tmp_path / "thesis_experiment_program.xlsx"
    _make_workbook(workbook_path)
    _set_executable_row(workbook_path)
    workbook = load_workbook(workbook_path)
    ws = workbook["Experiment_Definitions"]
    ws["E2"] = "artifact_example_123"
    workbook.save(workbook_path)

    with pytest.raises(ValueError, match="Invalid base artifact usage"):
        compile_workbook_file(workbook_path)
