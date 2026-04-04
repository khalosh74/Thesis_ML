from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_freeze_module():
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "freeze_workbook_registry.py"
    spec = importlib.util.spec_from_file_location("freeze_workbook_registry", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_freeze_script_uses_canonical_study_workbook_path() -> None:
    module = _load_freeze_module()
    repo_root = Path(__file__).resolve().parents[1]
    resolved = module._resolve_workbook_path(
        module.CANONICAL_STUDY_WORKBOOK_PATH,
        cwd=repo_root,
    )
    assert resolved == (
        repo_root
        / "workbooks"
        / "thesis_program_instances"
        / "thesis_experiment_program_revised_v1.xlsx"
    ).resolve()


def test_freeze_script_maps_legacy_study_workbook_path_to_canonical() -> None:
    module = _load_freeze_module()
    repo_root = Path(__file__).resolve().parents[1]
    resolved = module._resolve_workbook_path(
        module.LEGACY_REVISED_WORKBOOK_PATH,
        cwd=repo_root,
    )
    assert resolved == (
        repo_root
        / "workbooks"
        / "thesis_program_instances"
        / "thesis_experiment_program_revised_v1.xlsx"
    ).resolve()

