from __future__ import annotations

import json
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_authority_manifest_exists_and_declares_required_sections() -> None:
    manifest_path = _repo_root() / "configs" / "authority_manifest.json"
    assert manifest_path.exists()

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload.get("schema_version") == "authority-manifest-v1"

    assert isinstance(payload.get("scientific"), dict)
    assert isinstance(payload.get("runtime"), dict)
    assert isinstance(payload.get("generation"), dict)
    assert isinstance(payload.get("derived"), dict)
    assert isinstance(payload.get("archive"), dict)


def test_authority_manifest_paths_exist() -> None:
    repo_root = _repo_root()
    manifest_path = repo_root / "configs" / "authority_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))

    groups = [
        payload.get("scientific", {}),
        payload.get("runtime", {}),
        payload.get("generation", {}),
        payload.get("derived", {}),
        payload.get("archive", {}),
    ]
    for group in groups:
        for _, rel_path in group.items():
            path = repo_root / str(rel_path)
            assert path.exists(), f"Authority manifest path does not exist: {rel_path}"


def test_runtime_authority_points_to_revised_registry() -> None:
    repo_root = _repo_root()
    manifest_path = repo_root / "configs" / "authority_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))

    runtime = payload["runtime"]
    assert runtime["thesis_decision_support_registry"] == (
        "configs/decision_support_registry_revised_execution.json"
    )
    assert runtime["package_default_registry"] == "configs/decision_support_registry.json"


def test_generation_authority_distinguishes_template_and_study_instance() -> None:
    repo_root = _repo_root()
    manifest_path = repo_root / "configs" / "authority_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))

    generation = payload["generation"]
    assert generation["workbook_template"] == "templates/thesis_experiment_program.xlsx"
    assert (
        generation["study_workbook_instance"]
        == "workbooks/thesis_program_instances/thesis_experiment_program_revised_v1.xlsx"
    )
    assert generation["workbook_template"] != generation["study_workbook_instance"]
