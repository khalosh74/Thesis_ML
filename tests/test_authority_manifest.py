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

    assert isinstance(payload.get("scientific_authority"), dict)
    assert isinstance(payload.get("runtime_authority"), dict)
    assert isinstance(payload.get("generation_authority"), dict)
    assert isinstance(payload.get("derived_package_mirrors"), dict)


def test_authority_manifest_paths_exist() -> None:
    repo_root = _repo_root()
    manifest_path = repo_root / "configs" / "authority_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))

    groups = [
        payload.get("scientific_authority", {}),
        payload.get("runtime_authority", {}),
        payload.get("generation_authority", {}),
        payload.get("derived_package_mirrors", {}),
    ]
    for group in groups:
        for _, rel_path in group.items():
            path = repo_root / str(rel_path)
            assert path.exists(), f"Authority manifest path does not exist: {rel_path}"


def test_runtime_authority_points_to_revised_registry() -> None:
    repo_root = _repo_root()
    manifest_path = repo_root / "configs" / "authority_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))

    runtime = payload["runtime_authority"]
    assert runtime["thesis_decision_support_registry"] == (
        "configs/decision_support_registry_revised_execution.json"
    )
    assert runtime["package_default_registry"] == "configs/decision_support_registry.json"


def test_generation_authority_distinguishes_template_and_study_instance() -> None:
    repo_root = _repo_root()
    manifest_path = repo_root / "configs" / "authority_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))

    generation = payload["generation_authority"]
    assert generation["workbook_template"] == "templates/thesis_experiment_program.xlsx"
    assert (
        generation["study_workbook_instance"] == "templates/thesis_experiment_program_revised.xlsx"
    )
    assert generation["workbook_template"] != generation["study_workbook_instance"]
