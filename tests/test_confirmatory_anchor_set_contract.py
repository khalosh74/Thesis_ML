from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from Thesis_ML.orchestration.variant_expansion import materialize_experiment_cells
from Thesis_ML.verification.confirmatory_scope_runtime_alignment import (
    collect_runtime_confirmatory_anchors,
    verify_confirmatory_scope_runtime_alignment,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _to_variant_templates(experiment: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, template in enumerate(list(experiment.get("variant_templates") or []), start=1):
        if not isinstance(template, dict):
            continue
        rows.append(
            {
                "template_id": str(template.get("template_id") or f"variant_{index:03d}"),
                "variant_index": index,
                "supported": bool(template.get("supported", True)),
                "blocked_reason": template.get("unsupported_reason"),
                "params": dict(template.get("params") or {}),
                "factor_settings": dict(template.get("factor_settings") or {}),
                "fixed_controls": dict(template.get("fixed_controls") or {}),
                "design_metadata": dict(template.get("design_metadata") or {}),
            }
        )
    return rows


def _analysis_labels_from_cells(cells: list[dict[str, Any]]) -> set[str]:
    labels: set[str] = set()
    for cell in cells:
        params = dict(cell.get("params") or {})
        cv_mode = str(params.get("cv", "")).strip()
        if cv_mode == "within_subject_loso_session":
            subject = str(params.get("subject", "")).strip()
            if subject:
                labels.add(f"within_subject_loso_session:{subject}")
        elif cv_mode == "frozen_cross_person_transfer":
            train_subject = str(params.get("train_subject", "")).strip()
            test_subject = str(params.get("test_subject", "")).strip()
            if train_subject and test_subject:
                labels.add(f"frozen_cross_person_transfer:{train_subject}->{test_subject}")
    return labels


def test_thesis_runtime_anchor_set_matches_confirmatory_scope() -> None:
    root = _repo_root()
    scope_path = root / "configs" / "confirmatory" / "confirmatory_scope_v1.json"
    registry_path = root / "configs" / "decision_support_registry_revised_execution.json"

    alignment = verify_confirmatory_scope_runtime_alignment(
        scope_config_path=scope_path,
        runtime_registry_path=registry_path,
    )
    assert alignment["passed"] is True

    runtime_payload = json.loads(registry_path.read_text(encoding="utf-8"))
    anchors = collect_runtime_confirmatory_anchors(runtime_payload)
    labels = {str(row["analysis_label"]) for row in anchors}
    assert labels == {
        "within_subject_loso_session:sub-001",
        "within_subject_loso_session:sub-002",
        "frozen_cross_person_transfer:sub-001->sub-002",
        "frozen_cross_person_transfer:sub-002->sub-001",
    }


def test_e12_and_e13_materialize_against_full_runtime_anchor_set() -> None:
    root = _repo_root()
    registry_path = root / "configs" / "decision_support_registry_revised_execution.json"
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    experiments = [row for row in list(payload.get("experiments") or []) if isinstance(row, dict)]
    by_id = {str(row.get("experiment_id")): row for row in experiments}

    e12_cells, e12_warnings = materialize_experiment_cells(
        experiment={"experiment_id": "E12"},
        variants=_to_variant_templates(by_id["E12"]),
        dataset_scope={},
        n_permutations=5,
        registry_experiments=experiments,
    )
    e13_cells, e13_warnings = materialize_experiment_cells(
        experiment={"experiment_id": "E13"},
        variants=_to_variant_templates(by_id["E13"]),
        dataset_scope={},
        n_permutations=5,
        registry_experiments=experiments,
    )

    assert e12_warnings == []
    assert e13_warnings == []

    expected_labels = {
        "within_subject_loso_session:sub-001",
        "within_subject_loso_session:sub-002",
        "frozen_cross_person_transfer:sub-001->sub-002",
        "frozen_cross_person_transfer:sub-002->sub-001",
    }
    assert _analysis_labels_from_cells(e12_cells) == expected_labels
    assert _analysis_labels_from_cells(e13_cells) == expected_labels

