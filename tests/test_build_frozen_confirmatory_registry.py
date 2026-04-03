from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest


def _load_script_module(script_path: Path):
    spec = importlib.util.spec_from_file_location(script_path.stem, script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load script module: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_scope(path: Path) -> None:
    payload = {
        "scope_id": "confirmatory_scope_v1",
        "main_tasks": ["emo", "recog"],
        "main_modality": "audiovisual",
        "main_target": "coarse_affect",
        "within_subjects": ["sub-001", "sub-002"],
        "transfer_pairs": [
            {"train_subject": "sub-001", "test_subject": "sub-002"},
            {"train_subject": "sub-002", "test_subject": "sub-001"},
        ],
        "notes": {
            "task_pooling_stage": "advisory_only",
            "modality_pooling_stage": "advisory_only",
        },
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_bundle(path: Path) -> None:
    payload = {
        "bundle_id": "confirmatory_selection_bundle_campaign_a",
        "campaign_id": "campaign_a",
        "scope_id": "confirmatory_scope_v1",
        "review_sources": {
            "E01": "preflight_reviews/E01_review.json",
            "E04": "preflight_reviews/E04_review.json",
            "E06": "preflight_reviews/E06_review.json",
            "E07": "preflight_reviews/E07_review.json",
            "E08": "preflight_reviews/E08_review.json",
            "E09": "preflight_reviews/E09_review.json",
            "E10": "preflight_reviews/E10_review.json",
            "E11": "preflight_reviews/E11_review.json",
        },
        "selected": {
            "target": "coarse_affect",
            "cv_within_subject": "within_subject_loso_session",
            "cv_transfer": "frozen_cross_person_transfer",
            "model": "ridge",
            "class_weight_policy": "none",
            "methodology_policy_name": "fixed_baselines_only",
            "feature_space": "whole_brain_masked",
            "dimensionality_strategy": "none",
            "preprocessing_strategy": "standardize_zscore",
        },
        "advisory": {
            "task_pooling": "task_specific",
            "modality_pooling": "modality_specific",
        },
        "freeze_ready": True,
        "manual_review_required": False,
        "notes": [],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_index(path: Path, *, modality: str = "audiovisual") -> None:
    frame = pd.DataFrame(
        [
            {
                "sample_id": "s1",
                "subject": "sub-001",
                "session": "ses-01",
                "task": "emo",
                "modality": modality,
                "coarse_affect": "positive",
            },
            {
                "sample_id": "s2",
                "subject": "sub-001",
                "session": "ses-02",
                "task": "recog",
                "modality": modality,
                "coarse_affect": "negative",
            },
            {
                "sample_id": "s3",
                "subject": "sub-002",
                "session": "ses-01",
                "task": "emo",
                "modality": modality,
                "coarse_affect": "positive",
            },
            {
                "sample_id": "s4",
                "subject": "sub-002",
                "session": "ses-02",
                "task": "recog",
                "modality": modality,
                "coarse_affect": "negative",
            },
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _write_index_missing_subject(path: Path) -> None:
    frame = pd.DataFrame(
        [
            {
                "sample_id": "s1",
                "subject": "sub-001",
                "session": "ses-01",
                "task": "emo",
                "modality": "audiovisual",
                "coarse_affect": "positive",
            },
            {
                "sample_id": "s2",
                "subject": "sub-001",
                "session": "ses-02",
                "task": "recog",
                "modality": "audiovisual",
                "coarse_affect": "negative",
            },
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _write_index_missing_task(path: Path) -> None:
    frame = pd.DataFrame(
        [
            {
                "sample_id": "s1",
                "subject": "sub-001",
                "session": "ses-01",
                "task": "emo",
                "modality": "audiovisual",
                "coarse_affect": "positive",
            },
            {
                "sample_id": "s2",
                "subject": "sub-001",
                "session": "ses-02",
                "task": "emo",
                "modality": "audiovisual",
                "coarse_affect": "negative",
            },
            {
                "sample_id": "s3",
                "subject": "sub-002",
                "session": "ses-01",
                "task": "emo",
                "modality": "audiovisual",
                "coarse_affect": "positive",
            },
            {
                "sample_id": "s4",
                "subject": "sub-002",
                "session": "ses-02",
                "task": "emo",
                "modality": "audiovisual",
                "coarse_affect": "negative",
            },
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def test_build_frozen_confirmatory_registry_generates_expected_cells(tmp_path: Path) -> None:
    campaign_root = tmp_path / "campaigns" / "campaign_a"
    campaign_root.mkdir(parents=True, exist_ok=True)
    (campaign_root / "preflight_reviews").mkdir(parents=True, exist_ok=True)

    bundle_path = tmp_path / "bundle.json"
    scope_path = tmp_path / "scope.json"
    index_path = tmp_path / "index.csv"
    output_registry = (
        tmp_path / "configs" / "generated" / "frozen_confirmatory_registry_campaign_a.json"
    )

    _write_bundle(bundle_path)
    _write_scope(scope_path)
    _write_index(index_path)

    script = _load_script_module(Path("scripts") / "build_frozen_confirmatory_registry.py")
    exit_code = script.main(
        [
            "--campaign-root",
            str(campaign_root),
            "--selection-bundle",
            str(bundle_path),
            "--scope-config",
            str(scope_path),
            "--output-registry",
            str(output_registry),
            "--index-csv",
            str(index_path),
        ]
    )
    assert exit_code == 0
    assert output_registry.exists()

    manifest_path = output_registry.parent / "frozen_confirmatory_manifest_campaign_a.json"
    report_path = output_registry.parent / "frozen_confirmatory_report_campaign_a.md"
    assert manifest_path.exists()
    assert report_path.exists()

    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert (
        manifest_payload["selection_reporting_relationship"]
        == "preflight_selected_locked_confirmatory"
    )
    assert manifest_payload["validation_scope"] == "internal_project_dataset"
    assert manifest_payload["external_validation_equivalence"] == "not_equivalent"
    assert "same overall project dataset" in manifest_payload["interpretation_note"]

    report_text = report_path.read_text(encoding="utf-8")
    assert "## Selection and Validation Scope" in report_text
    assert "same overall project dataset" in report_text
    assert (
        "stronger than ad hoc tuning, but weaker than independent external validation"
        in report_text
    )

    registry_payload = json.loads(output_registry.read_text(encoding="utf-8"))
    templates = registry_payload["experiments"][0]["variant_templates"]
    assert len(templates) == 4

    within = [row for row in templates if row["params"]["cv"] == "within_subject_loso_session"]
    transfer = [row for row in templates if row["params"]["cv"] == "frozen_cross_person_transfer"]
    assert len(within) == 2
    assert len(transfer) == 2
    assert sorted(row["params"]["subject"] for row in within) == ["sub-001", "sub-002"]
    assert sorted(
        (row["params"]["train_subject"], row["params"]["test_subject"]) for row in transfer
    ) == [("sub-001", "sub-002"), ("sub-002", "sub-001")]

    for row in templates:
        params = row["params"]
        assert params["target"] == "coarse_affect"
        assert params["filter_modality"] == "audiovisual"
        assert params["scope_task_ids"] == ["emo", "recog"]
        assert params["model"] == "ridge"
        assert params["class_weight_policy"] == "none"
        assert params["methodology_policy_name"] == "fixed_baselines_only"
        assert params["feature_space"] == "whole_brain_masked"
        assert params["dimensionality_strategy"] == "none"
        assert params["preprocessing_strategy"] == "standardize_zscore"
        assert params["framework_mode"] == "confirmatory"
        protocol_context = params["protocol_context"]
        assert protocol_context["framework_mode"] == "confirmatory"
        assert protocol_context["protocol_id"] == "thesis_confirmatory_v1"

    outputs_path = campaign_root / "preflight_reviews" / "frozen_confirmatory_outputs.json"
    assert outputs_path.exists()
    outputs_payload = json.loads(outputs_path.read_text(encoding="utf-8"))
    assert "registry" in outputs_payload
    assert "manifest" in outputs_payload
    assert "report" in outputs_payload


@pytest.mark.parametrize(
    ("writer", "index_name"),
    [
        (_write_index_missing_subject, "index_missing_subject.csv"),
        (_write_index_missing_task, "index_missing_task.csv"),
        (lambda path: _write_index(path, modality="audio"), "index_missing_modality.csv"),
    ],
)
def test_build_frozen_confirmatory_registry_fails_when_scope_coverage_missing(
    tmp_path: Path,
    writer,
    index_name: str,
) -> None:
    campaign_root = tmp_path / "campaigns" / "campaign_a"
    campaign_root.mkdir(parents=True, exist_ok=True)

    bundle_path = tmp_path / "bundle.json"
    scope_path = tmp_path / "scope.json"
    index_path = tmp_path / index_name
    output_registry = (
        tmp_path / "configs" / "generated" / "frozen_confirmatory_registry_campaign_a.json"
    )

    _write_bundle(bundle_path)
    _write_scope(scope_path)
    writer(index_path)

    script = _load_script_module(Path("scripts") / "build_frozen_confirmatory_registry.py")
    with pytest.raises(Exception, match="Coverage validation failed"):
        script.main(
            [
                "--campaign-root",
                str(campaign_root),
                "--selection-bundle",
                str(bundle_path),
                "--scope-config",
                str(scope_path),
                "--output-registry",
                str(output_registry),
                "--index-csv",
                str(index_path),
            ]
        )
