from __future__ import annotations

import importlib.util
import json
from pathlib import Path


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


def _write_review(path: Path, *, candidate_winner: dict[str, object], manual: bool = False) -> None:
    payload = {
        "experiment_id": path.stem.replace("_review", ""),
        "candidate_winner": candidate_winner,
        "manual_review_required": bool(manual),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def test_confirmatory_selection_bundle_resolved_required_stages_freeze_ready(
    tmp_path: Path,
) -> None:
    campaign_root = tmp_path / "campaign"
    reviews_dir = campaign_root / "preflight_reviews"
    reviews_dir.mkdir(parents=True, exist_ok=True)

    scope_path = tmp_path / "scope.json"
    _write_scope(scope_path)

    _write_review(reviews_dir / "E01_review.json", candidate_winner={"target": "coarse_affect"})
    _write_review(
        reviews_dir / "E04_review.json",
        candidate_winner={"cv": "within_subject_loso_session"},
    )
    _write_review(reviews_dir / "E06_review.json", candidate_winner={"model": "ridge"})
    _write_review(
        reviews_dir / "E07_review.json",
        candidate_winner={"class_weight_policy": "none"},
    )
    _write_review(
        reviews_dir / "E08_review.json",
        candidate_winner={"methodology_policy_name": "fixed_baselines_only"},
    )
    _write_review(
        reviews_dir / "E09_review.json",
        candidate_winner={"feature_space": "whole_brain_masked"},
    )
    _write_review(
        reviews_dir / "E10_review.json",
        candidate_winner={"dimensionality_strategy": "none"},
    )
    _write_review(
        reviews_dir / "E11_review.json",
        candidate_winner={"preprocessing_strategy": "standardize_zscore"},
    )
    _write_review(
        reviews_dir / "E02_review.json",
        candidate_winner={"task_pooling_choice": "task_specific"},
    )
    _write_review(
        reviews_dir / "E03_review.json",
        candidate_winner={"modality_pooling_choice": "modality_specific"},
    )

    script = _load_script_module(Path("scripts") / "review_preflight_stage.py")
    bundle_path = script.emit_confirmatory_selection_bundle(
        campaign_root=campaign_root,
        computed_reviews={},
        scope_config_path=scope_path,
    )

    payload = json.loads(bundle_path.read_text(encoding="utf-8"))
    assert payload["scope_id"] == "confirmatory_scope_v1"
    assert payload["freeze_ready"] is True
    assert payload["manual_review_required"] is False

    assert payload["selected"]["target"] == "coarse_affect"
    assert payload["selected"]["cv_within_subject"] == "within_subject_loso_session"
    assert payload["selected"]["cv_transfer"] == "frozen_cross_person_transfer"
    assert payload["selected"]["model"] == "ridge"
    assert payload["selected"]["class_weight_policy"] == "none"
    assert payload["selected"]["methodology_policy_name"] == "fixed_baselines_only"
    assert payload["selected"]["feature_space"] == "whole_brain_masked"
    assert payload["selected"]["dimensionality_strategy"] == "none"
    assert payload["selected"]["preprocessing_strategy"] == "standardize_zscore"

    assert "task_pooling" in payload["advisory"]
    assert "modality_pooling" in payload["advisory"]
    assert "task_pooling" not in payload["selected"]
    assert "modality_pooling" not in payload["selected"]
    notes = payload["selection_reporting_notes"]
    assert notes["selection_source"] == "reviewed_preflight_stage_outputs"
    assert notes["reporting_mode"] == "locked_confirmatory"
    assert notes["dataset_relationship"] == "shared_overall_project_dataset"
    assert notes["external_validation_equivalence"] == "not_equivalent"
    assert "same overall project dataset" in notes["interpretation_note"]


def test_confirmatory_selection_bundle_missing_required_stage_sets_freeze_false(
    tmp_path: Path,
) -> None:
    campaign_root = tmp_path / "campaign"
    reviews_dir = campaign_root / "preflight_reviews"
    reviews_dir.mkdir(parents=True, exist_ok=True)
    scope_path = tmp_path / "scope.json"
    _write_scope(scope_path)

    # Deliberately omit E10 to force freeze_ready=false.
    _write_review(reviews_dir / "E01_review.json", candidate_winner={"target": "coarse_affect"})
    _write_review(
        reviews_dir / "E04_review.json",
        candidate_winner={"cv": "within_subject_loso_session"},
    )
    _write_review(reviews_dir / "E06_review.json", candidate_winner={"model": "ridge"})
    _write_review(
        reviews_dir / "E07_review.json",
        candidate_winner={"class_weight_policy": "none"},
    )
    _write_review(
        reviews_dir / "E08_review.json",
        candidate_winner={"methodology_policy_name": "fixed_baselines_only"},
    )
    _write_review(
        reviews_dir / "E09_review.json",
        candidate_winner={"feature_space": "whole_brain_masked"},
    )
    _write_review(
        reviews_dir / "E11_review.json",
        candidate_winner={"preprocessing_strategy": "none"},
    )

    script = _load_script_module(Path("scripts") / "review_preflight_stage.py")
    bundle_path = script.emit_confirmatory_selection_bundle(
        campaign_root=campaign_root,
        computed_reviews={},
        scope_config_path=scope_path,
    )

    payload = json.loads(bundle_path.read_text(encoding="utf-8"))
    assert payload["freeze_ready"] is False
    assert payload["manual_review_required"] is True
    assert any("E10:missing_review" in str(note) for note in payload["notes"])
