from __future__ import annotations

from Thesis_ML.observability.eta import build_runtime_keys


def test_build_runtime_keys_is_stable_and_contains_required_axes() -> None:
    metadata = {
        "experiment_id": "E16",
        "phase_name": "Confirmatory",
        "framework_mode": "confirmatory",
        "model_cost_tier": "official_fast",
        "feature_space": "whole_brain_masked",
        "preprocessing_strategy": "standardize_zscore",
        "dimensionality_strategy": "none",
        "tuning_enabled": False,
        "cv_mode": "within_subject_loso_session",
        "n_permutations": 200,
        "subject": "sub-001",
        "task": "emo",
        "modality": "audiovisual",
    }
    keys_1 = build_runtime_keys(metadata)
    keys_2 = build_runtime_keys(dict(metadata))

    assert keys_1 == keys_2
    assert keys_1["exact"].startswith("exact|")
    assert "experiment_id=e16" in keys_1["exact"]
    assert "framework_mode=confirmatory" in keys_1["exact"]
    assert "n_permutations=200" in keys_1["exact"]
    assert keys_1["backoff_1"].startswith("backoff_1|")
    assert "phase_name=confirmatory" in keys_1["backoff_1"]
    assert keys_1["backoff_2"].startswith("backoff_2|")


def test_build_runtime_keys_uses_missing_placeholders() -> None:
    keys = build_runtime_keys({})
    assert "__na__" in keys["exact"]
    assert "__na__" in keys["backoff_1"]
    assert "__na__" in keys["backoff_2"]
