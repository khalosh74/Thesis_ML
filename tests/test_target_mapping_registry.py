from __future__ import annotations

import json
from pathlib import Path

import pytest

from Thesis_ML.config.paths import DEFAULT_COARSE_AFFECT_TARGET_MAPPING_PATH
from Thesis_ML.data.target_mapping_registry import (
    canonical_mapping_object_hash,
    load_target_mapping,
)

_GROUND_TRUTH_MAPPING = {
    "anger": "negative",
    "anxiety": "negative",
    "disgust": "negative",
    "happiness": "positive",
    "interest": "positive",
    "neutral": "neutral",
    "pride": "positive",
    "relief": "positive",
    "sadness": "negative",
}
_GROUND_TRUTH_HASH = "3bbc30d6949e868f8d9d2ad64b7b10d1d14c93fff3d406d3321d76efee011ad3"


def test_affect_mapping_v2_loads_from_registry() -> None:
    entry = load_target_mapping("affect_mapping_v2")
    assert entry.version == "affect_mapping_v2"
    assert entry.path == DEFAULT_COARSE_AFFECT_TARGET_MAPPING_PATH.resolve()
    assert dict(entry.mapping) == _GROUND_TRUTH_MAPPING


def test_affect_mapping_v2_matches_locked_ground_truth_and_hash() -> None:
    entry = load_target_mapping("affect_mapping_v2")
    assert dict(entry.mapping) == _GROUND_TRUTH_MAPPING
    assert entry.mapping_hash == _GROUND_TRUTH_HASH


def test_canonical_hash_stable_across_formatting_differences(tmp_path: Path) -> None:
    mapping_path = tmp_path / "affect_mapping_v2_formatting_variant.json"
    mapping_path.write_text(
        json.dumps(
            {
                " Sadness ": " NEGATIVE ",
                "HAPPINESS": " positive ",
                " neutral ": " Neutral ",
                "interest": "Positive",
                "  pride": "POSITIVE",
                "anger": "negative",
                "DISGUST": " negative",
                "anxiety": "negative",
                "relief": "positive",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    entry = load_target_mapping(
        "affect_mapping_v2_formatting_variant",
        target_configs_dir=tmp_path,
    )
    assert dict(entry.mapping) == _GROUND_TRUTH_MAPPING
    assert entry.mapping_hash == _GROUND_TRUTH_HASH
    assert canonical_mapping_object_hash(entry.mapping) == _GROUND_TRUTH_HASH


def test_registry_rejects_invalid_output_labels(tmp_path: Path) -> None:
    mapping_path = tmp_path / "affect_mapping_invalid_output.json"
    mapping_path.write_text(
        json.dumps(
            {
                "anger": "negative",
                "happiness": "positive",
                "neutral": "neutral",
                "sadness": "unknown_label",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unsupported output label"):
        load_target_mapping("affect_mapping_invalid_output", target_configs_dir=tmp_path)
