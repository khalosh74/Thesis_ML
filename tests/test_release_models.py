from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from Thesis_ML.release.models import ReleaseBundle, ReleaseScience


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_release_science_model_validates_shipped_payload() -> None:
    science_path = _repo_root() / "releases" / "thesis_final_v1" / "science.json"
    payload = json.loads(science_path.read_text(encoding="utf-8"))
    model = ReleaseScience.model_validate(payload)
    assert model.release_id == "thesis_final_v1"
    assert model.model_policy.tuning_enabled is False


def test_release_bundle_model_rejects_extra_fields() -> None:
    release_path = _repo_root() / "releases" / "thesis_final_v1" / "release.json"
    payload = json.loads(release_path.read_text(encoding="utf-8"))
    payload["unexpected_field"] = "forbidden"

    with pytest.raises(ValidationError):
        ReleaseBundle.model_validate(payload)


def test_release_science_nested_extra_fields_are_rejected() -> None:
    science_path = _repo_root() / "releases" / "thesis_final_v1" / "science.json"
    payload = json.loads(science_path.read_text(encoding="utf-8"))
    payload["dataset_contract"]["extra_nested"] = True

    with pytest.raises(ValidationError):
        ReleaseScience.model_validate(payload)
