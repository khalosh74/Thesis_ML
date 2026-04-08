from __future__ import annotations

from pathlib import Path

from Thesis_ML.release.loader import load_release_bundle


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_release_bundle_loads_all_subordinate_authorities() -> None:
    release_path = _repo_root() / "releases" / "thesis_final_v1" / "release.json"
    loaded = load_release_bundle(release_path)

    assert loaded.release.release_id == "thesis_final_v1"
    assert loaded.science.release_id == loaded.release.release_id
    assert loaded.execution.hardware_mode == "cpu_only"
    assert loaded.environment.official_python == "3.13"
    assert loaded.evidence.verify_dataset_fingerprint is True
    assert loaded.claims.primary_claim

    assert loaded.science_path.name == "science.json"
    assert loaded.execution_path.name == "execution.json"
    assert loaded.environment_path.name == "environment.json"
    assert loaded.evidence_path.name == "evidence.json"
    assert loaded.claims_path.name == "claims.json"
