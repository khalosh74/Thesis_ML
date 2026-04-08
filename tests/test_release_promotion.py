from __future__ import annotations

import json
from pathlib import Path

import pytest

from Thesis_ML.release.models import RunClass
from Thesis_ML.release.promotion import promote_candidate_run
from Thesis_ML.release.runner import run_release
from tests._release_test_utils import (
    dataset_manifest_path,
    fake_compile_and_run_protocol_factory,
    make_temp_release_bundle,
)


def _create_candidate_run(
    *,
    release_path: Path,
    run_id: str,
) -> Path:
    result = run_release(
        release_ref=release_path,
        dataset_manifest_path=dataset_manifest_path(),
        run_class=RunClass.CANDIDATE,
        run_id=run_id,
    )
    return Path(str(result["run_dir"]))


def test_candidate_run_can_be_promoted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release_path = make_temp_release_bundle(tmp_path)
    monkeypatch.setattr(
        "Thesis_ML.release.runner.compile_and_run_protocol",
        fake_compile_and_run_protocol_factory(),
    )

    candidate_dir = _create_candidate_run(release_path=release_path, run_id="candidate_promote_ok")
    promotion = promote_candidate_run(candidate_dir)

    assert promotion["passed"] is True
    official_dir = Path(str(promotion["official_run_path"]))
    assert (official_dir / "run_manifest.json").exists()
    assert (official_dir / "promotion_manifest.json").exists()
    assert (official_dir / "official_release_manifest.json").exists()
    assert (official_dir / "official_lock.json").exists()

    official_manifest = json.loads((official_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert official_manifest["run_class"] == "official"
    assert official_manifest["official"] is True
    assert official_manifest["parent_run_id"] == "candidate_promote_ok"


def test_promotion_blocks_when_evidence_verification_failed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release_path = make_temp_release_bundle(tmp_path)
    monkeypatch.setattr(
        "Thesis_ML.release.runner.compile_and_run_protocol",
        fake_compile_and_run_protocol_factory(),
    )

    candidate_dir = _create_candidate_run(release_path=release_path, run_id="candidate_bad_verification")

    verification_path = candidate_dir / "verification" / "evidence_verification.json"
    payload = json.loads(verification_path.read_text(encoding="utf-8"))
    payload["passed"] = False
    verification_path.write_text(f"{json.dumps(payload, indent=2)}\n", encoding="utf-8")

    manifest_path = candidate_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["evidence_verified"] = False
    manifest["promotable"] = False
    manifest_path.write_text(f"{json.dumps(manifest, indent=2)}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="promotable=false"):
        promote_candidate_run(candidate_dir)


def test_second_official_run_for_same_release_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release_path = make_temp_release_bundle(tmp_path)
    monkeypatch.setattr(
        "Thesis_ML.release.runner.compile_and_run_protocol",
        fake_compile_and_run_protocol_factory(),
    )

    first_candidate = _create_candidate_run(release_path=release_path, run_id="candidate_first")
    promote_candidate_run(first_candidate)

    second_candidate = _create_candidate_run(release_path=release_path, run_id="candidate_second")
    with pytest.raises(ValueError, match="Official singleton violation"):
        promote_candidate_run(second_candidate)
