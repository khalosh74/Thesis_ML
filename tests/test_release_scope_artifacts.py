from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tests._release_test_utils import (
    dataset_manifest_path,
    fake_compile_and_run_protocol_factory,
    make_temp_release_bundle,
)
from Thesis_ML.release.models import RunClass
from Thesis_ML.release.runner import run_release


def test_candidate_run_writes_scope_artifacts_and_manifest_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release_path = make_temp_release_bundle(tmp_path)
    monkeypatch.setattr(
        "Thesis_ML.release.runner.compile_and_run_protocol",
        fake_compile_and_run_protocol_factory(),
    )

    result = run_release(
        release_ref=release_path,
        dataset_manifest_path=dataset_manifest_path(),
        run_class=RunClass.CANDIDATE,
        run_id="candidate_scope_artifacts",
    )
    run_dir = Path(str(result["run_dir"]))
    assert (run_dir / "artifacts" / "scope" / "selected_samples.csv").exists()
    assert (run_dir / "artifacts" / "scope" / "scope_manifest.json").exists()
    assert (run_dir / "verification" / "scope_alignment_verification.json").exists()

    run_manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert bool(run_manifest["compiled_scope_manifest_path"])
    assert bool(run_manifest["selected_samples_path"])
    assert len(str(run_manifest["selected_sample_ids_sha256"])) == 64
    assert bool(run_manifest["scope_alignment_passed"]) is True
    assert bool(run_manifest["promotable"]) is True


def test_candidate_run_is_not_promotable_when_scope_alignment_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release_path = make_temp_release_bundle(tmp_path)
    run_id = "candidate_scope_alignment_fails"
    monkeypatch.setattr(
        "Thesis_ML.release.runner.compile_and_run_protocol",
        fake_compile_and_run_protocol_factory(),
    )

    def _failing_scope_verifier(**_: Any) -> dict[str, Any]:
        return {
            "passed": False,
            "issues": [{"code": "scope_sample_id_mismatch", "message": "intentional failure"}],
        }

    monkeypatch.setattr(
        "Thesis_ML.release.runner.verify_scope_execution_alignment",
        _failing_scope_verifier,
    )

    with pytest.raises(ValueError, match="Scope alignment verification failed"):
        run_release(
            release_ref=release_path,
            dataset_manifest_path=dataset_manifest_path(),
            run_class=RunClass.CANDIDATE,
            run_id=run_id,
        )

    run_dir = tmp_path / "runs" / "candidate" / "thesis_final_v1" / run_id
    run_manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert run_manifest["status"] == "failed"
    assert bool(run_manifest["promotable"]) is False
    assert bool(run_manifest["scope_alignment_passed"]) is False
