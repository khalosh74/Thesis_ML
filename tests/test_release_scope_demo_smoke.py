from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests._release_test_utils import (
    dataset_manifest_path,
    fake_compile_and_run_protocol_factory,
    make_temp_release_bundle,
)
from Thesis_ML.cli.promote_run import main as promote_run_main
from Thesis_ML.cli.release_runner import main as release_runner_main


def test_release_scope_demo_smoke_candidate_then_promotion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release_path = make_temp_release_bundle(tmp_path)
    monkeypatch.setattr(
        "Thesis_ML.release.runner.compile_and_run_protocol",
        fake_compile_and_run_protocol_factory(),
    )

    run_exit = release_runner_main(
        [
            "--release",
            str(release_path),
            "--dataset-manifest",
            str(dataset_manifest_path()),
            "--run-class",
            "candidate",
        ]
    )
    assert run_exit == 0

    candidate_root = tmp_path / "runs" / "candidate" / "thesis_final_v1"
    candidates = [path for path in candidate_root.iterdir() if path.is_dir()]
    assert len(candidates) == 1
    candidate_dir = candidates[0]

    assert (candidate_dir / "artifacts" / "scope" / "selected_samples.csv").exists()
    assert (candidate_dir / "artifacts" / "scope" / "scope_manifest.json").exists()
    scope_alignment_path = candidate_dir / "verification" / "scope_alignment_verification.json"
    assert scope_alignment_path.exists()
    scope_alignment_payload = json.loads(scope_alignment_path.read_text(encoding="utf-8"))
    assert bool(scope_alignment_payload.get("passed", False)) is True

    promote_exit = promote_run_main(["--candidate-run", str(candidate_dir)])
    assert promote_exit == 0

    official_root = tmp_path / "runs" / "official" / "thesis_final_v1"
    official_runs = [path for path in official_root.iterdir() if path.is_dir()]
    assert len(official_runs) == 1
