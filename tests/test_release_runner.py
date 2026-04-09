from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from tests._release_test_utils import (
    dataset_manifest_path,
    fake_compile_and_run_protocol_factory,
    make_temp_release_bundle,
)
from Thesis_ML.cli.release_runner import main as release_runner_main
from Thesis_ML.release.models import RunClass
from Thesis_ML.release.runner import run_release


def test_release_runner_allows_scratch_exploratory_and_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release_path = make_temp_release_bundle(tmp_path)
    monkeypatch.setattr(
        "Thesis_ML.release.runner.compile_and_run_protocol",
        fake_compile_and_run_protocol_factory(),
    )

    for run_class in (RunClass.SCRATCH, RunClass.EXPLORATORY, RunClass.CANDIDATE):
        result = run_release(
            release_ref=release_path,
            dataset_manifest_path=dataset_manifest_path(),
            run_class=run_class,
            run_id=f"{run_class.value}_runner_test",
        )
        assert result["passed"] is True
        assert result["run_class"] == run_class.value
        run_dir = Path(str(result["run_dir"]))
        assert (run_dir / "run_manifest.json").exists()


def test_release_runner_rejects_official_run_class(tmp_path: Path) -> None:
    release_path = make_temp_release_bundle(tmp_path)

    with pytest.raises(ValueError, match="scratch/exploratory/candidate"):
        run_release(
            release_ref=release_path,
            dataset_manifest_path=dataset_manifest_path(),
            run_class=RunClass.OFFICIAL,
            run_id="official_should_fail",
        )


def test_candidate_run_writes_manifests_and_verification(
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
        run_id="candidate_manifest_test",
    )
    run_dir = Path(str(result["run_dir"]))

    assert (run_dir / "run_manifest.json").exists()
    assert (run_dir / "release_manifest.json").exists()
    assert (run_dir / "release_summary.json").exists()
    assert (run_dir / "dataset_snapshot.json").exists()
    assert (run_dir / "evidence_verification.json").exists()
    assert (run_dir / "verification" / "evidence_verification.json").exists()
    assert (run_dir / "artifacts" / "scope" / "selected_samples.csv").exists()
    assert (run_dir / "artifacts" / "scope" / "scope_manifest.json").exists()
    assert (run_dir / "verification" / "scope_alignment_verification.json").exists()

    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "succeeded"
    assert manifest["promotable"] is True
    assert manifest["scope_alignment_passed"] is True


def test_release_runner_cli_returns_nonzero_on_evidence_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release_path = make_temp_release_bundle(tmp_path)
    monkeypatch.setattr(
        "Thesis_ML.release.runner.compile_and_run_protocol",
        fake_compile_and_run_protocol_factory(missing_run_artifact="metrics.json"),
    )

    exit_code = release_runner_main(
        [
            "--release",
            str(release_path),
            "--dataset-manifest",
            str(dataset_manifest_path()),
            "--run-class",
            "candidate",
        ]
    )
    assert exit_code == 1


def test_release_runner_cli_verbose_and_live_progress_emit_stderr(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    release_path = make_temp_release_bundle(tmp_path)

    def _fake_run_release(**kwargs: object) -> dict[str, object]:
        callback = kwargs.get("event_callback")
        assert callable(callback)
        typed_callback = callback
        assert isinstance(typed_callback, Callable)
        typed_callback(
            {
                "event": "runtime_protocol.execute_start",
                "n_runs": 2,
            }
        )
        typed_callback(
            {
                "event": "parallel_execution.run_completed",
                "run_id": "run_a",
                "status": "success",
            }
        )
        typed_callback(
            {
                "event": "parallel_execution.run_completed",
                "run_id": "run_b",
                "status": "failed",
            }
        )
        return {
            "passed": True,
            "run_id": "candidate_test",
            "run_class": "candidate",
            "run_dir": str(tmp_path / "runs" / "candidate" / "candidate_test"),
            "promotable": False,
            "official": False,
            "release_id": "thesis_final_v1",
            "release_version": "1.0.0",
        }

    monkeypatch.setattr("Thesis_ML.cli.release_runner.run_release", _fake_run_release)

    exit_code = release_runner_main(
        [
            "--release",
            str(release_path),
            "--dataset-manifest",
            str(dataset_manifest_path()),
            "--run-class",
            "candidate",
            "--verbose",
            "--live-progress",
        ]
    )
    assert exit_code == 0
    captured = capsys.readouterr()
    assert '"passed": true' in captured.out.lower()
    assert "[thesisml]" in captured.err
    assert "[thesisml-live] completed=2/2" in captured.err
