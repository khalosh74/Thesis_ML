from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from tests._release_test_utils import (
    dataset_manifest_path,
    fake_compile_and_run_protocol_factory,
    make_temp_release_bundle,
    repo_root,
)
from Thesis_ML.protocols.compiler import compile_protocol as _compile_protocol
from Thesis_ML.release.models import RunClass
from Thesis_ML.release.runner import run_release


def _write_modified_dataset_manifest(
    *,
    tmp_path: Path,
    task_value: str | None = None,
    modality_value: str | None = None,
) -> Path:
    source_manifest = json.loads(dataset_manifest_path().read_text(encoding="utf-8"))
    source_index_path = (repo_root() / "demo_data" / "synthetic_v1" / source_manifest["index_csv"]).resolve()
    frame = pd.read_csv(source_index_path)

    if task_value is not None:
        frame["task"] = task_value
    if modality_value is not None:
        frame["modality"] = modality_value

    index_path = tmp_path / "modified_release_index.csv"
    frame.to_csv(index_path, index=False)

    source_manifest["index_csv"] = str(index_path.resolve())
    source_manifest["data_root"] = str((repo_root() / "demo_data" / "synthetic_v1" / "data_root").resolve())
    source_manifest["cache_dir"] = str((repo_root() / "demo_data" / "synthetic_v1" / "cache").resolve())
    manifest_path = tmp_path / "dataset_manifest_modified.json"
    manifest_path.write_text(f"{json.dumps(source_manifest, indent=2)}\n", encoding="utf-8")
    return manifest_path


def test_release_scope_task_mismatch_is_caught(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release_path = make_temp_release_bundle(tmp_path)
    bad_manifest = _write_modified_dataset_manifest(tmp_path=tmp_path, task_value="passive")

    monkeypatch.setattr(
        "Thesis_ML.release.runner.compile_and_run_protocol",
        fake_compile_and_run_protocol_factory(),
    )

    with pytest.raises(ValueError, match="empty selection"):
        run_release(
            release_ref=release_path,
            dataset_manifest_path=bad_manifest,
            run_class=RunClass.CANDIDATE,
            run_id="scope_task_mismatch",
        )


def test_release_scope_modality_mismatch_is_caught(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release_path = make_temp_release_bundle(tmp_path)
    bad_manifest = _write_modified_dataset_manifest(tmp_path=tmp_path, modality_value="audio")

    monkeypatch.setattr(
        "Thesis_ML.release.runner.compile_and_run_protocol",
        fake_compile_and_run_protocol_factory(),
    )

    with pytest.raises(ValueError, match="empty selection"):
        run_release(
            release_ref=release_path,
            dataset_manifest_path=bad_manifest,
            run_class=RunClass.CANDIDATE,
            run_id="scope_modality_mismatch",
        )


@pytest.mark.parametrize(
    ("config_overrides", "expected_issue"),
    [
        ({"feature_space": "roi_masked"}, "feature_space_alignment_failed"),
        ({"model_governance": {"logical_model_name": "logreg"}}, "model_family_alignment_failed"),
        ({"tuning_enabled": True}, "tuning_alignment_failed"),
    ],
)
def test_release_runtime_alignment_catches_feature_model_and_tuning_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    config_overrides: dict[str, object],
    expected_issue: str,
) -> None:
    release_path = make_temp_release_bundle(tmp_path)
    run_id = f"alignment_{expected_issue}"
    monkeypatch.setattr(
        "Thesis_ML.release.runner.compile_and_run_protocol",
        fake_compile_and_run_protocol_factory(config_overrides=config_overrides),
    )

    with pytest.raises(ValueError, match="Release evidence verification failed"):
        run_release(
            release_ref=release_path,
            dataset_manifest_path=dataset_manifest_path(),
            run_class=RunClass.CANDIDATE,
            run_id=run_id,
        )

    verification_path = (
        tmp_path
        / "runs"
        / "candidate"
        / "thesis_final_v1"
        / run_id
        / "verification"
        / "evidence_verification.json"
    )
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    assert verification["passed"] is False
    assert any(str(issue.get("code")) == expected_issue for issue in verification["issues"])


def test_release_scope_enforcement_rejects_compiled_runtime_task_filter(
    tmp_path: Path,
) -> None:
    release_path = make_temp_release_bundle(tmp_path)

    def _compile_protocol_with_runtime_filter(*args: Any, **kwargs: Any) -> Any:
        compiled = _compile_protocol(*args, **kwargs)
        first = compiled.runs[0].model_copy(update={"filter_task": "emo"})
        return compiled.model_copy(update={"runs": [first, *compiled.runs[1:]]})

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            "Thesis_ML.release.adapter.compile_protocol",
            _compile_protocol_with_runtime_filter,
        )
        with pytest.raises(
            ValueError,
            match="release scope enforcement requires filter_task=None",
        ):
            run_release(
                release_ref=release_path,
                dataset_manifest_path=dataset_manifest_path(),
                run_class=RunClass.CANDIDATE,
                run_id="reject_runtime_filter_authority",
            )
