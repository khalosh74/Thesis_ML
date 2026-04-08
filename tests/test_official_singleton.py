from __future__ import annotations

from pathlib import Path

import pytest

from Thesis_ML.release.loader import load_release_bundle
from Thesis_ML.release.models import RunClass
from Thesis_ML.release.paths import list_official_runs_for_release
from Thesis_ML.release.promotion import promote_candidate_run
from Thesis_ML.release.runner import run_release
from tests._release_test_utils import (
    dataset_manifest_path,
    fake_compile_and_run_protocol_factory,
    make_temp_release_bundle,
)


def test_exactly_one_official_run_is_allowed_per_release(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release_path = make_temp_release_bundle(tmp_path)
    monkeypatch.setattr(
        "Thesis_ML.release.runner.compile_and_run_protocol",
        fake_compile_and_run_protocol_factory(),
    )

    candidate_one = run_release(
        release_ref=release_path,
        dataset_manifest_path=dataset_manifest_path(),
        run_class=RunClass.CANDIDATE,
        run_id="singleton_candidate_one",
    )
    promote_candidate_run(Path(str(candidate_one["run_dir"])))

    loaded_release = load_release_bundle(release_path)
    official_runs = list_official_runs_for_release(
        execution=loaded_release.execution,
        release_id=loaded_release.release.release_id,
    )
    assert len(official_runs) == 1

    candidate_two = run_release(
        release_ref=release_path,
        dataset_manifest_path=dataset_manifest_path(),
        run_class=RunClass.CANDIDATE,
        run_id="singleton_candidate_two",
    )
    with pytest.raises(ValueError, match="Official singleton violation"):
        promote_candidate_run(Path(str(candidate_two["run_dir"])))
