from __future__ import annotations

from pathlib import Path

from Thesis_ML.release.paths import resolve_release_path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_release_alias_resolves_to_expected_release_bundle() -> None:
    resolved = resolve_release_path("release.thesis_final_current")
    expected = (_repo_root() / "releases" / "thesis_final_v1" / "release.json").resolve()
    assert resolved == expected
