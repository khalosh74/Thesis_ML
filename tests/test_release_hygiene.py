from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _init_git_repo(repo_root: Path) -> None:
    subprocess.run(
        ["git", "init"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )


def _script_path() -> Path:
    return Path(__file__).resolve().parents[1] / "scripts" / "release_hygiene_check.py"


def test_release_hygiene_fails_when_governance_files_missing(tmp_path: Path) -> None:
    _init_git_repo(tmp_path)
    completed = subprocess.run(
        [sys.executable, str(_script_path()), "--repo-root", str(tmp_path)],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 1
    assert "Missing required governance files:" in completed.stdout


def test_release_hygiene_passes_when_governance_files_present(tmp_path: Path) -> None:
    _init_git_repo(tmp_path)
    (tmp_path / "docs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "LICENSE").write_text("MIT\n", encoding="utf-8")
    (tmp_path / "CITATION.cff").write_text("cff-version: 1.2.0\n", encoding="utf-8")
    (tmp_path / "docs" / "PRIVACY_AND_DATA_HANDLING.md").write_text(
        "# Privacy\n", encoding="utf-8"
    )
    (tmp_path / "docs" / "USE_AND_MISUSE_BOUNDARIES.md").write_text(
        "# Boundaries\n", encoding="utf-8"
    )
    (tmp_path / "docs" / "CONFIRMATORY_READY.md").write_text(
        "# Confirmatory Ready\n", encoding="utf-8"
    )
    (tmp_path / "docs" / "REPRODUCIBILITY.md").write_text(
        "# Reproducibility\n", encoding="utf-8"
    )

    completed = subprocess.run(
        [sys.executable, str(_script_path()), "--repo-root", str(tmp_path)],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    assert "Release hygiene check passed." in completed.stdout
