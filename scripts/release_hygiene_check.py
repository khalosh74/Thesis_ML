from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
NOISE_PATTERNS = (
    "__pycache__/",
    ".pytest_cache/",
    ".ruff_cache/",
    "build/",
    "dist/",
    ".mypy_cache/",
)
SKIP_TOP_LEVEL = {".venv", ".venv-uv", ".venv-gpu "venv", "Data", "data"}
REQUIRED_GOVERNANCE_FILES = (
    "LICENSE",
    "CITATION.cff",
    "docs/PRIVACY_AND_DATA_HANDLING.md",
    "docs/USE_AND_MISUSE_BOUNDARIES.md",
    "docs/CONFIRMATORY_READY.md",
    "docs/REPRODUCIBILITY.md",
)


def _tracked_noise_files(repo_root: Path) -> list[str]:
    cmd = ["git", "ls-files"]
    result = subprocess.run(
        cmd,
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    tracked = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return [
        path
        for path in tracked
        if any(pattern in path.replace("\\", "/") for pattern in NOISE_PATTERNS)
    ]


def _working_tree_noise(repo_root: Path) -> list[str]:
    noise_hits: list[str] = []
    for pattern in ("**/__pycache__", ".pytest_cache", ".ruff_cache", "build", "dist"):
        for path in repo_root.glob(pattern):
            if not path.exists():
                continue
            relative = path.relative_to(repo_root)
            if relative.parts and relative.parts[0] in SKIP_TOP_LEVEL:
                continue
            noise_hits.append(str(relative))
    return sorted(set(noise_hits))


def _missing_required_governance_files(repo_root: Path) -> list[str]:
    missing: list[str] = []
    for relative in REQUIRED_GOVERNANCE_FILES:
        candidate = repo_root / relative
        if not candidate.exists() or not candidate.is_file():
            missing.append(relative)
    return missing


def _print_section(title: str, rows: list[str]) -> None:
    print(title)
    if not rows:
        print("  - none")
        return
    for row in rows:
        print(f"  - {row}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Checks release hygiene (tracked noise + local build/cache directories)."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help="Repository root (defaults to this script's parent repo).",
    )
    args = parser.parse_args(argv)
    repo_root = args.repo_root.resolve()

    tracked_noise = _tracked_noise_files(repo_root)
    working_tree_noise = _working_tree_noise(repo_root)
    missing_governance = _missing_required_governance_files(repo_root)

    _print_section("Tracked noise files:", tracked_noise)
    _print_section("Working tree noise directories:", working_tree_noise)
    _print_section("Missing required governance files:", missing_governance)

    if tracked_noise:
        print(
            "\nRelease hygiene failed: remove tracked noise files before cutting a release candidate.",
            file=sys.stderr,
        )
        return 1
    if missing_governance:
        print(
            (
                "\nRelease hygiene failed: required governance metadata/docs are missing; "
                "add the missing files before cutting a release candidate."
            ),
            file=sys.stderr,
        )
        return 1

    print("\nRelease hygiene check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
