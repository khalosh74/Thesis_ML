from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from Thesis_ML.config.paths import (
    DEFAULT_RELEASE_REGISTRY_PATH,
    PROJECT_ROOT,
)
from Thesis_ML.release.models import ReleaseExecution, ReleaseRegistry, RunClass


@dataclass(frozen=True)
class RunPaths:
    root: Path
    logs_dir: Path
    artifacts_dir: Path
    verification_dir: Path


def _load_release_registry(path: Path | str = DEFAULT_RELEASE_REGISTRY_PATH) -> ReleaseRegistry:
    registry_path = Path(path).resolve()
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    return ReleaseRegistry.model_validate(payload)


def resolve_release_path(release_ref: Path | str) -> Path:
    candidate = Path(str(release_ref))
    if candidate.exists():
        return candidate.resolve()

    if not candidate.is_absolute():
        relative_to_repo = (PROJECT_ROOT / candidate).resolve()
        if relative_to_repo.exists():
            return relative_to_repo

    registry = _load_release_registry()
    alias_key = str(release_ref).strip()
    mapped = registry.aliases.get(alias_key)
    if mapped is None:
        raise FileNotFoundError(
            f"Unknown release reference '{release_ref}'. Not a path and not a registered alias."
        )
    mapped_path = (PROJECT_ROOT / mapped).resolve()
    if not mapped_path.exists():
        raise FileNotFoundError(
            f"Release alias '{alias_key}' points to a missing file: {mapped_path}"
        )
    return mapped_path


def resolve_run_root(execution: ReleaseExecution, run_class: RunClass) -> Path:
    if run_class == RunClass.SCRATCH:
        return (PROJECT_ROOT / execution.scratch_root).resolve()
    if run_class == RunClass.EXPLORATORY:
        return (PROJECT_ROOT / execution.exploratory_root).resolve()
    if run_class == RunClass.CANDIDATE:
        return (PROJECT_ROOT / execution.candidate_root).resolve()
    if run_class == RunClass.OFFICIAL:
        return (PROJECT_ROOT / execution.official_root).resolve()
    raise ValueError(f"Unsupported run_class '{run_class.value}'")


def resolve_run_paths(
    *,
    execution: ReleaseExecution,
    run_class: RunClass,
    release_id: str,
    run_id: str,
) -> RunPaths:
    class_root = resolve_run_root(execution, run_class)
    root = class_root / release_id / run_id
    return RunPaths(
        root=root,
        logs_dir=root / "logs",
        artifacts_dir=root / "artifacts",
        verification_dir=root / "verification",
    )


def ensure_run_directories(paths: RunPaths) -> None:
    paths.root.mkdir(parents=True, exist_ok=True)
    paths.logs_dir.mkdir(parents=True, exist_ok=True)
    paths.artifacts_dir.mkdir(parents=True, exist_ok=True)
    paths.verification_dir.mkdir(parents=True, exist_ok=True)


def find_candidate_run_by_id(candidate_run: str) -> Path:
    candidate_input = Path(candidate_run)
    if candidate_input.exists():
        return candidate_input.resolve()

    candidate_root = (PROJECT_ROOT / "runs" / "candidate").resolve()
    if not candidate_root.exists():
        raise FileNotFoundError("No candidate runs root found under runs/candidate.")

    matches = list(candidate_root.glob(f"*/{candidate_run}"))
    if len(matches) == 1:
        return matches[0].resolve()
    if len(matches) > 1:
        raise ValueError(
            f"Candidate run id '{candidate_run}' is ambiguous ({len(matches)} matches). "
            "Pass a full path instead."
        )
    raise FileNotFoundError(f"Candidate run '{candidate_run}' was not found.")


def list_official_runs_for_release(*, execution: ReleaseExecution, release_id: str) -> list[Path]:
    official_release_root = resolve_run_root(execution, RunClass.OFFICIAL) / release_id
    if not official_release_root.exists():
        return []
    return sorted(path for path in official_release_root.iterdir() if path.is_dir())


__all__ = [
    "RunPaths",
    "ensure_run_directories",
    "find_candidate_run_by_id",
    "list_official_runs_for_release",
    "resolve_release_path",
    "resolve_run_paths",
    "resolve_run_root",
]

