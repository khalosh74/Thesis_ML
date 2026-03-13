from __future__ import annotations

from pathlib import Path
from typing import Any

from Thesis_ML.orchestration.campaign_runner import (
    _execute_variant,
    run_experiment,
)


def execute_variant(
    *,
    experiment: dict[str, Any],
    variant: dict[str, Any],
    campaign_id: str,
    experiment_root: Path,
    index_csv: Path,
    data_root: Path,
    cache_dir: Path,
    seed: int,
    n_permutations: int,
    dry_run: bool,
    artifact_registry_path: Path | None = None,
    code_ref: str | None = None,
) -> dict[str, Any]:
    return _execute_variant(
        experiment=experiment,
        variant=variant,
        campaign_id=campaign_id,
        experiment_root=experiment_root,
        index_csv=index_csv,
        data_root=data_root,
        cache_dir=cache_dir,
        seed=seed,
        n_permutations=n_permutations,
        dry_run=dry_run,
        artifact_registry_path=artifact_registry_path,
        code_ref=code_ref,
    )


__all__ = ["run_experiment", "execute_variant"]
