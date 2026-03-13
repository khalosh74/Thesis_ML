from __future__ import annotations

from pathlib import Path
from typing import Any

from Thesis_ML.orchestration.campaign_runner import (
    _collect_dataset_scope,
    _experiment_sort_key,
    _select_experiments,
    _stage_sort_key,
)
from Thesis_ML.orchestration.contracts import CompiledStudyManifest


def collect_dataset_scope(
    index_csv: Path,
    subjects_filter: list[str] | None = None,
    tasks_filter: list[str] | None = None,
    modalities_filter: list[str] | None = None,
) -> dict[str, Any]:
    return _collect_dataset_scope(
        index_csv=index_csv,
        subjects_filter=subjects_filter,
        tasks_filter=tasks_filter,
        modalities_filter=modalities_filter,
    )


def select_experiments(
    registry: CompiledStudyManifest,
    experiment_id: str | None,
    stage: str | None,
    run_all: bool,
) -> list[dict[str, Any]]:
    return _select_experiments(
        registry=registry,
        experiment_id=experiment_id,
        stage=stage,
        run_all=run_all,
    )


__all__ = ["collect_dataset_scope", "select_experiments", "_experiment_sort_key", "_stage_sort_key"]
