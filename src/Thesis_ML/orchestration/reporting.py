from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from Thesis_ML.orchestration.campaign_runner import (
    _status_snapshot,
    _summarize_by_experiment,
    _write_experiment_outputs,
    _write_run_log_export,
    _write_stage_summaries,
)


def write_experiment_outputs(
    experiment: dict[str, Any],
    experiment_root: Path,
    variant_records: list[dict[str, Any]],
    warnings: list[str],
) -> None:
    _write_experiment_outputs(
        experiment=experiment,
        experiment_root=experiment_root,
        variant_records=variant_records,
        warnings=warnings,
    )


def summarize_by_experiment(
    experiments: list[dict[str, Any]],
    variant_records: list[dict[str, Any]],
) -> pd.DataFrame:
    return _summarize_by_experiment(
        experiments=experiments,
        variant_records=variant_records,
    )


def write_stage_summaries(
    campaign_root: Path,
    variant_records: list[dict[str, Any]],
) -> list[Path]:
    return _write_stage_summaries(
        campaign_root=campaign_root,
        variant_records=variant_records,
    )


def write_run_log_export(
    campaign_root: Path,
    variant_records: list[dict[str, Any]],
    dataset_name: str,
    seed: int,
    commit: str | None,
) -> Path:
    return _write_run_log_export(
        campaign_root=campaign_root,
        variant_records=variant_records,
        dataset_name=dataset_name,
        seed=seed,
        commit=commit,
    )


def status_snapshot(records: list[dict[str, Any]]) -> dict[str, int]:
    return _status_snapshot(records)


__all__ = [
    "write_experiment_outputs",
    "summarize_by_experiment",
    "write_stage_summaries",
    "write_run_log_export",
    "status_snapshot",
]
