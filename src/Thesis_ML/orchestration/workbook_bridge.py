from __future__ import annotations

from pathlib import Path
from typing import Any

from Thesis_ML.orchestration.campaign_runner import (
    _build_machine_status_rows,
    _build_run_log_writeback_rows,
    _build_trial_results_rows,
    _status_for_machine_sheet,
)


def status_for_machine_sheet(variant_records: list[dict[str, Any]]) -> str:
    return _status_for_machine_sheet(variant_records)


def build_machine_status_rows(
    *,
    campaign_id: str,
    source_workbook_path: Path,
    variant_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return _build_machine_status_rows(
        campaign_id=campaign_id,
        source_workbook_path=source_workbook_path,
        variant_records=variant_records,
    )


def build_trial_results_rows(variant_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return _build_trial_results_rows(variant_records)


def build_run_log_writeback_rows(
    *,
    variant_records: list[dict[str, Any]],
    dataset_name: str,
    seed: int,
    commit: str | None,
) -> list[dict[str, Any]]:
    return _build_run_log_writeback_rows(
        variant_records=variant_records,
        dataset_name=dataset_name,
        seed=seed,
        commit=commit,
    )


__all__ = [
    "status_for_machine_sheet",
    "build_machine_status_rows",
    "build_trial_results_rows",
    "build_run_log_writeback_rows",
]
