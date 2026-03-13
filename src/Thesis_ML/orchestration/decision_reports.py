from __future__ import annotations

from pathlib import Path
from typing import Any

from Thesis_ML.orchestration.campaign_runner import (
    _decision_text_for_experiment,
    _write_decision_reports,
)


def decision_text_for_experiment(
    experiment: dict[str, Any],
    rows: list[dict[str, Any]],
) -> list[str]:
    return _decision_text_for_experiment(experiment=experiment, rows=rows)


def write_decision_reports(
    campaign_root: Path,
    experiments: list[dict[str, Any]],
    variant_records: list[dict[str, Any]],
) -> tuple[Path, list[Path]]:
    return _write_decision_reports(
        campaign_root=campaign_root,
        experiments=experiments,
        variant_records=variant_records,
    )


__all__ = ["decision_text_for_experiment", "write_decision_reports"]
