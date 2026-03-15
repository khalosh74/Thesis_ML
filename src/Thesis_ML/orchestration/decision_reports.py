from __future__ import annotations

from pathlib import Path
from typing import Any

from Thesis_ML.config.metric_policy import validate_metric_name
from Thesis_ML.orchestration.experiment_selection import STAGE_ORDER
from Thesis_ML.orchestration.reporting import STAGE_SUMMARY_FILENAMES


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def decision_text_for_experiment(
    experiment: dict[str, Any],
    rows: list[dict[str, Any]],
) -> list[str]:
    raw_primary_metric = experiment.get("primary_metric")
    if raw_primary_metric is None or not str(raw_primary_metric).strip():
        raise ValueError(
            f"Experiment '{experiment.get('experiment_id')}' is missing required primary_metric."
        )
    primary_metric = validate_metric_name(str(raw_primary_metric).strip())
    completed = [row for row in rows if row.get("status") == "completed"]
    blocked = [row for row in rows if row.get("status") == "blocked"]
    failed = [row for row in rows if row.get("status") == "failed"]
    dry_run = [row for row in rows if row.get("status") == "dry_run"]

    lines = [f"### {experiment['experiment_id']} - {experiment['title']}"]
    lines.append(f"- Decision linkage: {experiment.get('decision_id', 'n/a')}")
    lines.append(f"- Manipulated factor: {experiment.get('manipulated_factor', '')}")
    lines.append(f"- Held constant: {experiment.get('fixed_controls', '')}")
    lines.append(f"- Primary metric: {primary_metric}")
    if completed:
        metric_rows_raw = [
            (row.get("variant_id"), _safe_float(row.get("primary_metric_value")))
            for row in completed
        ]
        metric_rows: list[tuple[Any, float]] = [
            (variant_id, metric_value)
            for variant_id, metric_value in metric_rows_raw
            if metric_value is not None
        ]
        if metric_rows:
            best_variant, best_value = max(metric_rows, key=lambda pair: pair[1])
            lines.append(
                f"- Pattern favoring one option (executed variants only): "
                f"{best_variant} had best {primary_metric}={best_value:.4f}."
            )
        else:
            lines.append(
                "- Pattern favoring one option: completed runs exist, but primary metric "
                "was not available in the captured payload."
            )
    else:
        lines.append("- Pattern favoring one option: not available (no completed variants).")

    uncertainty_parts: list[str] = []
    if blocked:
        uncertainty_parts.append(f"blocked={len(blocked)}")
    if failed:
        uncertainty_parts.append(f"failed={len(failed)}")
    if dry_run:
        uncertainty_parts.append(f"dry_run={len(dry_run)}")
    if uncertainty_parts:
        lines.append("- Remaining uncertainty: " + ", ".join(uncertainty_parts))
    else:
        lines.append("- Remaining uncertainty: none flagged by orchestrator status layer.")
    lines.append("")
    return lines


def write_decision_reports(
    campaign_root: Path,
    experiments: list[dict[str, Any]],
    variant_records: list[dict[str, Any]],
) -> tuple[Path, list[Path]]:
    record_map: dict[str, list[dict[str, Any]]] = {}
    for row in variant_records:
        record_map.setdefault(str(row["experiment_id"]), []).append(row)

    stage_markdowns: list[Path] = []
    lines: list[str] = [
        "# Decision-Support Recommendations",
        "",
        "This report summarizes decision-support evidence only (E01-E11).",
        "It does not include confirmatory-stage claims.",
        "",
    ]

    for stage in STAGE_ORDER:
        stage_experiments = [exp for exp in experiments if str(exp.get("stage")) == stage]
        if not stage_experiments:
            continue
        lines.append(f"## {stage}")
        lines.append("")
        stage_lines: list[str] = [f"# {stage}", ""]
        for experiment in stage_experiments:
            exp_lines = decision_text_for_experiment(
                experiment=experiment,
                rows=record_map.get(str(experiment["experiment_id"]), []),
            )
            lines.extend(exp_lines)
            stage_lines.extend(exp_lines)

        stage_name = STAGE_SUMMARY_FILENAMES.get(stage, stage.replace(" ", "_").lower())
        stage_path = campaign_root / f"{stage_name}_decision_notes.md"
        stage_path.write_text("\n".join(stage_lines) + "\n", encoding="utf-8")
        stage_markdowns.append(stage_path)

    out_path = campaign_root / "decision_recommendations.md"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path, stage_markdowns


__all__ = ["decision_text_for_experiment", "write_decision_reports"]
