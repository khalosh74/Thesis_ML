from __future__ import annotations

from typing import Any

from Thesis_ML.orchestration.result_aggregation_core import (
    safe_float,
    xai_method_for_model,
)

SUMMARY_COLUMNS = [
    "summary_type",
    "summary_key",
    "primary_metric_name",
    "primary_metric_value",
    "run_id",
    "experiment_id",
    "start_section",
    "end_section",
    "model",
    "cv",
    "target",
    "xai_method",
    "report_path",
    "notes",
]


def _base_summary_row(
    *,
    summary_type: str,
    summary_key: str,
    primary_metric_name: str,
    primary_metric_value: float | None,
    run_id: Any = None,
    experiment_id: Any = None,
    start_section: Any = None,
    end_section: Any = None,
    model: Any = None,
    cv: Any = None,
    target: Any = None,
    xai_method: Any = None,
    report_path: Any = None,
    notes: str = "",
) -> dict[str, Any]:
    row = {
        "summary_type": summary_type,
        "summary_key": summary_key,
        "primary_metric_name": primary_metric_name,
        "primary_metric_value": primary_metric_value,
        "run_id": run_id,
        "experiment_id": experiment_id,
        "start_section": start_section,
        "end_section": end_section,
        "model": model,
        "cv": cv,
        "target": target,
        "xai_method": xai_method,
        "report_path": report_path,
        "notes": notes,
    }
    return {column: row.get(column) for column in SUMMARY_COLUMNS}


def build_summary_output_rows(aggregation: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for rank, row in enumerate(aggregation.get("best_full_pipeline_runs", []), start=1):
        rows.append(
            _base_summary_row(
                summary_type="best_full_pipeline",
                summary_key=f"rank_{rank}",
                primary_metric_name=str(row.get("primary_metric_name")),
                primary_metric_value=safe_float(row.get("primary_metric_value_float")),
                run_id=row.get("run_id"),
                experiment_id=row.get("experiment_id"),
                start_section=row.get("start_section_norm"),
                end_section=row.get("end_section_norm"),
                model=row.get("model"),
                cv=row.get("cv"),
                target=row.get("target"),
                xai_method=row.get("xai_method"),
                report_path=row.get("report_dir"),
                notes=f"xai_status={row.get('xai_status', 'unknown')}",
            )
        )

    for rank, row in enumerate(aggregation.get("best_segment_runs", []), start=1):
        rows.append(
            _base_summary_row(
                summary_type="best_segment_run",
                summary_key=f"rank_{rank}",
                primary_metric_name=str(row.get("primary_metric_name")),
                primary_metric_value=safe_float(row.get("primary_metric_value_float")),
                run_id=row.get("run_id"),
                experiment_id=row.get("experiment_id"),
                start_section=row.get("start_section_norm"),
                end_section=row.get("end_section_norm"),
                model=row.get("model"),
                cv=row.get("cv"),
                target=row.get("target"),
                xai_method=row.get("xai_method"),
                report_path=row.get("report_dir"),
                notes=f"xai_status={row.get('xai_status', 'unknown')}",
            )
        )

    for row in aggregation.get("section_level_effects", []):
        rows.append(
            _base_summary_row(
                summary_type="section_effect",
                summary_key=str(row.get("section_key") or ""),
                primary_metric_name=str(row.get("primary_metric_name")),
                primary_metric_value=safe_float(row.get("mean_primary_metric_value")),
                run_id=row.get("best_run_id"),
                experiment_id=row.get("best_experiment_id"),
                start_section=row.get("start_section"),
                end_section=row.get("end_section"),
                model=row.get("best_model"),
                cv=row.get("best_cv"),
                target=row.get("best_target"),
                xai_method=xai_method_for_model(row.get("best_model")),
                report_path=row.get("best_report_path"),
                notes=(
                    f"n_runs={row.get('n_runs')}; "
                    f"best_primary_metric={row.get('best_primary_metric_value')}"
                ),
            )
        )

    patterns = aggregation.get("cross_run_patterns", {})
    for row in patterns.get("by_model", []):
        rows.append(
            _base_summary_row(
                summary_type="pattern_by_model",
                summary_key=str(row.get("key") or ""),
                primary_metric_name=str(row.get("primary_metric_name")),
                primary_metric_value=safe_float(row.get("mean_primary_metric_value")),
                run_id=row.get("best_run_id"),
                experiment_id=row.get("best_experiment_id"),
                model=row.get("key"),
                xai_method=xai_method_for_model(row.get("key")),
                notes=(
                    f"n_runs={row.get('n_runs')}; "
                    f"best_primary_metric={row.get('best_primary_metric_value')}"
                ),
            )
        )
    for row in patterns.get("by_cv", []):
        rows.append(
            _base_summary_row(
                summary_type="pattern_by_cv",
                summary_key=str(row.get("key") or ""),
                primary_metric_name=str(row.get("primary_metric_name")),
                primary_metric_value=safe_float(row.get("mean_primary_metric_value")),
                run_id=row.get("best_run_id"),
                experiment_id=row.get("best_experiment_id"),
                cv=row.get("key"),
                notes=(
                    f"n_runs={row.get('n_runs')}; "
                    f"best_primary_metric={row.get('best_primary_metric_value')}"
                ),
            )
        )

    xai = aggregation.get("xai", {})
    for row in xai.get("method_effects", []):
        rows.append(
            _base_summary_row(
                summary_type="xai_method_effect",
                summary_key=str(row.get("xai_method") or ""),
                primary_metric_name=str(row.get("primary_metric_name")),
                primary_metric_value=safe_float(row.get("mean_primary_metric_value")),
                run_id=row.get("best_run_id"),
                experiment_id=row.get("best_experiment_id"),
                xai_method=row.get("xai_method"),
                notes=(
                    f"n_runs={row.get('n_runs')}; performed={row.get('performed_count')}; "
                    f"not_performed={row.get('not_performed_count')}; unknown={row.get('unknown_count')}"
                ),
            )
        )

    factorial = aggregation.get("factorial", {})
    for row in factorial.get("best_by_study", []):
        rows.append(
            _base_summary_row(
                summary_type="factorial_best_by_study",
                summary_key=str(row.get("study_id") or ""),
                primary_metric_name=str(row.get("primary_metric_name")),
                primary_metric_value=safe_float(row.get("mean_primary_metric_value")),
                run_id=row.get("best_run_id"),
                experiment_id=row.get("best_experiment_id"),
                notes=(
                    f"study_id={row.get('study_id')}; best_trial_id={row.get('best_trial_id')}; "
                    "descriptive_only=true"
                ),
            )
        )
    for row in factorial.get("by_factor_level", []):
        rows.append(
            _base_summary_row(
                summary_type="factor_level_effect_descriptive",
                summary_key=str(row.get("summary_key") or ""),
                primary_metric_name=str(row.get("primary_metric_name")),
                primary_metric_value=safe_float(row.get("mean_primary_metric_value")),
                notes=(
                    f"factor_level_key={row.get('factor_level_key')}; "
                    f"best_trial_id={row.get('best_trial_id')}; descriptive_only=true"
                ),
            )
        )
    for row in factorial.get("by_factor_combination", []):
        rows.append(
            _base_summary_row(
                summary_type="factor_combination_effect_descriptive",
                summary_key=str(row.get("summary_key") or ""),
                primary_metric_name=str(row.get("primary_metric_name")),
                primary_metric_value=safe_float(row.get("mean_primary_metric_value")),
                notes=(
                    f"factor_level_key={row.get('factor_level_key')}; "
                    f"best_trial_id={row.get('best_trial_id')}; descriptive_only=true"
                ),
            )
        )
    for row in factorial.get("interaction_descriptive", []):
        rows.append(
            _base_summary_row(
                summary_type="factor_interaction_descriptive",
                summary_key=str(row.get("summary_key") or ""),
                primary_metric_name=str(row.get("primary_metric_name")),
                primary_metric_value=safe_float(row.get("mean_primary_metric_value")),
                notes=(
                    f"interaction_key={row.get('interaction_key')}; "
                    f"best_trial_id={row.get('best_trial_id')}; descriptive_only=true"
                ),
            )
        )

    return rows


__all__ = ["SUMMARY_COLUMNS", "build_summary_output_rows"]
