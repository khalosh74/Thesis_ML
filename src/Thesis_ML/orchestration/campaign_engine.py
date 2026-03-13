from __future__ import annotations

import json
import subprocess
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from Thesis_ML.artifacts.registry import (
    ARTIFACT_TYPE_METRICS_BUNDLE,
    compute_config_hash,
    register_artifact,
)
from Thesis_ML.orchestration.contracts import CompiledStudyManifest
from Thesis_ML.orchestration.decision_reports import (
    write_decision_reports as _write_decision_reports,
)
from Thesis_ML.orchestration.execution_bridge import execute_variant as _execute_variant
from Thesis_ML.orchestration.experiment_selection import (
    collect_dataset_scope as _collect_dataset_scope,
)
from Thesis_ML.orchestration.experiment_selection import (
    select_experiments as _select_experiments,
)
from Thesis_ML.orchestration.reporting import (
    status_snapshot as _status_snapshot,
)
from Thesis_ML.orchestration.reporting import (
    summarize_by_experiment as _summarize_by_experiment,
)
from Thesis_ML.orchestration.reporting import (
    write_experiment_outputs as _write_experiment_outputs,
)
from Thesis_ML.orchestration.reporting import (
    write_run_log_export as _write_run_log_export,
)
from Thesis_ML.orchestration.reporting import (
    write_stage_summaries as _write_stage_summaries,
)
from Thesis_ML.orchestration.result_aggregation import (
    aggregate_variant_records,
    build_summary_output_rows,
)
from Thesis_ML.orchestration.search_space import build_search_space_map
from Thesis_ML.orchestration.study_loading import (
    read_registry_manifest,
    read_workbook_manifest,
)
from Thesis_ML.orchestration.variant_expansion import (
    expand_experiment_variants as _expand_experiment_variants,
)
from Thesis_ML.orchestration.workbook_bridge import (
    build_machine_status_rows as _build_machine_status_rows,
)
from Thesis_ML.orchestration.workbook_bridge import (
    build_run_log_writeback_rows as _build_run_log_writeback_rows,
)
from Thesis_ML.orchestration.workbook_bridge import (
    build_trial_results_rows as _build_trial_results_rows,
)
from Thesis_ML.orchestration.workbook_writeback import write_workbook_results


def _now_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def _utc_timestamp() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _git_commit() -> str | None:
    try:
        process = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    commit = process.stdout.strip()
    return commit or None


def run_decision_support_campaign(
    *,
    registry_path: Path,
    index_csv: Path,
    data_root: Path,
    cache_dir: Path,
    output_root: Path,
    experiment_id: str | None,
    stage: str | None,
    run_all: bool,
    seed: int,
    n_permutations: int,
    dry_run: bool,
    subjects_filter: list[str] | None = None,
    tasks_filter: list[str] | None = None,
    modalities_filter: list[str] | None = None,
    max_runs_per_experiment: int | None = None,
    dataset_name: str = "Internal BAS2",
    registry_manifest: CompiledStudyManifest | None = None,
    write_back_to_workbook: bool = False,
    workbook_source_path: Path | None = None,
    workbook_output_dir: Path | None = None,
    append_workbook_run_log: bool = True,
    search_mode: str = "deterministic",
    optuna_trials: int | None = None,
    run_experiment_fn: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if run_experiment_fn is None:
        from Thesis_ML.experiments.run_experiment import run_experiment as _run_experiment

        run_experiment_fn = _run_experiment

    registry = (
        registry_manifest
        if registry_manifest is not None
        else read_registry_manifest(registry_path)
    )
    selected_experiments = _select_experiments(
        registry=registry,
        experiment_id=experiment_id,
        stage=stage,
        run_all=run_all,
    )
    if experiment_id and len(selected_experiments) == 1:
        selected = selected_experiments[0]
        if not bool(selected.get("executable_now", True)):
            reasons = selected.get("blocked_reasons", [])
            reason_text = "; ".join(str(reason) for reason in reasons) if reasons else "unspecified"
            raise RuntimeError(f"Experiment '{experiment_id}' is not executable now: {reason_text}")
    dataset_scope = _collect_dataset_scope(
        index_csv=index_csv,
        subjects_filter=subjects_filter,
        tasks_filter=tasks_filter,
        modalities_filter=modalities_filter,
    )

    campaign_id = _now_timestamp()
    campaign_root = output_root / "campaigns" / campaign_id
    campaign_root.mkdir(parents=True, exist_ok=False)

    commit = _git_commit()
    artifact_registry_path = output_root / "artifact_registry.sqlite3"
    search_space_map = build_search_space_map(list(registry.search_spaces))
    search_mode_value = str(search_mode).strip().lower()
    if search_mode_value not in {"deterministic", "optuna"}:
        raise ValueError("search_mode must be one of: deterministic, optuna")
    optuna_enabled = search_mode_value == "optuna"
    all_variant_records: list[dict[str, Any]] = []
    blocked_experiments: list[dict[str, Any]] = []
    experiment_roots: dict[str, str] = {}

    for experiment in selected_experiments:
        exp_id = str(experiment["experiment_id"])
        experiment_root = output_root / exp_id / campaign_id
        experiment_root.mkdir(parents=True, exist_ok=False)
        experiment_roots[exp_id] = str(experiment_root.resolve())

        variants, warnings = _expand_experiment_variants(
            experiment=experiment,
            dataset_scope=dataset_scope,
            search_space_map=search_space_map,
            search_seed=seed,
            optuna_enabled=optuna_enabled,
            optuna_trials=optuna_trials,
            max_runs_per_experiment=max_runs_per_experiment,
        )
        variant_records: list[dict[str, Any]] = []
        for variant in variants:
            record = _execute_variant(
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
                run_experiment_fn=run_experiment_fn,
                artifact_registry_path=artifact_registry_path,
                code_ref=commit,
            )
            variant_records.append(record)
            all_variant_records.append(record)

        _write_experiment_outputs(
            experiment=experiment,
            experiment_root=experiment_root,
            variant_records=variant_records,
            warnings=warnings,
        )

        if variant_records and all(row["status"] == "blocked" for row in variant_records):
            blocked_reasons = sorted(
                {
                    str(row.get("blocked_reason"))
                    for row in variant_records
                    if row.get("blocked_reason")
                }
            )
            blocked_experiments.append(
                {
                    "experiment_id": exp_id,
                    "reasons": blocked_reasons,
                }
            )

    if experiment_id:
        selected_records = [
            row
            for row in all_variant_records
            if str(row.get("experiment_id")) == str(experiment_id)
        ]
        if selected_records and all(
            str(row.get("status")) == "blocked" for row in selected_records
        ):
            reasons = sorted(
                {
                    str(row.get("blocked_reason"))
                    for row in selected_records
                    if row.get("blocked_reason")
                }
            )
            reason_text = "; ".join(reasons) if reasons else "unspecified"
            raise RuntimeError(f"Experiment '{experiment_id}' is not executable now: {reason_text}")

    stage_summary_paths = _write_stage_summaries(
        campaign_root=campaign_root,
        variant_records=all_variant_records,
    )

    summary_df = _summarize_by_experiment(
        experiments=selected_experiments,
        variant_records=all_variant_records,
    )
    decision_summary_path = campaign_root / "decision_support_summary.csv"
    summary_df.to_csv(decision_summary_path, index=False)

    run_log_path = _write_run_log_export(
        campaign_root=campaign_root,
        variant_records=all_variant_records,
        dataset_name=dataset_name,
        seed=seed,
        commit=commit,
    )

    decision_report_path, stage_decision_paths = _write_decision_reports(
        campaign_root=campaign_root,
        experiments=selected_experiments,
        variant_records=all_variant_records,
    )

    aggregation = aggregate_variant_records(all_variant_records, top_k=5)
    aggregation_path = campaign_root / "result_aggregation.json"
    aggregation_path.write_text(
        f"{json.dumps(aggregation, indent=2)}\n",
        encoding="utf-8",
    )
    summary_output_rows = build_summary_output_rows(aggregation)
    summary_output_path = campaign_root / "summary_outputs_export.csv"
    import pandas as pd

    pd.DataFrame(summary_output_rows).to_csv(summary_output_path, index=False)

    campaign_metrics_artifact = register_artifact(
        registry_path=artifact_registry_path,
        artifact_type=ARTIFACT_TYPE_METRICS_BUNDLE,
        run_id=campaign_id,
        upstream_artifact_ids=[],
        config_hash=compute_config_hash(
            {
                "campaign_id": campaign_id,
                "seed": int(seed),
                "n_permutations": int(n_permutations),
                "dry_run": bool(dry_run),
                "selected_experiments": [str(exp["experiment_id"]) for exp in selected_experiments],
            }
        ),
        code_ref=commit,
        path=decision_summary_path,
        status="created",
    )

    workbook_output_path: Path | None = None
    if write_back_to_workbook:
        if workbook_source_path is None:
            raise ValueError("write_back_to_workbook=True requires workbook_source_path.")
        machine_rows = _build_machine_status_rows(
            campaign_id=campaign_id,
            source_workbook_path=workbook_source_path,
            variant_records=all_variant_records,
        )
        trial_rows = _build_trial_results_rows(all_variant_records)
        summary_rows = list(summary_output_rows)
        run_log_rows = _build_run_log_writeback_rows(
            variant_records=all_variant_records,
            dataset_name=dataset_name,
            seed=seed,
            commit=commit,
        )
        workbook_output_path = write_workbook_results(
            source_workbook_path=workbook_source_path,
            version_tag=campaign_id,
            machine_status_rows=machine_rows,
            trial_result_rows=trial_rows,
            summary_output_rows=summary_rows,
            run_log_rows=run_log_rows,
            append_run_log=append_workbook_run_log,
            output_dir=workbook_output_dir,
        )

    campaign_manifest = {
        "campaign_id": campaign_id,
        "created_at": _utc_timestamp(),
        "registry_path": str(registry_path.resolve()),
        "selected_experiments": [str(exp["experiment_id"]) for exp in selected_experiments],
        "dataset_scope": dataset_scope,
        "seed": int(seed),
        "n_permutations": int(n_permutations),
        "dry_run": bool(dry_run),
        "search_mode": search_mode_value,
        "optuna_trials": int(optuna_trials) if optuna_trials is not None else None,
        "search_space_ids": sorted(search_space_map.keys()),
        "status_counts": _status_snapshot(all_variant_records),
        "experiment_roots": experiment_roots,
        "exports": {
            "run_log_export": str(run_log_path.resolve()),
            "decision_support_summary": str(decision_summary_path.resolve()),
            "decision_recommendations": str(decision_report_path.resolve()),
            "result_aggregation": str(aggregation_path.resolve()),
            "summary_outputs_export": str(summary_output_path.resolve()),
            "stage_summaries": [str(path.resolve()) for path in stage_summary_paths],
            "stage_decision_notes": [str(path.resolve()) for path in stage_decision_paths],
            "workbook_output_path": (
                str(workbook_output_path.resolve()) if workbook_output_path is not None else None
            ),
        },
        "artifact_registry_path": str(artifact_registry_path.resolve()),
        "campaign_metrics_artifact_id": campaign_metrics_artifact.artifact_id,
        "blocked_experiments": blocked_experiments,
    }
    manifest_path = campaign_root / "campaign_manifest.json"
    manifest_path.write_text(f"{json.dumps(campaign_manifest, indent=2)}\n", encoding="utf-8")

    return {
        "campaign_id": campaign_id,
        "campaign_root": str(campaign_root.resolve()),
        "campaign_manifest_path": str(manifest_path.resolve()),
        "run_log_export_path": str(run_log_path.resolve()),
        "decision_support_summary_path": str(decision_summary_path.resolve()),
        "decision_recommendations_path": str(decision_report_path.resolve()),
        "result_aggregation_path": str(aggregation_path.resolve()),
        "summary_outputs_export_path": str(summary_output_path.resolve()),
        "selected_experiments": [str(exp["experiment_id"]) for exp in selected_experiments],
        "status_counts": _status_snapshot(all_variant_records),
        "blocked_experiments": blocked_experiments,
        "workbook_output_path": (
            str(workbook_output_path.resolve()) if workbook_output_path is not None else None
        ),
    }


def run_workbook_decision_support_campaign(
    *,
    workbook_path: Path,
    index_csv: Path,
    data_root: Path,
    cache_dir: Path,
    output_root: Path,
    experiment_id: str | None,
    stage: str | None,
    run_all: bool,
    seed: int,
    n_permutations: int,
    dry_run: bool,
    subjects_filter: list[str] | None = None,
    tasks_filter: list[str] | None = None,
    modalities_filter: list[str] | None = None,
    max_runs_per_experiment: int | None = None,
    dataset_name: str = "Internal BAS2",
    write_back_to_workbook: bool = True,
    workbook_output_dir: Path | None = None,
    append_workbook_run_log: bool = True,
    search_mode: str = "deterministic",
    optuna_trials: int | None = None,
    run_experiment_fn: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    workbook_manifest = read_workbook_manifest(workbook_path)
    return run_decision_support_campaign(
        registry_path=workbook_path,
        index_csv=index_csv,
        data_root=data_root,
        cache_dir=cache_dir,
        output_root=output_root,
        experiment_id=experiment_id,
        stage=stage,
        run_all=run_all,
        seed=seed,
        n_permutations=n_permutations,
        dry_run=dry_run,
        subjects_filter=subjects_filter,
        tasks_filter=tasks_filter,
        modalities_filter=modalities_filter,
        max_runs_per_experiment=max_runs_per_experiment,
        dataset_name=dataset_name,
        registry_manifest=workbook_manifest,
        write_back_to_workbook=write_back_to_workbook,
        workbook_source_path=workbook_path,
        workbook_output_dir=workbook_output_dir,
        append_workbook_run_log=append_workbook_run_log,
        search_mode=search_mode,
        optuna_trials=optuna_trials,
        run_experiment_fn=run_experiment_fn,
    )


__all__ = [
    "run_decision_support_campaign",
    "run_workbook_decision_support_campaign",
]
