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
from Thesis_ML.config.framework_mode import FrameworkMode
from Thesis_ML.experiments.compute_policy import resolve_compute_policy
from Thesis_ML.experiments.compute_scheduler import (
    ComputeRunAssignment,
    ComputeRunRequest,
    plan_compute_schedule,
)
from Thesis_ML.orchestration.contracts import CompiledStudyManifest
from Thesis_ML.orchestration.decision_reports import (
    write_decision_reports as _write_decision_reports,
)
from Thesis_ML.orchestration.execution_bridge import (
    build_variant_official_job as _build_variant_official_job,
)
from Thesis_ML.orchestration.execution_bridge import (
    execute_official_jobs as _execute_official_jobs,
)
from Thesis_ML.orchestration.execution_bridge import execute_variant as _execute_variant
from Thesis_ML.orchestration.execution_bridge import (
    resolve_variant_run_id as _resolve_variant_run_id,
)
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
from Thesis_ML.orchestration.variant_expansion import (
    materialize_experiment_cells as _materialize_experiment_cells,
)
from Thesis_ML.orchestration.workbook_bridge import (
    build_effect_summary_rows as _build_effect_summary_rows,
)
from Thesis_ML.orchestration.workbook_bridge import (
    build_generated_design_rows as _build_generated_design_rows,
)
from Thesis_ML.orchestration.workbook_bridge import (
    build_machine_status_rows as _build_machine_status_rows,
)
from Thesis_ML.orchestration.workbook_bridge import (
    build_run_log_writeback_rows as _build_run_log_writeback_rows,
)
from Thesis_ML.orchestration.workbook_bridge import (
    build_study_review_rows as _build_study_review_rows,
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


_STUDY_GUARDRAIL_POLICY = {
    "exploratory": {
        "core_fields_required": ["question", "generalization_claim", "primary_metric", "cv_scheme"],
        "non_core_gaps": "warnings",
        "disposition_when_core_present": ["allowed", "warning"],
    },
    "confirmatory": {
        "core_fields_required": ["question", "generalization_claim", "primary_metric", "cv_scheme"],
        "strict_requirements": [
            "leakage_risk_reviewed",
            "unit_of_analysis_defined",
            "data_hierarchy_defined",
            "primary_contrast",
            "interpretation_rules",
            "confirmatory_lock_applied",
            "multiplicity_handling",
        ],
        "non_compliance": "blocked",
    },
}

_PHASE_BLUEPRINT_AUTO: list[dict[str, Any]] = [
    {"phase_name": "Preflight", "groups": []},
    {"phase_name": "Stage 1 target/scope lock", "groups": [["E01"], ["E02", "E03"]]},
    {"phase_name": "Stage 2 split/transfer lock", "groups": [["E04"], ["E05"]]},
    {"phase_name": "Stage 3 model lock", "groups": [["E06"], ["E07"], ["E08"]]},
    {"phase_name": "Stage 4 representation/preprocessing lock", "groups": [["E09"], ["E10"], ["E11"]]},
    {"phase_name": "Freeze final confirmatory pipeline", "groups": []},
    {"phase_name": "Confirmatory", "groups": [["E16", "E17"]]},
    {"phase_name": "Blocking robustness", "groups": [["E12", "E13", "E20"]]},
    {"phase_name": "Context robustness", "groups": [["E21", "E22", "E23", "E15"]]},
    {"phase_name": "Reproducibility audit", "groups": [["E24"]]},
]

_PHASE_ARTIFACTS: dict[str, str] = {
    "Stage 1 target/scope lock": "stage1_lock.json",
    "Stage 2 split/transfer lock": "stage2_lock.json",
    "Stage 3 model lock": "stage3_lock.json",
    "Stage 4 representation/preprocessing lock": "stage4_lock.json",
    "Freeze final confirmatory pipeline": "final_confirmatory_pipeline.json",
}


def _resolve_phase_plan(phase_plan: str) -> str:
    resolved = str(phase_plan or "auto").strip().lower()
    if resolved not in {"auto", "flat"}:
        raise ValueError("phase_plan must be one of: auto, flat")
    return resolved


def _build_phase_batches(
    *,
    selected_experiments: list[dict[str, Any]],
    phase_plan: str,
) -> list[dict[str, Any]]:
    selected_by_id = {str(exp["experiment_id"]): exp for exp in selected_experiments}
    if phase_plan == "flat":
        groups = [[selected_by_id[key]] for key in selected_by_id]
        return [
            {
                "phase_name": "Flat selected sequence",
                "phase_order_index": 0,
                "groups": groups,
                "expected_experiment_ids": sorted(selected_by_id.keys()),
            }
        ]

    phases: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, entry in enumerate(_PHASE_BLUEPRINT_AUTO):
        phase_name = str(entry["phase_name"])
        expected_ids: list[str] = []
        groups: list[list[dict[str, Any]]] = []
        for group_ids in entry["groups"]:
            present_group: list[dict[str, Any]] = []
            for experiment_id in group_ids:
                expected_ids.append(str(experiment_id))
                experiment = selected_by_id.get(str(experiment_id))
                if experiment is None:
                    continue
                present_group.append(experiment)
                seen_ids.add(str(experiment_id))
            if present_group:
                groups.append(present_group)
        phases.append(
            {
                "phase_name": phase_name,
                "phase_order_index": int(index),
                "groups": groups,
                "expected_experiment_ids": expected_ids,
            }
        )

    remaining = [exp for exp in selected_experiments if str(exp["experiment_id"]) not in seen_ids]
    if remaining:
        phases.append(
            {
                "phase_name": "Unmapped selected experiments",
                "phase_order_index": len(phases),
                "groups": [[exp] for exp in remaining],
                "expected_experiment_ids": [str(exp["experiment_id"]) for exp in remaining],
            }
        )
    return phases


def _coerce_experiment_ids(group: list[dict[str, Any]]) -> list[str]:
    return [str(exp["experiment_id"]) for exp in group]


def _phase_status_from_records(records: list[dict[str, Any]]) -> str:
    if not records:
        return "no_runs"
    statuses = {str(record.get("status")) for record in records}
    if statuses.issubset({"dry_run"}):
        return "dry_run"
    if statuses.issubset({"completed"}):
        return "completed"
    if statuses.issubset({"blocked"}):
        return "blocked"
    if "failed" in statuses:
        return "failed"
    if "completed" in statuses:
        return "partial"
    return "mixed"


def _write_phase_artifact(
    *,
    campaign_root: Path,
    filename: str,
    payload: dict[str, Any],
) -> Path:
    output_path = campaign_root / filename
    output_path.write_text(f"{json.dumps(payload, indent=2)}\n", encoding="utf-8")
    return output_path


def _is_sequential_only_group(
    *,
    group_experiment_ids: list[str],
    cells: list[dict[str, Any]],
) -> bool:
    if "E24" in group_experiment_ids:
        return True
    for cell in cells:
        if bool(cell.get("sequential_only")):
            return True
        design_metadata = cell.get("design_metadata")
        if isinstance(design_metadata, dict) and bool(design_metadata.get("sequential_only")):
            return True
    return False


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
    max_parallel_runs: int = 1,
    max_parallel_gpu_runs: int = 1,
    hardware_mode: str = "cpu_only",
    gpu_device_id: int | None = None,
    deterministic_compute: bool = False,
    allow_backend_fallback: bool = False,
    phase_plan: str = "auto",
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

    study_review_summary_path = campaign_root / "study_review_summary.json"
    study_reviews_payload = [review.model_dump(mode="python") for review in registry.study_reviews]
    study_review_summary = {
        "generated_at": _utc_timestamp(),
        "guardrail_policy": _STUDY_GUARDRAIL_POLICY,
        "studies": study_reviews_payload,
    }
    study_review_summary_path.write_text(
        f"{json.dumps(study_review_summary, indent=2)}\n",
        encoding="utf-8",
    )

    commit = _git_commit()
    artifact_registry_path = output_root / "artifact_registry.sqlite3"
    search_space_map = build_search_space_map(list(registry.search_spaces))
    search_mode_value = str(search_mode).strip().lower()
    if search_mode_value not in {"deterministic", "optuna"}:
        raise ValueError("search_mode must be one of: deterministic, optuna")
    if int(max_parallel_runs) <= 0:
        raise ValueError("max_parallel_runs must be >= 1.")
    if int(max_parallel_gpu_runs) < 0:
        raise ValueError("max_parallel_gpu_runs must be >= 0.")
    if int(max_parallel_gpu_runs) > int(max_parallel_runs):
        raise ValueError("max_parallel_gpu_runs cannot exceed max_parallel_runs.")
    phase_plan_value = _resolve_phase_plan(phase_plan)
    optuna_enabled = search_mode_value == "optuna"
    base_compute_policy = resolve_compute_policy(
        framework_mode=FrameworkMode.EXPLORATORY,
        hardware_mode=hardware_mode,
        gpu_device_id=gpu_device_id,
        deterministic_compute=bool(deterministic_compute),
        allow_backend_fallback=bool(allow_backend_fallback),
    )
    all_variant_records: list[dict[str, Any]] = []
    blocked_experiments: list[dict[str, Any]] = []
    phase_skip_rows: list[dict[str, Any]] = []
    phase_artifact_paths: list[str] = []
    experiment_roots: dict[str, str] = {}
    experiment_records: dict[str, list[dict[str, Any]]] = {}
    experiment_warnings: dict[str, list[str]] = {}

    for experiment in selected_experiments:
        exp_id = str(experiment["experiment_id"])
        experiment_root = output_root / exp_id / campaign_id
        experiment_root.mkdir(parents=True, exist_ok=False)
        experiment_roots[exp_id] = str(experiment_root.resolve())
        experiment_records[exp_id] = []
        experiment_warnings[exp_id] = []

    phase_batches = _build_phase_batches(
        selected_experiments=selected_experiments,
        phase_plan=phase_plan_value,
    )

    global_order_index = 0
    selected_by_id = {str(experiment["experiment_id"]) for experiment in selected_experiments}
    for phase in phase_batches:
        phase_name = str(phase["phase_name"])
        phase_records: list[dict[str, Any]] = []
        phase_experiment_ids: list[str] = []

        expected_ids = [str(value) for value in phase.get("expected_experiment_ids", [])]
        missing_expected = [value for value in expected_ids if value not in selected_by_id]
        for missing_id in missing_expected:
            if missing_id in {"E09", "E10", "E11"}:
                phase_skip_rows.append(
                    {
                        "phase_name": phase_name,
                        "experiment_id": missing_id,
                        "reason": "experiment not present in executable registry selection",
                    }
                )

        for group in phase.get("groups", []):
            group_ids = _coerce_experiment_ids(group)
            phase_experiment_ids.extend(group_ids)
            group_cells: list[tuple[dict[str, Any], dict[str, Any]]] = []

            for experiment in group:
                exp_id = str(experiment["experiment_id"])
                if exp_id == "E22" and len(list(dataset_scope.get("modalities", []))) <= 1:
                    phase_skip_rows.append(
                        {
                            "phase_name": phase_name,
                            "experiment_id": exp_id,
                            "reason": "not applicable under single-modality dataset scope",
                        }
                    )
                    continue

                variants, warnings = _expand_experiment_variants(
                    experiment=experiment,
                    dataset_scope=dataset_scope,
                    search_space_map=search_space_map,
                    search_seed=seed,
                    optuna_enabled=optuna_enabled,
                    optuna_trials=optuna_trials,
                    max_runs_per_experiment=max_runs_per_experiment,
                )
                cells, materialization_warnings = _materialize_experiment_cells(
                    experiment=experiment,
                    variants=variants,
                    dataset_scope=dataset_scope,
                    n_permutations=n_permutations,
                )
                combined_warnings = list(warnings) + list(materialization_warnings)
                if combined_warnings:
                    experiment_warnings[exp_id].extend(combined_warnings)
                if not cells:
                    reason = (
                        "; ".join(combined_warnings)
                        if combined_warnings
                        else "no runnable cells were materialized"
                    )
                    phase_skip_rows.append(
                        {
                            "phase_name": phase_name,
                            "experiment_id": exp_id,
                            "reason": reason,
                        }
                    )
                    continue
                for cell in cells:
                    group_cells.append((experiment, cell))

            if not group_cells:
                continue

            runnable_cells = [
                (experiment, cell)
                for experiment, cell in group_cells
                if bool(cell.get("supported", False))
            ]

            assignments_by_run_id: dict[str, dict[str, Any]] = {}
            job_results_by_run_id: dict[str, dict[str, Any]] = {}
            job_builder_blocked: dict[str, str] = {}
            sequential_only_group = _is_sequential_only_group(
                group_experiment_ids=group_ids,
                cells=[cell for _, cell in group_cells],
            )

            if not dry_run and runnable_cells:
                run_requests: list[ComputeRunRequest] = []
                request_cells: list[tuple[dict[str, Any], dict[str, Any], str]] = []
                for experiment, cell in runnable_cells:
                    exp_id = str(experiment["experiment_id"])
                    run_id = _resolve_variant_run_id(
                        experiment_id=exp_id,
                        variant=cell,
                        campaign_id=campaign_id,
                    )
                    run_requests.append(
                        ComputeRunRequest(
                            order_index=int(global_order_index),
                            run_id=str(run_id),
                            model_name=str(cell.get("params", {}).get("model", "")),
                        )
                    )
                    request_cells.append((experiment, cell, str(run_id)))
                    global_order_index += 1

                schedule = plan_compute_schedule(
                    run_requests=run_requests,
                    base_compute_policy=base_compute_policy,
                    max_parallel_runs=(
                        1 if sequential_only_group else int(max_parallel_runs)
                    ),
                    max_parallel_gpu_runs=(
                        0
                        if sequential_only_group
                        else int(max_parallel_gpu_runs)
                    ),
                )
                assignments_by_run_id = {
                    str(assignment.run_id): assignment.to_payload() for assignment in schedule
                }

                jobs = []
                for order_index, (experiment, cell, run_id) in enumerate(request_cells):
                    assignment_payload = assignments_by_run_id.get(run_id)
                    assigned_order_index = (
                        int(assignment_payload.get("order_index"))
                        if isinstance(assignment_payload, dict)
                        and assignment_payload.get("order_index") is not None
                        else int(order_index)
                    )
                    assignment = (
                        None
                        if assignment_payload is None
                        else ComputeRunAssignment.from_payload(
                            dict(assignment_payload),
                            default_order_index=int(assigned_order_index),
                            default_run_id=str(run_id),
                            default_model_name=str(cell.get("params", {}).get("model", "")),
                        )
                    )
                    job, blocked_reason, _ = _build_variant_official_job(
                        experiment=experiment,
                        variant=cell,
                        campaign_id=campaign_id,
                        experiment_root=output_root / str(experiment["experiment_id"]) / campaign_id,
                        index_csv=index_csv,
                        data_root=data_root,
                        cache_dir=cache_dir,
                        seed=seed,
                        n_permutations=n_permutations,
                        phase_name=phase_name,
                        order_index=int(assigned_order_index),
                        hardware_mode=hardware_mode,
                        gpu_device_id=gpu_device_id,
                        deterministic_compute=bool(deterministic_compute),
                        allow_backend_fallback=bool(allow_backend_fallback),
                        scheduled_compute_assignment=assignment,
                    )
                    if job is None:
                        job_builder_blocked[run_id] = str(blocked_reason or "job_build_failed")
                        continue
                    jobs.append(job)

                effective_parallelism = 1 if sequential_only_group else int(max_parallel_runs)

                job_payloads = _execute_official_jobs(
                    jobs=jobs,
                    max_parallel_runs=effective_parallelism,
                    run_experiment_fn=run_experiment_fn,
                )
                job_results_by_run_id = {
                    str(payload["run_id"]): payload for payload in job_payloads if "run_id" in payload
                }

            for experiment, cell in group_cells:
                exp_id = str(experiment["experiment_id"])
                run_id = _resolve_variant_run_id(
                    experiment_id=exp_id,
                    variant=cell,
                    campaign_id=campaign_id,
                )

                cell_for_record = dict(cell)
                if run_id in job_builder_blocked:
                    cell_for_record["supported"] = False
                    cell_for_record["blocked_reason"] = str(job_builder_blocked[run_id])
                job_execution_override = job_results_by_run_id.get(run_id)
                if (
                    not dry_run
                    and bool(cell_for_record.get("supported", False))
                    and run_id not in job_builder_blocked
                    and job_execution_override is None
                ):
                    job_execution_override = {
                        "watchdog_result": None,
                        "execution_error": {
                            "error": "official_job_result_missing_for_scheduled_run"
                        },
                    }

                record = _execute_variant(
                    experiment=experiment,
                    variant=cell_for_record,
                    campaign_id=campaign_id,
                    experiment_root=output_root / exp_id / campaign_id,
                    index_csv=index_csv,
                    data_root=data_root,
                    cache_dir=cache_dir,
                    seed=seed,
                    n_permutations=n_permutations,
                    dry_run=dry_run,
                    run_experiment_fn=run_experiment_fn,
                    hardware_mode=hardware_mode,
                    gpu_device_id=gpu_device_id,
                    deterministic_compute=bool(deterministic_compute),
                    allow_backend_fallback=bool(allow_backend_fallback),
                    max_parallel_runs=int(max_parallel_runs),
                    max_parallel_gpu_runs=int(max_parallel_gpu_runs),
                    scheduled_compute_assignment=assignments_by_run_id.get(run_id),
                    job_execution_result=job_execution_override,
                    artifact_registry_path=artifact_registry_path,
                    code_ref=commit,
                )
                experiment_records[exp_id].append(record)
                phase_records.append(record)
                all_variant_records.append(record)

        phase_payload = {
            "campaign_id": campaign_id,
            "generated_at": _utc_timestamp(),
            "phase_name": phase_name,
            "experiment_ids": sorted(set(phase_experiment_ids)),
            "status": _phase_status_from_records(phase_records),
            "selected_or_completed_cells": [
                {
                    "experiment_id": str(record.get("experiment_id")),
                    "variant_id": str(record.get("variant_id")),
                    "status": str(record.get("status")),
                }
                for record in phase_records
            ],
            "skipped_experiments": [
                row
                for row in phase_skip_rows
                if str(row.get("phase_name")) == phase_name
            ],
            "decision_note": None,
        }
        artifact_filename = _PHASE_ARTIFACTS.get(phase_name)
        if artifact_filename:
            path = _write_phase_artifact(
                campaign_root=campaign_root,
                filename=artifact_filename,
                payload=phase_payload,
            )
            phase_artifact_paths.append(str(path.resolve()))

    phase_skip_summary_payload = {
        "campaign_id": campaign_id,
        "generated_at": _utc_timestamp(),
        "phase_name": "phase_skip_summary",
        "status": "created",
        "skipped_experiments": phase_skip_rows,
    }
    phase_skip_summary_path = _write_phase_artifact(
        campaign_root=campaign_root,
        filename="phase_skip_summary.json",
        payload=phase_skip_summary_payload,
    )
    phase_artifact_paths.append(str(phase_skip_summary_path.resolve()))

    for experiment in selected_experiments:
        exp_id = str(experiment["experiment_id"])
        variant_records = list(experiment_records.get(exp_id, []))
        warnings = list(experiment_warnings.get(exp_id, []))
        _write_experiment_outputs(
            experiment=experiment,
            experiment_root=output_root / exp_id / campaign_id,
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
                "phase_plan": str(phase_plan_value),
                "max_parallel_runs": int(max_parallel_runs),
                "max_parallel_gpu_runs": int(max_parallel_gpu_runs),
                "hardware_mode": str(hardware_mode),
                "gpu_device_id": int(gpu_device_id) if gpu_device_id is not None else None,
                "deterministic_compute": bool(deterministic_compute),
                "allow_backend_fallback": bool(allow_backend_fallback),
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
        generated_design_rows = _build_generated_design_rows(all_variant_records)
        effect_rows = _build_effect_summary_rows(aggregation)
        summary_rows = list(summary_output_rows)
        run_log_rows = _build_run_log_writeback_rows(
            variant_records=all_variant_records,
            dataset_name=dataset_name,
            seed=seed,
            commit=commit,
        )
        study_review_rows = _build_study_review_rows(
            study_reviews=study_reviews_payload,
            variant_records=all_variant_records,
        )
        workbook_output_path = write_workbook_results(
            source_workbook_path=workbook_source_path,
            version_tag=campaign_id,
            machine_status_rows=machine_rows,
            trial_result_rows=trial_rows,
            summary_output_rows=summary_rows,
            generated_design_rows=generated_design_rows,
            effect_summary_rows=effect_rows,
            study_review_rows=study_review_rows,
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
        "phase_plan": str(phase_plan_value),
        "max_parallel_runs": int(max_parallel_runs),
        "max_parallel_gpu_runs": int(max_parallel_gpu_runs),
        "hardware_mode": str(hardware_mode),
        "gpu_device_id": int(gpu_device_id) if gpu_device_id is not None else None,
        "deterministic_compute": bool(deterministic_compute),
        "allow_backend_fallback": bool(allow_backend_fallback),
        "search_mode": search_mode_value,
        "optuna_trials": int(optuna_trials) if optuna_trials is not None else None,
        "search_space_ids": sorted(search_space_map.keys()),
        "status_counts": _status_snapshot(all_variant_records),
        "experiment_roots": experiment_roots,
        "exports": {
            "run_log_export": str(run_log_path.resolve()),
            "decision_support_summary": str(decision_summary_path.resolve()),
            "decision_recommendations": str(decision_report_path.resolve()),
            "study_review_summary": str(study_review_summary_path.resolve()),
            "result_aggregation": str(aggregation_path.resolve()),
            "summary_outputs_export": str(summary_output_path.resolve()),
            "stage_summaries": [str(path.resolve()) for path in stage_summary_paths],
            "stage_decision_notes": [str(path.resolve()) for path in stage_decision_paths],
            "phase_artifacts": list(phase_artifact_paths),
            "phase_skip_summary": str(phase_skip_summary_path.resolve()),
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
        "study_review_summary_path": str(study_review_summary_path.resolve()),
        "result_aggregation_path": str(aggregation_path.resolve()),
        "summary_outputs_export_path": str(summary_output_path.resolve()),
        "phase_skip_summary_path": str(phase_skip_summary_path.resolve()),
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
    max_parallel_runs: int = 1,
    max_parallel_gpu_runs: int = 1,
    hardware_mode: str = "cpu_only",
    gpu_device_id: int | None = None,
    deterministic_compute: bool = False,
    allow_backend_fallback: bool = False,
    phase_plan: str = "auto",
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
        max_parallel_runs=max_parallel_runs,
        max_parallel_gpu_runs=max_parallel_gpu_runs,
        hardware_mode=hardware_mode,
        gpu_device_id=gpu_device_id,
        deterministic_compute=deterministic_compute,
        allow_backend_fallback=allow_backend_fallback,
        phase_plan=phase_plan,
        run_experiment_fn=run_experiment_fn,
    )


__all__ = [
    "run_decision_support_campaign",
    "run_workbook_decision_support_campaign",
]
