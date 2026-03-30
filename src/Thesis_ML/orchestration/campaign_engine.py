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
from Thesis_ML.config.methodology import MethodologyPolicyName
from Thesis_ML.experiments.compute_policy import resolve_compute_policy
from Thesis_ML.experiments.compute_scheduler import (
    ComputeRunAssignment,
    ComputeRunRequest,
    plan_compute_schedule,
)
from Thesis_ML.experiments.model_catalog import (
    get_model_cost_entry,
)
from Thesis_ML.experiments.model_catalog import (
    projected_runtime_seconds as resolve_projected_runtime_seconds,
)
from Thesis_ML.observability import (
    AnomalyEngine,
    ConsoleReporter,
    EtaEstimator,
    ExecutionEventBus,
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
    resolve_variant_id as _resolve_variant_id,
)
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
from Thesis_ML.orchestration.reporting import (
    write_campaign_execution_report as _write_campaign_execution_report,
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


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(str(value))
    except Exception:
        return None


def _safe_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except Exception:
        return None


def _resolve_tuning_enabled(params: dict[str, Any]) -> bool:
    methodology = str(params.get("methodology_policy_name") or "").strip().lower()
    if methodology == MethodologyPolicyName.GROUPED_NESTED_TUNING.value:
        return True
    explicit = params.get("tuning_enabled")
    if isinstance(explicit, bool):
        return bool(explicit)
    if explicit is None:
        return False
    return str(explicit).strip().lower() in {"1", "true", "yes", "on"}


def _resolve_planned_projected_runtime_seconds(
    *,
    params: dict[str, Any],
    n_permutations: int,
) -> float | None:
    explicit_projected = _safe_float(params.get("projected_runtime_seconds"))
    if explicit_projected is not None and explicit_projected > 0.0:
        return float(explicit_projected)
    model_name = str(params.get("model") or "").strip().lower()
    if not model_name:
        return None
    try:
        projected = resolve_projected_runtime_seconds(
            model_name=model_name,
            framework_mode=FrameworkMode.EXPLORATORY,
            methodology_policy=str(params.get("methodology_policy_name") or "").strip() or None,
            tuning_enabled=_resolve_tuning_enabled(params),
        )
    except Exception:
        return None
    if int(n_permutations) > 0:
        projected = int(projected) + int(max(0, int(n_permutations)))
    return float(max(1, int(projected)))


def _build_eta_planning_metadata(
    *,
    campaign_id: str,
    phase_name: str,
    experiment_id: str,
    run_id: str,
    params: dict[str, Any],
    effective_n_permutations: int,
) -> dict[str, Any]:
    model_name = str(params.get("model") or "").strip().lower()
    model_cost_tier: str | None = None
    if model_name:
        try:
            model_cost_tier = str(get_model_cost_entry(model_name).cost_tier.value)
        except Exception:
            model_cost_tier = None
    return {
        "campaign_id": str(campaign_id),
        "phase_name": str(phase_name),
        "experiment_id": str(experiment_id),
        "run_id": str(run_id),
        "framework_mode": FrameworkMode.EXPLORATORY.value,
        "model": model_name if model_name else None,
        "model_cost_tier": model_cost_tier,
        "feature_space": (
            str(params.get("feature_space"))
            if params.get("feature_space") not in (None, "")
            else "whole_brain_masked"
        ),
        "preprocessing_strategy": (
            str(params.get("preprocessing_strategy"))
            if params.get("preprocessing_strategy") not in (None, "")
            else "none"
        ),
        "dimensionality_strategy": (
            str(params.get("dimensionality_strategy"))
            if params.get("dimensionality_strategy") not in (None, "")
            else "none"
        ),
        "tuning_enabled": bool(_resolve_tuning_enabled(params)),
        "cv_mode": str(params.get("cv")) if params.get("cv") not in (None, "") else None,
        "n_permutations": int(max(0, int(effective_n_permutations))),
        "subject": str(params.get("subject")) if params.get("subject") else None,
        "train_subject": str(params.get("train_subject")) if params.get("train_subject") else None,
        "test_subject": str(params.get("test_subject")) if params.get("test_subject") else None,
        "task": str(params.get("filter_task")) if params.get("filter_task") else None,
        "modality": str(params.get("filter_modality")) if params.get("filter_modality") else None,
        "projected_runtime_seconds": _resolve_planned_projected_runtime_seconds(
            params=params,
            n_permutations=int(max(0, int(effective_n_permutations))),
        ),
    }


def _extract_actual_runtime_seconds(record: dict[str, Any]) -> float | None:
    stage_timings = record.get("stage_timings_seconds")
    if isinstance(stage_timings, dict):
        total = _safe_float(stage_timings.get("total"))
        if total is not None and total > 0.0:
            return float(total)
    return None


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
    {
        "phase_name": "Stage 4 representation/preprocessing lock",
        "groups": [["E09"], ["E10"], ["E11"]],
    },
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
        flat_groups = [[selected_by_id[key]] for key in selected_by_id]
        return [
            {
                "phase_name": "Flat selected sequence",
                "phase_order_index": 0,
                "groups": flat_groups,
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
    runtime_profile_summary: Path | None = None,
    quiet_progress: bool = False,
    progress_interval_seconds: float = 15.0,
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
    history_path = campaign_root.parent / "runtime_history.jsonl"
    anomaly_engine: AnomalyEngine | None = None
    try:
        anomaly_engine = AnomalyEngine(
            campaign_root=campaign_root,
            campaign_id=campaign_id,
        )
    except Exception:
        anomaly_engine = None
    eta_estimator: EtaEstimator | None = None
    try:
        eta_estimator = EtaEstimator(
            campaign_root=campaign_root,
            campaign_id=campaign_id,
            history_path=history_path,
            runtime_profile_summary_path=(
                Path(runtime_profile_summary) if runtime_profile_summary is not None else None
            ),
        )
    except Exception:
        eta_estimator = None
    console_reporter: ConsoleReporter | None = None
    try:
        console_reporter = ConsoleReporter(
            interval_seconds=float(progress_interval_seconds),
            quiet=bool(quiet_progress),
        )
    except Exception:
        console_reporter = None
    event_bus: ExecutionEventBus | None = None
    try:
        event_bus = ExecutionEventBus(
            campaign_root=campaign_root,
            campaign_id=campaign_id,
            eta_estimator=eta_estimator,
            anomaly_engine=anomaly_engine,
            console_reporter=console_reporter,
        )
    except Exception:
        event_bus = None

    def _emit_campaign_event(**kwargs: Any) -> None:
        if event_bus is None:
            return
        try:
            event_bus.emit_event(**kwargs)
        except Exception:
            return

    _emit_campaign_event(
        event_name="campaign_started",
        scope="campaign",
        status="running",
        stage="campaign",
        message="campaign started",
        metadata={
            "selected_experiments": [str(exp["experiment_id"]) for exp in selected_experiments],
            "experiments_total": int(len(selected_experiments)),
            "dry_run": bool(dry_run),
        },
    )

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
    if float(progress_interval_seconds) <= 0.0:
        raise ValueError("progress_interval_seconds must be > 0.")
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
    experiment_phase_by_id: dict[str, str] = {}
    experiment_started_ids: set[str] = set()
    for phase in phase_batches:
        phase_name = str(phase["phase_name"])
        phase_records: list[dict[str, Any]] = []
        phase_experiment_ids: list[str] = []
        _emit_campaign_event(
            event_name="phase_started",
            scope="phase",
            status="running",
            stage="campaign",
            phase_name=phase_name,
            message="phase started",
            metadata={
                "dry_run": bool(dry_run),
                "expected_experiment_ids": [
                    str(value) for value in phase.get("expected_experiment_ids", [])
                ],
            },
        )

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
                experiment_phase_by_id.setdefault(exp_id, phase_name)
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
                    _emit_campaign_event(
                        event_name="experiment_skipped",
                        scope="experiment",
                        status="skipped",
                        stage="campaign",
                        phase_name=phase_name,
                        experiment_id=exp_id,
                        message="experiment selected but produced no materialized cells",
                        metadata={
                            "reason": reason,
                            "dry_run": bool(dry_run),
                        },
                    )
                    continue
                if exp_id not in experiment_started_ids:
                    _emit_campaign_event(
                        event_name="experiment_started",
                        scope="experiment",
                        status="running",
                        stage="campaign",
                        phase_name=phase_name,
                        experiment_id=exp_id,
                        message="experiment started",
                    )
                    experiment_started_ids.add(exp_id)
                for cell in cells:
                    group_cells.append((experiment, cell))

            if not group_cells:
                continue

            for experiment, cell in group_cells:
                exp_id = str(experiment["experiment_id"])
                variant_id = _resolve_variant_id(cell)
                run_id = _resolve_variant_run_id(
                    experiment_id=exp_id,
                    variant=cell,
                    campaign_id=campaign_id,
                )
                params_for_eta = (
                    dict(cell.get("params", {})) if isinstance(cell.get("params"), dict) else {}
                )
                resolved_permutation_override = _optional_int(cell.get("n_permutations_override"))
                effective_n_permutations = (
                    int(resolved_permutation_override)
                    if resolved_permutation_override is not None
                    else int(n_permutations)
                )
                eta_planning_metadata = _build_eta_planning_metadata(
                    campaign_id=campaign_id,
                    phase_name=phase_name,
                    experiment_id=exp_id,
                    run_id=run_id,
                    params=params_for_eta,
                    effective_n_permutations=int(effective_n_permutations),
                )
                _emit_campaign_event(
                    event_name="run_planned",
                    scope="run",
                    status="planned",
                    stage="campaign",
                    phase_name=phase_name,
                    experiment_id=exp_id,
                    variant_id=variant_id,
                    run_id=run_id,
                    message="run planned",
                    metadata={
                        "dry_run": bool(dry_run),
                        "supported": bool(cell.get("supported", False)),
                        "blocked_reason": cell.get("blocked_reason"),
                        **eta_planning_metadata,
                    },
                )

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
                    max_parallel_runs=(1 if sequential_only_group else int(max_parallel_runs)),
                    max_parallel_gpu_runs=(
                        0 if sequential_only_group else int(max_parallel_gpu_runs)
                    ),
                )
                assignments_by_run_id = {
                    str(assignment.run_id): assignment.to_payload() for assignment in schedule
                }

                jobs = []
                for order_index, (experiment, cell, run_id) in enumerate(request_cells):
                    assignment_payload = assignments_by_run_id.get(run_id)
                    assigned_order_index_override = (
                        _optional_int(assignment_payload.get("order_index"))
                        if isinstance(assignment_payload, dict)
                        else None
                    )
                    assigned_order_index = (
                        int(assigned_order_index_override)
                        if assigned_order_index_override is not None
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
                        experiment_root=output_root
                        / str(experiment["experiment_id"])
                        / campaign_id,
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
                    _emit_campaign_event(
                        event_name="run_dispatched",
                        scope="run",
                        status="dispatched",
                        stage="campaign",
                        phase_name=phase_name,
                        experiment_id=str(experiment["experiment_id"]),
                        variant_id=_resolve_variant_id(cell),
                        run_id=str(run_id),
                        message="run dispatched",
                    )
                    _emit_campaign_event(
                        event_name="run_started",
                        scope="run",
                        status="running",
                        stage="campaign",
                        phase_name=phase_name,
                        experiment_id=str(experiment["experiment_id"]),
                        variant_id=_resolve_variant_id(cell),
                        run_id=str(run_id),
                        message="run started",
                    )

                effective_parallelism = 1 if sequential_only_group else int(max_parallel_runs)

                job_payloads = _execute_official_jobs(
                    jobs=jobs,
                    max_parallel_runs=effective_parallelism,
                    run_experiment_fn=run_experiment_fn,
                )
                job_results_by_run_id = {
                    str(payload["run_id"]): payload
                    for payload in job_payloads
                    if "run_id" in payload
                }

            for experiment, cell in group_cells:
                exp_id = str(experiment["experiment_id"])
                variant_id = _resolve_variant_id(cell)
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
                    progress_callback=(
                        event_bus.build_progress_callback(
                            phase_name=phase_name,
                            experiment_id=exp_id,
                            variant_id=variant_id,
                            run_id=run_id,
                        )
                        if event_bus is not None
                        else None
                    ),
                    artifact_registry_path=artifact_registry_path,
                    code_ref=commit,
                )
                experiment_records[exp_id].append(record)
                phase_records.append(record)
                all_variant_records.append(record)
                record_status = str(record.get("status"))
                terminal_params = (
                    dict(cell_for_record.get("params", {}))
                    if isinstance(cell_for_record.get("params"), dict)
                    else {}
                )
                terminal_override = _optional_int(cell_for_record.get("n_permutations_override"))
                terminal_n_permutations = (
                    int(terminal_override) if terminal_override is not None else int(n_permutations)
                )
                eta_terminal_metadata = _build_eta_planning_metadata(
                    campaign_id=campaign_id,
                    phase_name=phase_name,
                    experiment_id=exp_id,
                    run_id=run_id,
                    params=terminal_params,
                    effective_n_permutations=int(terminal_n_permutations),
                )
                eta_terminal_metadata["dry_run"] = bool(dry_run)
                eta_terminal_metadata["actual_runtime_seconds"] = _extract_actual_runtime_seconds(
                    record
                )
                if record.get("framework_mode") not in (None, ""):
                    eta_terminal_metadata["framework_mode"] = str(record.get("framework_mode"))
                if record.get("model_cost_tier") not in (None, ""):
                    eta_terminal_metadata["model_cost_tier"] = str(record.get("model_cost_tier"))
                projected_runtime = _safe_float(record.get("projected_runtime_seconds"))
                if projected_runtime is not None and projected_runtime > 0.0:
                    eta_terminal_metadata["projected_runtime_seconds"] = float(projected_runtime)
                if record.get("cv") not in (None, ""):
                    eta_terminal_metadata["cv_mode"] = str(record.get("cv"))
                if record.get("feature_space") not in (None, ""):
                    eta_terminal_metadata["feature_space"] = str(record.get("feature_space"))
                if record.get("preprocessing_strategy") not in (None, ""):
                    eta_terminal_metadata["preprocessing_strategy"] = str(
                        record.get("preprocessing_strategy")
                    )
                if record.get("dimensionality_strategy") not in (None, ""):
                    eta_terminal_metadata["dimensionality_strategy"] = str(
                        record.get("dimensionality_strategy")
                    )
                if record.get("tuning_enabled") is not None:
                    eta_terminal_metadata["tuning_enabled"] = bool(record.get("tuning_enabled"))
                anomaly_terminal_metadata = dict(eta_terminal_metadata)
                anomaly_terminal_metadata.update(
                    {
                        "status": str(record_status),
                        "roi_spec_path": record.get("roi_spec_path"),
                        "stage_timings_seconds": record.get("stage_timings_seconds"),
                        "process_profile_summary": record.get("process_profile_summary"),
                    }
                )
                if anomaly_engine is not None:
                    try:
                        anomaly_engine.inspect_terminal_run(anomaly_terminal_metadata)
                    except Exception:
                        pass
                if record_status == "completed":
                    _emit_campaign_event(
                        event_name="run_finished",
                        scope="run",
                        status="completed",
                        stage="campaign",
                        phase_name=phase_name,
                        experiment_id=exp_id,
                        variant_id=variant_id,
                        run_id=run_id,
                        message="run finished",
                        metadata=eta_terminal_metadata,
                    )
                elif record_status == "failed":
                    _emit_campaign_event(
                        event_name="run_failed",
                        scope="run",
                        status="failed",
                        stage="campaign",
                        phase_name=phase_name,
                        experiment_id=exp_id,
                        variant_id=variant_id,
                        run_id=run_id,
                        message="run failed",
                        metadata={
                            "error": record.get("error"),
                            **eta_terminal_metadata,
                        },
                    )
                elif record_status == "blocked":
                    _emit_campaign_event(
                        event_name="run_blocked",
                        scope="run",
                        status="blocked",
                        stage="campaign",
                        phase_name=phase_name,
                        experiment_id=exp_id,
                        variant_id=variant_id,
                        run_id=run_id,
                        message="run blocked",
                        metadata={
                            "dry_run": bool(dry_run),
                            "blocked_reason": record.get("blocked_reason"),
                            **eta_terminal_metadata,
                        },
                    )
                elif record_status == "dry_run":
                    _emit_campaign_event(
                        event_name="run_dry_run",
                        scope="run",
                        status="dry_run",
                        stage="campaign",
                        phase_name=phase_name,
                        experiment_id=exp_id,
                        variant_id=variant_id,
                        run_id=run_id,
                        message="run dry-run",
                        metadata=eta_terminal_metadata,
                    )

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
                row for row in phase_skip_rows if str(row.get("phase_name")) == phase_name
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
        _emit_campaign_event(
            event_name="phase_finished",
            scope="phase",
            status=str(phase_payload["status"]),
            stage="campaign",
            phase_name=phase_name,
            message="phase finished",
            metadata={
                "dry_run": bool(dry_run),
                "experiment_ids": list(sorted(set(phase_experiment_ids))),
                "selected_or_completed_cells": list(phase_payload["selected_or_completed_cells"]),
            },
        )

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
        experiment_status = _phase_status_from_records(variant_records)
        if not variant_records and warnings:
            experiment_status = "skipped"
        _write_experiment_outputs(
            experiment=experiment,
            experiment_root=output_root / exp_id / campaign_id,
            variant_records=variant_records,
            warnings=warnings,
        )
        _emit_campaign_event(
            event_name="experiment_finished",
            scope="experiment",
            status=experiment_status,
            stage="campaign",
            phase_name=experiment_phase_by_id.get(exp_id),
            experiment_id=exp_id,
            message="experiment finished",
            metadata={"warnings": list(warnings)},
        )
        if not variant_records and warnings:
            blocked_experiments.append(
                {
                    "experiment_id": exp_id,
                    "reasons": sorted({str(item) for item in warnings if str(item)}),
                }
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
        warnings_by_experiment=experiment_warnings,
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
                "quiet_progress": bool(quiet_progress),
                "progress_interval_seconds": float(progress_interval_seconds),
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

    eta_calibration_path: str | None = None
    if eta_estimator is not None:
        try:
            eta_estimator.finalize()
            eta_calibration_path = str((campaign_root / "campaign_eta_calibration.json").resolve())
        except Exception:
            eta_calibration_path = None

    anomaly_report_path: str | None = None
    if anomaly_engine is not None:
        try:
            anomaly_engine.finalize()
            anomaly_report_path = str((campaign_root / "campaign_anomaly_report.json").resolve())
        except Exception:
            anomaly_report_path = None

    execution_report_md_path: str | None = None
    execution_report_json_path: str | None = None
    try:
        execution_report_md, execution_report_json = _write_campaign_execution_report(
            campaign_root=campaign_root,
            campaign_id=campaign_id,
        )
        execution_report_md_path = str(execution_report_md.resolve())
        execution_report_json_path = str(execution_report_json.resolve())
    except Exception:
        execution_report_md_path = None
        execution_report_json_path = None

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
        "quiet_progress": bool(quiet_progress),
        "progress_interval_seconds": float(progress_interval_seconds),
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
            "eta_state": str((campaign_root / "eta_state.json").resolve()),
            "eta_calibration": eta_calibration_path,
            "runtime_history": str(history_path.resolve()),
            "anomalies": str((campaign_root / "anomalies.jsonl").resolve()),
            "anomaly_report": anomaly_report_path,
            "campaign_execution_report_md": execution_report_md_path,
            "campaign_execution_report_json": execution_report_json_path,
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
    _emit_campaign_event(
        event_name="campaign_finished",
        scope="campaign",
        status="finished",
        stage="campaign",
        message="campaign finished",
        metadata={"status_counts": dict(campaign_manifest["status_counts"])},
    )

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
        "eta_state_path": str((campaign_root / "eta_state.json").resolve()),
        "eta_calibration_path": eta_calibration_path,
        "runtime_history_path": str(history_path.resolve()),
        "anomalies_path": str((campaign_root / "anomalies.jsonl").resolve()),
        "anomaly_report_path": anomaly_report_path,
        "campaign_execution_report_md_path": execution_report_md_path,
        "campaign_execution_report_json_path": execution_report_json_path,
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
    runtime_profile_summary: Path | None = None,
    quiet_progress: bool = False,
    progress_interval_seconds: float = 15.0,
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
        runtime_profile_summary=runtime_profile_summary,
        quiet_progress=bool(quiet_progress),
        progress_interval_seconds=float(progress_interval_seconds),
        run_experiment_fn=run_experiment_fn,
    )


__all__ = [
    "run_decision_support_campaign",
    "run_workbook_decision_support_campaign",
]
