from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from Thesis_ML.config.paths import (
    DEFAULT_DECISION_SUPPORT_OUTPUT_ROOT,
    DEFAULT_DECISION_SUPPORT_REGISTRY,
    DEFAULT_WORKBOOK_OUTPUT_DIR,
)
from Thesis_ML.config.runtime_selection import resolve_runtime_config_path
from Thesis_ML.experiments.compute_policy import HARDWARE_MODE_CHOICES
from Thesis_ML.orchestration.contracts import CompiledStudyManifest
from Thesis_ML.orchestration.execution_bridge import command_to_text as _command_to_text
from Thesis_ML.orchestration.experiment_selection import (
    experiment_sort_key as _experiment_sort_key,
)
from Thesis_ML.orchestration.study_loading import (
    read_registry_manifest as _read_registry_manifest,
)
from Thesis_ML.orchestration.study_loading import (
    read_workbook_manifest as _read_workbook_manifest,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Automate thesis decision-support experiments (E01-E11) using the existing "
            "thesisml-run-experiment execution path."
        )
    )
    parser.add_argument(
        "--registry",
        default=None,
        help="Path to decision-support experiment registry JSON.",
    )
    parser.add_argument(
        "--registry-alias",
        default=None,
        help="Registry alias for decision-support registry selection when --registry is not provided.",
    )
    parser.add_argument(
        "--workbook",
        default=None,
        help="Optional workbook path. If provided, compile Experiment_Definitions and run from workbook.",
    )
    parser.add_argument(
        "--index-csv",
        default=str(Path("Data") / "processed" / "dataset_index.csv"),
        help="Dataset index CSV used by the runner.",
    )
    parser.add_argument(
        "--data-root",
        default="Data",
        help="Root path for beta/mask paths in index CSV.",
    )
    parser.add_argument(
        "--cache-dir",
        default=str(Path("Data") / "processed" / "feature_cache"),
        help="Feature cache directory for runner.",
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_DECISION_SUPPORT_OUTPUT_ROOT),
        help="Root folder for decision-support manifests and summaries.",
    )
    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument("--experiment-id", help="Run one experiment ID (e.g., E01).")
    target_group.add_argument("--stage", help="Run one full stage (exact stage name).")
    target_group.add_argument("--all", action="store_true", help="Run all experiments in registry.")
    parser.add_argument("--seed", type=int, default=42, help="Seed passed to runner.")
    parser.add_argument(
        "--n-permutations",
        type=int,
        default=0,
        help="Optional permutation rounds for each run.",
    )
    parser.add_argument(
        "--subjects",
        nargs="*",
        default=None,
        help="Optional subject filter for variant expansion.",
    )
    parser.add_argument(
        "--tasks",
        nargs="*",
        default=None,
        help="Optional task filter for variant expansion.",
    )
    parser.add_argument(
        "--modalities",
        nargs="*",
        default=None,
        help="Optional modality filter for variant expansion.",
    )
    parser.add_argument(
        "--max-runs-per-experiment",
        type=int,
        default=None,
        help="Optional cap on executable variants per experiment.",
    )
    parser.add_argument(
        "--dataset-name",
        default="Internal BAS2",
        help="Dataset label stored in workbook-export logs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve variants and write manifests without invoking model runs.",
    )
    parser.add_argument(
        "--search-mode",
        choices=["deterministic", "optuna"],
        default="deterministic",
        help=(
            "Search execution mode for optional Search_Spaces. "
            "'deterministic' expands grid dimensions; 'optuna' enables Optuna-backed sampling."
        ),
    )
    parser.add_argument(
        "--optuna-trials",
        type=int,
        default=None,
        help="Optional number of Optuna trials per search space when --search-mode optuna is used.",
    )
    parser.add_argument(
        "--max-parallel-runs",
        type=int,
        default=1,
        help=(
            "Operational scheduling control: maximum process-level runs dispatched in parallel "
            "within a phase group."
        ),
    )
    parser.add_argument(
        "--max-parallel-gpu-runs",
        type=int,
        default=1,
        help=(
            "Operational scheduling control: GPU-lane cap per dispatch window when hardware mode "
            "supports GPU usage."
        ),
    )
    parser.add_argument(
        "--hardware-mode",
        default="cpu_only",
        choices=list(HARDWARE_MODE_CHOICES),
        help="Operational compute policy for dispatched decision-support runs.",
    )
    parser.add_argument(
        "--gpu-device-id",
        type=int,
        default=None,
        help="Optional GPU device ID for gpu_only/max_both operational scheduling.",
    )
    parser.add_argument(
        "--deterministic-compute",
        action="store_true",
        help="Operational flag: request deterministic compute mode for dispatched runs.",
    )
    parser.add_argument(
        "--allow-backend-fallback",
        action="store_true",
        help="Operational flag: allow GPU requests to fall back to CPU reference backend.",
    )
    parser.add_argument(
        "--phase-plan",
        default="auto",
        choices=["auto", "flat"],
        help="Operational scheduler phase plan selection (default: auto).",
    )
    parser.add_argument(
        "--runtime-profile-summary",
        default=None,
        help=(
            "Optional runtime profile summary JSON from verify_campaign_runtime_profile.py "
            "used to seed adaptive ETA estimates."
        ),
    )
    parser.add_argument(
        "--quiet-progress",
        action="store_true",
        help="Suppress periodic heartbeat lines while keeping final completion/error output.",
    )
    parser.add_argument(
        "--progress-interval-seconds",
        type=float,
        default=15.0,
        help="Heartbeat summary interval in seconds for live console progress output.",
    )
    parser.add_argument(
        "--write-back-workbook",
        action="store_true",
        help="When --workbook is used, write machine/trial outputs back to a versioned workbook copy.",
    )
    parser.add_argument(
        "--workbook-output-dir",
        default=str(DEFAULT_WORKBOOK_OUTPUT_DIR),
        help="Optional directory for versioned workbook write-back output.",
    )
    parser.add_argument(
        "--no-write-back-run-log",
        action="store_true",
        help="When writing back workbook results, skip appending summary rows to Run_Log.",
    )
    return parser


def print_registry_status(registry: CompiledStudyManifest) -> None:
    experiments = sorted(
        [experiment.model_dump(mode="python") for experiment in registry.experiments],
        key=_experiment_sort_key,
    )
    executable = [exp for exp in experiments if bool(exp.get("executable_now"))]
    blocked = [exp for exp in experiments if not bool(exp.get("executable_now"))]

    print("Executable now:")
    for exp in executable:
        exp_id = str(exp.get("experiment_id"))
        status = str(exp.get("execution_status", "unknown"))
        print(f"  - {exp_id}: {exp.get('title')} [{status}]")

    print("Blocked now:")
    for exp in blocked:
        exp_id = str(exp.get("experiment_id"))
        reasons = exp.get("blocked_reasons", [])
        reason_text = "; ".join(str(reason) for reason in reasons) if reasons else "unspecified"
        print(f"  - {exp_id}: {exp.get('title')} -> {reason_text}")


def print_stage1_commands(args: argparse.Namespace) -> None:
    base = [
        sys.executable,
        "run_decision_support_experiments.py",
        "--index-csv",
        str(args.index_csv),
        "--data-root",
        str(args.data_root),
        "--cache-dir",
        str(args.cache_dir),
        "--output-root",
        str(args.output_root),
        "--stage",
        "Stage 1 - Target lock",
        "--seed",
        str(args.seed),
    ]
    if args.workbook:
        base.extend(["--workbook", str(args.workbook)])
    else:
        if args.registry:
            base.extend(["--registry", str(args.registry)])
        elif args.registry_alias:
            base.extend(["--registry-alias", str(args.registry_alias)])
    if args.n_permutations > 0:
        base.extend(["--n-permutations", str(args.n_permutations)])
    if args.max_runs_per_experiment:
        base.extend(["--max-runs-per-experiment", str(args.max_runs_per_experiment)])
    if args.subjects:
        base.extend(["--subjects", *args.subjects])
    if args.tasks:
        base.extend(["--tasks", *args.tasks])
    if args.modalities:
        base.extend(["--modalities", *args.modalities])
    if str(args.search_mode) != "deterministic":
        base.extend(["--search-mode", str(args.search_mode)])
    if args.optuna_trials:
        base.extend(["--optuna-trials", str(args.optuna_trials)])
    if int(args.max_parallel_runs) != 1:
        base.extend(["--max-parallel-runs", str(args.max_parallel_runs)])
    if int(args.max_parallel_gpu_runs) != 1:
        base.extend(["--max-parallel-gpu-runs", str(args.max_parallel_gpu_runs)])
    if str(args.hardware_mode) != "cpu_only":
        base.extend(["--hardware-mode", str(args.hardware_mode)])
    if args.gpu_device_id is not None:
        base.extend(["--gpu-device-id", str(args.gpu_device_id)])
    if bool(args.deterministic_compute):
        base.append("--deterministic-compute")
    if bool(args.allow_backend_fallback):
        base.append("--allow-backend-fallback")
    if str(args.phase_plan).strip().lower() != "auto":
        base.extend(["--phase-plan", str(args.phase_plan)])
    if args.runtime_profile_summary:
        base.extend(["--runtime-profile-summary", str(args.runtime_profile_summary)])
    if bool(args.quiet_progress):
        base.append("--quiet-progress")
    if float(args.progress_interval_seconds) != 15.0:
        base.extend(["--progress-interval-seconds", str(args.progress_interval_seconds)])

    command = _command_to_text(base)
    print("Stage 1 command:")
    print(f"  {command}")


def main(
    argv: list[str] | None = None,
    *,
    run_decision_support_campaign_fn: Callable[..., dict[str, Any]],
    run_workbook_decision_support_campaign_fn: Callable[..., dict[str, Any]],
    read_registry_manifest_fn: Callable[[Path], CompiledStudyManifest] = _read_registry_manifest,
    read_workbook_manifest_fn: Callable[[Path], CompiledStudyManifest] = _read_workbook_manifest,
) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    workbook_path = Path(args.workbook) if args.workbook else None
    registry_path = (
        resolve_runtime_config_path(
            args.registry,
            args.registry_alias,
            default_alias="registry.decision_support_default",
            fallback_path=DEFAULT_DECISION_SUPPORT_REGISTRY,
        )
        if workbook_path is None
        else None
    )
    if workbook_path is None and registry_path is None:
        raise ValueError("registry_path resolution failed while workbook mode is not selected.")
    resolved_registry_path = Path(registry_path) if registry_path is not None else None
    index_csv = Path(args.index_csv)
    data_root = Path(args.data_root)
    cache_dir = Path(args.cache_dir)
    output_root = Path(args.output_root)
    runtime_profile_summary_path = (
        Path(args.runtime_profile_summary) if args.runtime_profile_summary else None
    )
    workbook_output_dir = Path(args.workbook_output_dir) if args.workbook_output_dir else None

    if workbook_path is None:
        if resolved_registry_path is None:
            raise ValueError("registry_path resolution failed while workbook mode is not selected.")
        registry = read_registry_manifest_fn(resolved_registry_path)
    else:
        registry = read_workbook_manifest_fn(workbook_path)
    try:
        if workbook_path is not None:
            result = run_workbook_decision_support_campaign_fn(
                workbook_path=workbook_path,
                index_csv=index_csv,
                data_root=data_root,
                cache_dir=cache_dir,
                output_root=output_root,
                experiment_id=args.experiment_id,
                stage=args.stage,
                run_all=bool(args.all),
                seed=args.seed,
                n_permutations=args.n_permutations,
                dry_run=bool(args.dry_run),
                subjects_filter=args.subjects,
                tasks_filter=args.tasks,
                modalities_filter=args.modalities,
                max_runs_per_experiment=args.max_runs_per_experiment,
                dataset_name=args.dataset_name,
                write_back_to_workbook=bool(args.write_back_workbook),
                workbook_output_dir=workbook_output_dir,
                append_workbook_run_log=not bool(args.no_write_back_run_log),
                search_mode=str(args.search_mode),
                optuna_trials=args.optuna_trials,
                max_parallel_runs=int(args.max_parallel_runs),
                max_parallel_gpu_runs=int(args.max_parallel_gpu_runs),
                hardware_mode=str(args.hardware_mode),
                gpu_device_id=args.gpu_device_id,
                deterministic_compute=bool(args.deterministic_compute),
                allow_backend_fallback=bool(args.allow_backend_fallback),
                phase_plan=str(args.phase_plan),
                runtime_profile_summary=runtime_profile_summary_path,
                quiet_progress=bool(args.quiet_progress),
                progress_interval_seconds=float(args.progress_interval_seconds),
            )
        else:
            if resolved_registry_path is None:
                raise ValueError(
                    "registry_path resolution failed while workbook mode is not selected."
                )
            result = run_decision_support_campaign_fn(
                registry_path=resolved_registry_path,
                index_csv=index_csv,
                data_root=data_root,
                cache_dir=cache_dir,
                output_root=output_root,
                experiment_id=args.experiment_id,
                stage=args.stage,
                run_all=bool(args.all),
                seed=args.seed,
                n_permutations=args.n_permutations,
                dry_run=bool(args.dry_run),
                subjects_filter=args.subjects,
                tasks_filter=args.tasks,
                modalities_filter=args.modalities,
                max_runs_per_experiment=args.max_runs_per_experiment,
                dataset_name=args.dataset_name,
                search_mode=str(args.search_mode),
                optuna_trials=args.optuna_trials,
                max_parallel_runs=int(args.max_parallel_runs),
                max_parallel_gpu_runs=int(args.max_parallel_gpu_runs),
                hardware_mode=str(args.hardware_mode),
                gpu_device_id=args.gpu_device_id,
                deterministic_compute=bool(args.deterministic_compute),
                allow_backend_fallback=bool(args.allow_backend_fallback),
                phase_plan=str(args.phase_plan),
                runtime_profile_summary=runtime_profile_summary_path,
                quiet_progress=bool(args.quiet_progress),
                progress_interval_seconds=float(args.progress_interval_seconds),
            )
    except RuntimeError as exc:
        print_registry_status(registry)
        print(f"\nError: {exc}")
        if args.experiment_id:
            print_stage1_commands(args)
        return 2

    print("Campaign complete")
    print(f"- campaign_id: {result['campaign_id']}")
    print(f"- campaign_root: {result['campaign_root']}")
    print(f"- selected_experiments: {', '.join(result['selected_experiments'])}")
    print(f"- status_counts: {result['status_counts']}")
    print(f"- run_log_export: {result['run_log_export_path']}")
    print(f"- decision_summary: {result['decision_support_summary_path']}")
    print(f"- decision_report: {result['decision_recommendations_path']}")
    print(f"- aggregation: {result['result_aggregation_path']}")
    print(f"- summary_outputs_export: {result['summary_outputs_export_path']}")
    print(f"- manifest: {result['campaign_manifest_path']}")
    if result.get("workbook_output_path"):
        print(f"- workbook_output_path: {result['workbook_output_path']}")
    return 0


__all__ = ["build_parser", "main", "print_registry_status", "print_stage1_commands"]
