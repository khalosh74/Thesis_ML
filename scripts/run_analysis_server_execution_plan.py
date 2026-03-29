from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from Thesis_ML.orchestration.campaign_runner import run_decision_support_campaign
from Thesis_ML.orchestration.study_loading import read_registry_manifest

_THREAD_CAP_ENV: dict[str, str] = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}

GPU_ELIGIBLE_EXPERIMENTS: set[str] = {
    "E01",
    "E02",
    "E03",
    "E04",
    "E05",
    "E09",
    "E10",
    "E11",
    "E12",
    "E13",
    "E15",
    "E16",
    "E17",
    "E20",
    "E21",
    "E22",
    "E23",
    "E24",
}
CPU_ONLY_EXPERIMENTS: set[str] = {"E06", "E07", "E08"}


@dataclass(frozen=True)
class BatchSpec:
    experiments: tuple[str, ...]


@dataclass(frozen=True)
class PhaseSpec:
    phase_id: str
    description: str
    batches: tuple[BatchSpec, ...]
    stop_on_failure: bool = True
    gpu_mode_for_ridge: str = "max_both"
    manual_lock_name: str | None = None


PHASE_SPECS: tuple[PhaseSpec, ...] = (
    PhaseSpec(
        phase_id="phase0_preflight",
        description="Preflight validation and environment capture.",
        batches=(),
        stop_on_failure=True,
        gpu_mode_for_ridge="cpu_only",
        manual_lock_name=None,
    ),
    PhaseSpec(
        phase_id="phase1_target_scope_lock",
        description="Run E01, then E02 and E03 in parallel.",
        batches=(BatchSpec(("E01",)), BatchSpec(("E02", "E03"))),
        stop_on_failure=True,
        gpu_mode_for_ridge="max_both",
        manual_lock_name="stage1_lock.json",
    ),
    PhaseSpec(
        phase_id="phase2_split_transfer_boundary",
        description="Run E04, then E05.",
        batches=(BatchSpec(("E04",)), BatchSpec(("E05",))),
        stop_on_failure=True,
        gpu_mode_for_ridge="max_both",
        manual_lock_name="stage2_lock.json",
    ),
    PhaseSpec(
        phase_id="phase3_model_lock",
        description="Run E06, then E07, then E08 sequentially.",
        batches=(BatchSpec(("E06",)), BatchSpec(("E07",)), BatchSpec(("E08",))),
        stop_on_failure=True,
        gpu_mode_for_ridge="cpu_only",
        manual_lock_name="stage3_lock.json",
    ),
    PhaseSpec(
        phase_id="phase4_representation_preprocessing_lock",
        description="Run E09, E10, E11 sequentially if present.",
        batches=(BatchSpec(("E09",)), BatchSpec(("E10",)), BatchSpec(("E11",))),
        stop_on_failure=True,
        gpu_mode_for_ridge="max_both",
        manual_lock_name="stage4_lock.json",
    ),
    PhaseSpec(
        phase_id="phase5_confirmatory",
        description="Run E16 and E17 in parallel.",
        batches=(BatchSpec(("E16", "E17")),),
        stop_on_failure=True,
        gpu_mode_for_ridge="gpu_only",
        manual_lock_name="final_confirmatory_pipeline.json",
    ),
    PhaseSpec(
        phase_id="phase6_blocking_robustness",
        description="Run E12, E13, E20 in parallel.",
        batches=(BatchSpec(("E12", "E13", "E20")),),
        stop_on_failure=False,
        gpu_mode_for_ridge="max_both",
        manual_lock_name=None,
    ),
    PhaseSpec(
        phase_id="phase7_context_robustness",
        description="Run E21, E22, E23, E15 in parallel.",
        batches=(BatchSpec(("E21", "E22", "E23", "E15")),),
        stop_on_failure=False,
        gpu_mode_for_ridge="max_both",
        manual_lock_name=None,
    ),
    PhaseSpec(
        phase_id="phase8_reproducibility_audit",
        description="Run E24 last.",
        batches=(BatchSpec(("E24",)),),
        stop_on_failure=False,
        gpu_mode_for_ridge="max_both",
        manual_lock_name=None,
    ),
)


@dataclass(frozen=True)
class WorkerPayload:
    experiment_id: str
    registry_path: str
    index_csv: str
    data_root: str
    cache_dir: str
    output_root: str
    seed: int
    n_permutations: int
    dry_run: bool
    subjects_filter: list[str] | None
    tasks_filter: list[str] | None
    modalities_filter: list[str] | None
    max_runs_per_experiment: int | None
    dataset_name: str
    execution_mode: str
    gpu_device_id: int | None
    deterministic_compute: bool
    allow_backend_fallback: bool
    per_run_max_parallel_runs: int
    per_run_max_parallel_gpu_runs: int


@dataclass
class ExperimentResult:
    experiment_id: str
    status: str
    campaign_id: str | None = None
    campaign_root: str | None = None
    campaign_manifest_path: str | None = None
    status_counts: dict[str, int] = field(default_factory=dict)
    error: str | None = None
    execution_mode: str | None = None
    per_run_max_parallel_runs: int | None = None
    per_run_max_parallel_gpu_runs: int | None = None



def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()



def _timestamp_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")



def _apply_thread_caps() -> None:
    for key, value in _THREAD_CAP_ENV.items():
        os.environ.setdefault(key, value)



def _experiment_uses_gpu_lane(experiment_id: str) -> bool:
    return experiment_id in GPU_ELIGIBLE_EXPERIMENTS and experiment_id not in CPU_ONLY_EXPERIMENTS



def _run_single_experiment_worker(payload: WorkerPayload) -> dict[str, Any]:
    _apply_thread_caps()

    def _wrapped_run_experiment(**kwargs: Any) -> dict[str, Any]:
        from Thesis_ML.experiments.run_experiment import run_experiment

        model_name = str(kwargs.get("model", "")).strip().lower()
        requested_mode = str(payload.execution_mode)
        effective_mode = requested_mode
        gpu_device_id = payload.gpu_device_id
        max_parallel_gpu_runs = int(payload.per_run_max_parallel_gpu_runs)
        if requested_mode != "cpu_only" and model_name != "ridge":
            effective_mode = "cpu_only"
            gpu_device_id = None
            max_parallel_gpu_runs = 0
        if effective_mode == "cpu_only":
            gpu_device_id = None
            max_parallel_gpu_runs = 0
        return run_experiment(
            **kwargs,
            hardware_mode=effective_mode,
            gpu_device_id=gpu_device_id,
            deterministic_compute=bool(payload.deterministic_compute),
            allow_backend_fallback=bool(payload.allow_backend_fallback),
            max_parallel_runs=int(payload.per_run_max_parallel_runs),
            max_parallel_gpu_runs=int(max_parallel_gpu_runs),
        )

    result = run_decision_support_campaign(
        registry_path=Path(payload.registry_path),
        index_csv=Path(payload.index_csv),
        data_root=Path(payload.data_root),
        cache_dir=Path(payload.cache_dir),
        output_root=Path(payload.output_root),
        experiment_id=str(payload.experiment_id),
        stage=None,
        run_all=False,
        seed=int(payload.seed),
        n_permutations=int(payload.n_permutations),
        dry_run=bool(payload.dry_run),
        subjects_filter=list(payload.subjects_filter) if payload.subjects_filter else None,
        tasks_filter=list(payload.tasks_filter) if payload.tasks_filter else None,
        modalities_filter=list(payload.modalities_filter) if payload.modalities_filter else None,
        max_runs_per_experiment=payload.max_runs_per_experiment,
        dataset_name=str(payload.dataset_name),
        run_experiment_fn=_wrapped_run_experiment,
    )
    return {
        "experiment_id": payload.experiment_id,
        "status": "completed",
        "campaign_id": result.get("campaign_id"),
        "campaign_root": result.get("campaign_root"),
        "campaign_manifest_path": result.get("campaign_manifest_path"),
        "status_counts": dict(result.get("status_counts", {})),
        "execution_mode": payload.execution_mode,
        "per_run_max_parallel_runs": int(payload.per_run_max_parallel_runs),
        "per_run_max_parallel_gpu_runs": int(payload.per_run_max_parallel_gpu_runs),
    }



def _safe_worker(payload: WorkerPayload) -> dict[str, Any]:
    try:
        return _run_single_experiment_worker(payload)
    except Exception as exc:  # pragma: no cover - worker failure path
        return {
            "experiment_id": payload.experiment_id,
            "status": "failed",
            "campaign_id": None,
            "campaign_root": None,
            "campaign_manifest_path": None,
            "status_counts": {},
            "error": str(exc),
            "execution_mode": payload.execution_mode,
            "per_run_max_parallel_runs": int(payload.per_run_max_parallel_runs),
            "per_run_max_parallel_gpu_runs": int(payload.per_run_max_parallel_gpu_runs),
        }



def _coerce_positive_int(value: int, *, field_name: str) -> int:
    if int(value) <= 0:
        raise ValueError(f"{field_name} must be >= 1")
    return int(value)



def _preflight(
    *,
    registry_path: Path,
    index_csv: Path,
    data_root: Path,
    cache_dir: Path,
    output_root: Path,
    gpu_device_id: int | None,
    dry_run: bool,
) -> dict[str, Any]:
    checks: dict[str, Any] = {
        "started_at_utc": _utc_now(),
        "registry_path": str(registry_path.resolve()),
        "index_csv": str(index_csv.resolve()),
        "data_root": str(data_root.resolve()),
        "cache_dir": str(cache_dir.resolve()),
        "output_root": str(output_root.resolve()),
        "dry_run": bool(dry_run),
        "thread_caps": dict(_THREAD_CAP_ENV),
    }
    checks["registry_exists"] = registry_path.exists()
    checks["index_exists"] = index_csv.exists()
    checks["data_root_exists"] = data_root.exists()
    checks["cache_dir_exists"] = cache_dir.exists()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = read_registry_manifest(registry_path)
    checks["registry_schema_version"] = str(manifest.schema_version)
    checks["available_experiment_ids"] = [str(exp.experiment_id) for exp in manifest.experiments]

    gpu_snapshot: dict[str, Any] | None = None
    if gpu_device_id is not None:
        try:
            from Thesis_ML.experiments.compute_capabilities import detect_compute_capabilities

            snapshot = detect_compute_capabilities(requested_device_id=int(gpu_device_id))
            gpu_snapshot = snapshot.to_dict() if hasattr(snapshot, "to_dict") else dict(snapshot)
        except Exception as exc:  # pragma: no cover - hardware environment dependent
            gpu_snapshot = {"error": str(exc)}
    checks["gpu_snapshot"] = gpu_snapshot
    checks["ended_at_utc"] = _utc_now()
    checks["ok"] = bool(
        checks["registry_exists"]
        and checks["index_exists"]
        and checks["data_root_exists"]
        and checks["cache_dir_exists"]
    )
    return checks



def _phase_experiments_available(phase: PhaseSpec, available: set[str]) -> tuple[list[str], list[str]]:
    requested = [exp_id for batch in phase.batches for exp_id in batch.experiments]
    present = [exp_id for exp_id in requested if exp_id in available]
    missing = [exp_id for exp_id in requested if exp_id not in available]
    return present, missing



def _split_cpu_budget(total_budget: int, n_jobs: int) -> int:
    return max(1, total_budget // max(1, n_jobs))



def _build_batch_payloads(
    *,
    batch: BatchSpec,
    available_experiments: set[str],
    registry_path: Path,
    index_csv: Path,
    data_root: Path,
    cache_dir: Path,
    phase_root: Path,
    seed: int,
    n_permutations: int,
    dry_run: bool,
    subjects_filter: list[str] | None,
    tasks_filter: list[str] | None,
    modalities_filter: list[str] | None,
    max_runs_per_experiment: int | None,
    dataset_name: str,
    cpu_budget: int,
    gpu_device_id: int | None,
    deterministic_compute: bool,
    allow_backend_fallback: bool,
    gpu_mode_for_ridge: str,
) -> tuple[list[WorkerPayload], list[str]]:
    present = [exp_id for exp_id in batch.experiments if exp_id in available_experiments]
    missing = [exp_id for exp_id in batch.experiments if exp_id not in available_experiments]
    if not present:
        return [], missing
    gpu_assigned = False
    per_run_cpu_parallelism = _split_cpu_budget(cpu_budget, len(present))
    payloads: list[WorkerPayload] = []
    for exp_id in present:
        execution_mode = "cpu_only"
        per_run_gpu_parallel_runs = 0
        if gpu_device_id is not None and not gpu_assigned and _experiment_uses_gpu_lane(exp_id):
            execution_mode = gpu_mode_for_ridge
            per_run_gpu_parallel_runs = 1 if execution_mode != "cpu_only" else 0
            gpu_assigned = execution_mode != "cpu_only"
        payloads.append(
            WorkerPayload(
                experiment_id=exp_id,
                registry_path=str(registry_path),
                index_csv=str(index_csv),
                data_root=str(data_root),
                cache_dir=str(cache_dir),
                output_root=str(phase_root),
                seed=int(seed),
                n_permutations=int(n_permutations),
                dry_run=bool(dry_run),
                subjects_filter=list(subjects_filter) if subjects_filter else None,
                tasks_filter=list(tasks_filter) if tasks_filter else None,
                modalities_filter=list(modalities_filter) if modalities_filter else None,
                max_runs_per_experiment=max_runs_per_experiment,
                dataset_name=dataset_name,
                execution_mode=execution_mode,
                gpu_device_id=gpu_device_id if execution_mode != "cpu_only" else None,
                deterministic_compute=bool(deterministic_compute),
                allow_backend_fallback=bool(allow_backend_fallback),
                per_run_max_parallel_runs=int(per_run_cpu_parallelism),
                per_run_max_parallel_gpu_runs=int(per_run_gpu_parallel_runs),
            )
        )
    return payloads, missing



def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")



def _phase_status_from_results(results: list[ExperimentResult]) -> str:
    statuses = [item.status for item in results]
    if not statuses:
        return "skipped"
    if any(status == "failed" for status in statuses):
        return "failed"
    if all(status == "completed" for status in statuses):
        return "completed"
    return "partial"



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Phased analysis-server execution plan for the thesis experiment program. "
            "Uses the existing decision-support campaign runner and adds phase gating, "
            "controlled parallelism, and explicit CPU/GPU lane assignment."
        )
    )
    parser.add_argument("--registry", required=True)
    parser.add_argument("--index-csv", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--campaign-tag", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-permutations", type=int, default=0)
    parser.add_argument("--dataset-name", default="Internal BAS2")
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument("--modalities", nargs="*", default=None)
    parser.add_argument("--max-runs-per-experiment", type=int, default=None)
    parser.add_argument("--cpu-budget", type=int, default=12)
    parser.add_argument("--max-parallel-experiments", type=int, default=4)
    parser.add_argument("--gpu-device-id", type=int, default=0)
    parser.add_argument("--disable-gpu", action="store_true")
    parser.add_argument("--deterministic-compute", action="store_true")
    parser.add_argument("--allow-backend-fallback", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--execution-mode",
        choices=["fresh", "resume", "force"],
        default="fresh",
        help="fresh: fail if campaign tag exists; resume: reuse existing top-level root; force: overwrite.",
    )
    return parser



def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _apply_thread_caps()

    registry_path = Path(args.registry).resolve()
    index_csv = Path(args.index_csv).resolve()
    data_root = Path(args.data_root).resolve()
    cache_dir = Path(args.cache_dir).resolve()
    output_root = Path(args.output_root).resolve()

    cpu_budget = _coerce_positive_int(args.cpu_budget, field_name="cpu_budget")
    max_parallel_experiments = _coerce_positive_int(
        args.max_parallel_experiments,
        field_name="max_parallel_experiments",
    )
    campaign_tag = args.campaign_tag or _timestamp_tag()
    campaign_root = output_root / "server_execution" / campaign_tag

    if campaign_root.exists() and args.execution_mode == "fresh":
        raise FileExistsError(
            f"Campaign root already exists: {campaign_root}. Use --execution-mode resume or force."
        )
    if campaign_root.exists() and args.execution_mode == "force":
        import shutil

        shutil.rmtree(campaign_root)
    campaign_root.mkdir(parents=True, exist_ok=True)

    overall_manifest = {
        "schema_version": "analysis-server-execution-plan-v1",
        "campaign_tag": campaign_tag,
        "campaign_root": str(campaign_root.resolve()),
        "started_at_utc": _utc_now(),
        "registry": str(registry_path),
        "index_csv": str(index_csv),
        "data_root": str(data_root),
        "cache_dir": str(cache_dir),
        "output_root": str(output_root),
        "cpu_budget": cpu_budget,
        "max_parallel_experiments": int(max_parallel_experiments),
        "gpu_device_id": None if args.disable_gpu else int(args.gpu_device_id),
        "deterministic_compute": bool(args.deterministic_compute),
        "allow_backend_fallback": bool(args.allow_backend_fallback),
        "dry_run": bool(args.dry_run),
        "execution_mode": str(args.execution_mode),
        "thread_caps": dict(_THREAD_CAP_ENV),
        "phases": [],
    }

    preflight_root = campaign_root / PHASE_SPECS[0].phase_id
    preflight_root.mkdir(parents=True, exist_ok=True)
    preflight_summary = _preflight(
        registry_path=registry_path,
        index_csv=index_csv,
        data_root=data_root,
        cache_dir=cache_dir,
        output_root=preflight_root,
        gpu_device_id=None if args.disable_gpu else int(args.gpu_device_id),
        dry_run=bool(args.dry_run),
    )
    _write_json(preflight_root / "preflight_summary.json", preflight_summary)
    overall_manifest["phases"].append(
        {
            "phase_id": PHASE_SPECS[0].phase_id,
            "status": "completed" if preflight_summary.get("ok") else "failed",
            "phase_root": str(preflight_root.resolve()),
            "summary_path": str((preflight_root / "preflight_summary.json").resolve()),
        }
    )
    if not preflight_summary.get("ok"):
        overall_manifest["ended_at_utc"] = _utc_now()
        overall_manifest["status"] = "failed_preflight"
        _write_json(campaign_root / "execution_manifest.json", overall_manifest)
        return 2

    manifest = read_registry_manifest(registry_path)
    available_experiments = {str(exp.experiment_id) for exp in manifest.experiments}

    for phase in PHASE_SPECS[1:]:
        phase_root = campaign_root / phase.phase_id
        phase_root.mkdir(parents=True, exist_ok=True)
        phase_results: list[ExperimentResult] = []
        missing_in_phase: list[str] = []
        for batch_index, batch in enumerate(phase.batches, start=1):
            batch_root = phase_root / f"batch_{batch_index:02d}"
            batch_root.mkdir(parents=True, exist_ok=True)
            payloads, missing = _build_batch_payloads(
                batch=batch,
                available_experiments=available_experiments,
                registry_path=registry_path,
                index_csv=index_csv,
                data_root=data_root,
                cache_dir=cache_dir,
                phase_root=batch_root,
                seed=int(args.seed),
                n_permutations=int(args.n_permutations),
                dry_run=bool(args.dry_run),
                subjects_filter=list(args.subjects) if args.subjects else None,
                tasks_filter=list(args.tasks) if args.tasks else None,
                modalities_filter=list(args.modalities) if args.modalities else None,
                max_runs_per_experiment=args.max_runs_per_experiment,
                dataset_name=str(args.dataset_name),
                cpu_budget=cpu_budget,
                gpu_device_id=None if args.disable_gpu else int(args.gpu_device_id),
                deterministic_compute=bool(args.deterministic_compute),
                allow_backend_fallback=bool(args.allow_backend_fallback),
                gpu_mode_for_ridge=phase.gpu_mode_for_ridge,
            )
            missing_in_phase.extend(missing)
            if not payloads:
                continue
            max_workers = min(len(payloads), max_parallel_experiments)
            with ProcessPoolExecutor(max_workers=max_workers) as pool:
                future_map = {pool.submit(_safe_worker, payload): payload for payload in payloads}
                for future in as_completed(future_map):
                    payload = future_map[future]
                    raw_result = future.result()
                    phase_results.append(
                        ExperimentResult(
                            experiment_id=str(raw_result.get("experiment_id", payload.experiment_id)),
                            status=str(raw_result.get("status", "failed")),
                            campaign_id=raw_result.get("campaign_id"),
                            campaign_root=raw_result.get("campaign_root"),
                            campaign_manifest_path=raw_result.get("campaign_manifest_path"),
                            status_counts=dict(raw_result.get("status_counts", {})),
                            error=raw_result.get("error"),
                            execution_mode=raw_result.get("execution_mode"),
                            per_run_max_parallel_runs=raw_result.get("per_run_max_parallel_runs"),
                            per_run_max_parallel_gpu_runs=raw_result.get(
                                "per_run_max_parallel_gpu_runs"
                            ),
                        )
                    )
            _write_json(
                batch_root / "batch_summary.json",
                {
                    "phase_id": phase.phase_id,
                    "batch_index": batch_index,
                    "requested_experiments": list(batch.experiments),
                    "missing_experiments": missing,
                    "results": [
                        asdict(item)
                        for item in phase_results
                        if item.experiment_id in set(batch.experiments)
                    ],
                },
            )
        phase_status = _phase_status_from_results(phase_results)
        phase_payload = {
            "phase_id": phase.phase_id,
            "description": phase.description,
            "status": phase_status,
            "phase_root": str(phase_root.resolve()),
            "missing_experiments": sorted(set(missing_in_phase)),
            "results": [asdict(item) for item in phase_results],
            "manual_lock_name": phase.manual_lock_name,
            "started_at_utc": None,
            "ended_at_utc": _utc_now(),
        }
        summary_path = phase_root / "phase_summary.json"
        _write_json(summary_path, phase_payload)
        if phase.manual_lock_name:
            _write_json(
                phase_root / phase.manual_lock_name,
                {
                    "phase_id": phase.phase_id,
                    "manual_decision_required": True,
                    "source_experiments": [item.experiment_id for item in phase_results],
                    "source_campaign_roots": [
                        item.campaign_root for item in phase_results if item.campaign_root
                    ],
                    "generated_at_utc": _utc_now(),
                },
            )
        overall_manifest["phases"].append(
            {
                "phase_id": phase.phase_id,
                "status": phase_status,
                "phase_root": str(phase_root.resolve()),
                "summary_path": str(summary_path.resolve()),
            }
        )
        if phase.stop_on_failure and phase_status == "failed":
            overall_manifest["ended_at_utc"] = _utc_now()
            overall_manifest["status"] = f"stopped_after_{phase.phase_id}"
            _write_json(campaign_root / "execution_manifest.json", overall_manifest)
            return 1

    overall_manifest["ended_at_utc"] = _utc_now()
    overall_manifest["status"] = "completed"
    _write_json(campaign_root / "execution_manifest.json", overall_manifest)
    print(json.dumps({
        "campaign_tag": campaign_tag,
        "campaign_root": str(campaign_root.resolve()),
        "status": overall_manifest["status"],
    }, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
