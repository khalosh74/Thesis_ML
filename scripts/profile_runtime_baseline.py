from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

from Thesis_ML.experiments.run_experiment import run_experiment
from Thesis_ML.orchestration.campaign_runner import run_decision_support_campaign
from Thesis_ML.verification.campaign_runtime_profile import verify_campaign_runtime_profile
from Thesis_ML.verification.performance_baseline import (
    BenchmarkCaseResult,
    BenchmarkCaseSpec,
    BenchmarkFingerprint,
    build_benchmark_fingerprint,
    build_case_result_from_run_payload,
    run_baseline_suite,
    write_baseline_bundle,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE_ROOT = REPO_ROOT / "outputs" / "performance" / "baselines"
DEFAULT_DEMO_ROOT = REPO_ROOT / "demo_data" / "synthetic_v1"
DEFAULT_REGISTRY = REPO_ROOT / "configs" / "decision_support_registry_revised_execution.json"
DEFAULT_PROTOCOL = REPO_ROOT / "configs" / "protocols" / "thesis_canonical_nested_v2.json"
DEFAULT_COMPARISONS = [
    REPO_ROOT / "configs" / "comparisons" / "model_family_comparison_v1.json",
    REPO_ROOT / "configs" / "comparisons" / "model_family_grouped_nested_comparison_v2.json",
]


def _timestamp_tag() -> str:
    return str(int(perf_counter() * 1_000_000))


def _resolve_compare_path(value: str | None) -> Path | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    candidate = Path(text).resolve()
    if candidate.is_dir():
        candidate = candidate / "baseline_bundle.json"
    return candidate


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    return payload


def _case_performance_smoke(case_spec: BenchmarkCaseSpec, case_dir: Path) -> BenchmarkCaseResult:
    summary_path = case_dir / "performance_smoke_summary.json"
    reports_root = case_dir / "reports"
    command = [
        sys.executable,
        "scripts/performance_smoke.py",
        "--output",
        str(summary_path),
        "--reports-root",
        str(reports_root),
    ]
    started = perf_counter()
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    elapsed_seconds = float(perf_counter() - started)
    stdout_path = case_dir / "stdout.log"
    stderr_path = case_dir / "stderr.log"
    stdout_path.write_text(str(completed.stdout), encoding="utf-8")
    stderr_path.write_text(str(completed.stderr), encoding="utf-8")
    summary_payload = _load_json(summary_path) if summary_path.exists() else {}
    status = "passed" if completed.returncode == 0 and summary_path.exists() else "failed"
    observability = {
        "expected_artifacts": [str(summary_path)],
        "artifact_presence": {
            str(summary_path): bool(summary_path.exists()),
        },
        "schema_checks": {
            "summary_is_json_object": bool(isinstance(summary_payload, dict)),
            "comparison_dry_run_recorded": bool(
                isinstance(summary_payload.get("comparison_dry_run"), dict)
            ),
            "protocol_dry_run_recorded": bool(
                isinstance(summary_payload.get("protocol_dry_run"), dict)
            ),
        },
    }
    fingerprints = BenchmarkFingerprint(
        metrics_fingerprint=build_benchmark_fingerprint(metrics_payload=summary_payload).metrics_fingerprint
    )
    return BenchmarkCaseResult(
        case_id=case_spec.case_id,
        status=status,
        elapsed_seconds=elapsed_seconds,
        artifact_refs={
            "summary_path": str(summary_path.resolve()),
            "stdout_path": str(stdout_path.resolve()),
            "stderr_path": str(stderr_path.resolve()),
        },
        fingerprints=fingerprints,
        scientific_metrics={},
        observability=observability,
        summary={
            "returncode": int(completed.returncode),
        },
    )


def _case_single_run_ridge_direct(
    case_spec: BenchmarkCaseSpec,
    case_dir: Path,
    *,
    index_csv: Path,
    data_root: Path,
    cache_dir: Path,
) -> BenchmarkCaseResult:
    started = perf_counter()
    run_payload = run_experiment(
        index_csv=index_csv,
        data_root=data_root,
        cache_dir=cache_dir,
        target="coarse_affect",
        model="ridge",
        cv="within_subject_loso_session",
        subject="sub-001",
        seed=42,
        n_permutations=0,
        run_id="single_run_ridge_direct",
        reports_root=case_dir / "reports",
        process_profile_enabled=True,
        process_sample_interval_seconds=0.2,
        process_include_io_counters=True,
    )
    elapsed_seconds = float(perf_counter() - started)
    result = build_case_result_from_run_payload(
        case_id=case_spec.case_id,
        elapsed_seconds=elapsed_seconds,
        run_payload=run_payload,
    )
    report_dir = Path(str(run_payload["report_dir"]))
    progress_events_path = report_dir / "progress_events.jsonl"
    config_path = Path(str(run_payload["config_path"]))
    metrics_path = Path(str(run_payload["metrics_path"]))
    run_status_payload = _load_json(Path(str(run_payload["run_status_path"])))
    config_payload = _load_json(config_path)
    metrics_payload = _load_json(metrics_path)

    expected_artifacts = list(result.observability.get("expected_artifacts", []))
    expected_artifacts.extend([str(progress_events_path.resolve()), str(config_path.resolve()), str(metrics_path.resolve())])
    artifact_presence = dict(result.observability.get("artifact_presence", {}))
    artifact_presence[str(progress_events_path.resolve())] = bool(progress_events_path.exists())
    artifact_presence[str(config_path.resolve())] = bool(config_path.exists())
    artifact_presence[str(metrics_path.resolve())] = bool(metrics_path.exists())
    schema_checks = dict(result.observability.get("schema_checks", {}))
    schema_checks["run_status_completed"] = str(run_status_payload.get("status")) in {
        "completed",
        "success",
    }
    schema_checks["config_has_stage_execution"] = isinstance(
        config_payload.get("stage_execution"), dict
    )
    schema_checks["metrics_has_stage_execution"] = isinstance(
        metrics_payload.get("stage_execution"), dict
    )
    case_status = (
        "passed"
        if all(bool(artifact_presence.get(path)) for path in expected_artifacts)
        and all(bool(value) for value in schema_checks.values())
        else "failed"
    )
    artifact_refs = dict(result.artifact_refs)
    artifact_refs["progress_events_path"] = str(progress_events_path.resolve())
    return BenchmarkCaseResult(
        case_id=result.case_id,
        status=case_status,
        elapsed_seconds=result.elapsed_seconds,
        artifact_refs=artifact_refs,
        fingerprints=result.fingerprints,
        scientific_metrics=dict(result.scientific_metrics),
        observability={
            "expected_artifacts": expected_artifacts,
            "artifact_presence": artifact_presence,
            "schema_checks": schema_checks,
        },
        summary=dict(result.summary),
    )


def _case_runtime_profile_precheck(
    case_spec: BenchmarkCaseSpec,
    case_dir: Path,
    *,
    index_csv: Path,
    data_root: Path,
    cache_dir: Path,
) -> BenchmarkCaseResult:
    started = perf_counter()
    summary = verify_campaign_runtime_profile(
        index_csv=index_csv,
        data_root=data_root,
        cache_dir=cache_dir,
        confirmatory_protocol=DEFAULT_PROTOCOL,
        comparison_specs=DEFAULT_COMPARISONS,
        profile_root=case_dir / "profiles",
        hardware_mode="cpu_only",
        profile_permutations=2,
        profile_inner_folds=1,
        profile_tuning_candidates=2,
    )
    elapsed_seconds = float(perf_counter() - started)
    summary_path = case_dir / "campaign_runtime_profile_summary.json"
    summary_path.write_text(f"{json.dumps(summary, indent=2)}\n", encoding="utf-8")
    profile_artifact_paths = [str(path) for path in list(summary.get("profile_artifact_paths", []))]
    artifact_presence = {str(summary_path.resolve()): bool(summary_path.exists())}
    for artifact_path in profile_artifact_paths:
        artifact_presence[str(Path(artifact_path).resolve())] = bool(Path(artifact_path).exists())
    schema_checks = {
        "summary_has_inputs": isinstance(summary.get("inputs"), dict),
        "summary_has_warnings": isinstance(summary.get("warnings"), list),
        "summary_has_recommendations": isinstance(summary.get("recommendations"), list),
        "summary_has_memoization": isinstance(summary.get("feature_matrix_memoization"), dict),
    }
    status = "passed" if bool(summary.get("passed", False)) else "failed"
    fingerprints = BenchmarkFingerprint(
        metrics_fingerprint=build_benchmark_fingerprint(metrics_payload=summary).metrics_fingerprint
    )
    return BenchmarkCaseResult(
        case_id=case_spec.case_id,
        status=status,
        elapsed_seconds=elapsed_seconds,
        artifact_refs={
            "summary_path": str(summary_path.resolve()),
            "profile_root": str((case_dir / "profiles").resolve()),
        },
        fingerprints=fingerprints,
        scientific_metrics={},
        observability={
            "expected_artifacts": [str(summary_path.resolve())],
            "artifact_presence": artifact_presence,
            "schema_checks": schema_checks,
        },
        summary={
            "profiling_runs_executed": int(summary.get("profiling_runs_executed", 0)),
            "fallback_estimates_used": int(summary.get("fallback_estimates_used", 0)),
            "n_cohorts": int(summary.get("n_cohorts", 0)),
        },
    )


def _case_decision_support_dry_run(
    case_spec: BenchmarkCaseSpec,
    case_dir: Path,
    *,
    index_csv: Path,
    data_root: Path,
    cache_dir: Path,
) -> BenchmarkCaseResult:
    started = perf_counter()
    dry_run_result = run_decision_support_campaign(
        registry_path=DEFAULT_REGISTRY,
        index_csv=index_csv,
        data_root=data_root,
        cache_dir=cache_dir,
        output_root=case_dir / "decision_support",
        experiment_id="E01",
        stage=None,
        run_all=False,
        seed=42,
        n_permutations=0,
        dry_run=True,
        subjects_filter=None,
        tasks_filter=None,
        modalities_filter=None,
        max_runs_per_experiment=1,
        dataset_name="baseline_phase0",
        max_parallel_runs=1,
        max_parallel_gpu_runs=0,
        hardware_mode="cpu_only",
        gpu_device_id=None,
        deterministic_compute=False,
        allow_backend_fallback=False,
        phase_plan="flat",
        quiet_progress=True,
        progress_interval_seconds=60.0,
    )
    elapsed_seconds = float(perf_counter() - started)
    decision_summary_path = Path(str(dry_run_result["decision_support_summary_path"]))
    campaign_manifest_path = Path(str(dry_run_result["campaign_manifest_path"]))
    campaign_manifest_payload = (
        _load_json(campaign_manifest_path) if campaign_manifest_path.exists() else {}
    )
    status_counts = dict(dry_run_result.get("status_counts", {}))
    observability = {
        "expected_artifacts": [
            str(decision_summary_path.resolve()),
            str(campaign_manifest_path.resolve()),
        ],
        "artifact_presence": {
            str(decision_summary_path.resolve()): bool(decision_summary_path.exists()),
            str(campaign_manifest_path.resolve()): bool(campaign_manifest_path.exists()),
        },
        "schema_checks": {
            "manifest_has_campaign_id": bool(campaign_manifest_payload.get("campaign_id")),
            "manifest_has_status_counts": isinstance(
                campaign_manifest_payload.get("status_counts"), dict
            ),
        },
    }
    status = (
        "passed"
        if decision_summary_path.exists()
        and campaign_manifest_path.exists()
        and int(status_counts.get("completed", 0)) >= 0
        and all(bool(value) for value in observability["schema_checks"].values())
        else "failed"
    )
    fingerprints = BenchmarkFingerprint(
        metrics_fingerprint=build_benchmark_fingerprint(
            metrics_payload=campaign_manifest_payload
        ).metrics_fingerprint
    )
    return BenchmarkCaseResult(
        case_id=case_spec.case_id,
        status=status,
        elapsed_seconds=elapsed_seconds,
        artifact_refs={
            "decision_support_summary_path": str(decision_summary_path.resolve()),
            "campaign_manifest_path": str(campaign_manifest_path.resolve()),
        },
        fingerprints=fingerprints,
        scientific_metrics={},
        observability=observability,
        summary={
            "campaign_id": str(dry_run_result.get("campaign_id")),
            "status_counts": status_counts,
        },
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build canonical Phase 0 runtime baseline bundles and optionally compare against a prior baseline."
        )
    )
    parser.add_argument(
        "--mode",
        choices=["ci_synthetic", "operator_dataset"],
        default="ci_synthetic",
        help="ci_synthetic uses shipped demo assets; operator_dataset uses explicit dataset paths.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_BASELINE_ROOT,
        help="Directory where timestamped baseline bundle folders are written.",
    )
    parser.add_argument(
        "--bundle-tag",
        default=None,
        help="Optional stable tag for the bundle folder name. Defaults to a generated timestamp tag.",
    )
    parser.add_argument(
        "--compare-against",
        default=None,
        help="Optional previous baseline bundle JSON path (or folder containing baseline_bundle.json).",
    )
    parser.add_argument("--index-csv", type=Path, default=None, help="Dataset index CSV (operator mode).")
    parser.add_argument("--data-root", type=Path, default=None, help="Dataset root path (operator mode).")
    parser.add_argument("--cache-dir", type=Path, default=None, help="Feature cache directory (operator mode).")
    parser.add_argument(
        "--metrics-tolerance",
        type=float,
        default=1e-8,
        help="Absolute tolerance used when comparing scientific metric values.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    compare_path = _resolve_compare_path(args.compare_against)

    if args.mode == "ci_synthetic":
        index_csv = (DEFAULT_DEMO_ROOT / "dataset_index.csv").resolve()
        data_root = (DEFAULT_DEMO_ROOT / "data_root").resolve()
        cache_dir_base = (DEFAULT_DEMO_ROOT / "cache").resolve()
    else:
        if args.index_csv is None or args.data_root is None or args.cache_dir is None:
            raise ValueError(
                "operator_dataset mode requires --index-csv, --data-root, and --cache-dir."
            )
        index_csv = Path(args.index_csv).resolve()
        data_root = Path(args.data_root).resolve()
        cache_dir_base = Path(args.cache_dir).resolve()

    bundle_tag = str(args.bundle_tag).strip() if args.bundle_tag else _timestamp_tag()
    bundle_root = Path(args.output_root).resolve() / bundle_tag
    bundle_root.mkdir(parents=True, exist_ok=True)

    case_specs = [
        BenchmarkCaseSpec(
            case_id="performance_smoke_existing",
            mode=str(args.mode),
            description="Shipped performance_smoke.py benchmark on workbook + dry-run paths.",
            inputs={"script": "scripts/performance_smoke.py"},
        ),
        BenchmarkCaseSpec(
            case_id="single_run_ridge_direct",
            mode=str(args.mode),
            description="One deterministic direct run_experiment benchmark with direct process profiling.",
            inputs={
                "target": "coarse_affect",
                "model": "ridge",
                "cv": "within_subject_loso_session",
                "subject": "sub-001",
                "seed": 42,
                "n_permutations": 0,
                "process_profile_enabled": True,
            },
        ),
        BenchmarkCaseSpec(
            case_id="runtime_profile_precheck",
            mode=str(args.mode),
            description="Campaign runtime profile precheck over shipped protocol/comparison configs.",
            inputs={
                "confirmatory_protocol": str(DEFAULT_PROTOCOL),
                "comparison_specs": [str(path) for path in DEFAULT_COMPARISONS],
                "profile_permutations": 2,
                "profile_inner_folds": 1,
                "profile_tuning_candidates": 2,
            },
        ),
        BenchmarkCaseSpec(
            case_id="decision_support_dry_run",
            mode=str(args.mode),
            description="Decision-support orchestration dry-run startup path.",
            inputs={
                "registry": str(DEFAULT_REGISTRY),
                "experiment_id": "E01",
                "dry_run": True,
                "max_runs_per_experiment": 1,
            },
        ),
    ]

    def _run_case(case_spec: BenchmarkCaseSpec, case_dir: Path) -> BenchmarkCaseResult:
        if case_spec.case_id == "performance_smoke_existing":
            return _case_performance_smoke(case_spec, case_dir)
        if case_spec.case_id == "single_run_ridge_direct":
            return _case_single_run_ridge_direct(
                case_spec,
                case_dir,
                index_csv=index_csv,
                data_root=data_root,
                cache_dir=(case_dir / "cache" if args.mode == "ci_synthetic" else cache_dir_base),
            )
        if case_spec.case_id == "runtime_profile_precheck":
            return _case_runtime_profile_precheck(
                case_spec,
                case_dir,
                index_csv=index_csv,
                data_root=data_root,
                cache_dir=(case_dir / "cache" if args.mode == "ci_synthetic" else cache_dir_base),
            )
        if case_spec.case_id == "decision_support_dry_run":
            return _case_decision_support_dry_run(
                case_spec,
                case_dir,
                index_csv=index_csv,
                data_root=data_root,
                cache_dir=(case_dir / "cache" if args.mode == "ci_synthetic" else cache_dir_base),
            )
        raise ValueError(f"Unsupported baseline case id: {case_spec.case_id}")

    bundle = run_baseline_suite(
        suite_id="phase0_runtime_baseline",
        baseline_mode=str(args.mode),
        output_root=bundle_root,
        case_specs=case_specs,
        case_runner=_run_case,
        compare_against=compare_path,
        metrics_tolerance=float(args.metrics_tolerance),
    )
    bundle_path = write_baseline_bundle(bundle, bundle_root / "baseline_bundle.json")
    if isinstance(bundle.comparison, dict):
        (bundle_root / "baseline_comparison.json").write_text(
            f"{json.dumps(bundle.comparison, indent=2)}\n",
            encoding="utf-8",
        )

    print(
        json.dumps(
            {
                "bundle_path": str(bundle_path.resolve()),
                "mode": str(args.mode),
                "n_cases": int(bundle.aggregate_summary.get("n_cases", 0)),
                "n_failed": int(bundle.aggregate_summary.get("n_failed", 0)),
                "comparison": bundle.comparison,
            },
            indent=2,
        )
    )

    if int(bundle.aggregate_summary.get("n_failed", 0)) > 0:
        return 1
    if isinstance(bundle.comparison, dict):
        if not bool(bundle.comparison.get("scientific_parity", False)):
            return 1
        if not bool(bundle.comparison.get("observability_parity", False)):
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
