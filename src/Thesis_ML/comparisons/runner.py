from __future__ import annotations

from pathlib import Path
from typing import Any

from Thesis_ML.comparisons.artifacts import write_comparison_artifacts
from Thesis_ML.comparisons.compiler import compile_comparison
from Thesis_ML.comparisons.models import (
    ComparisonRunResult,
    ComparisonSpec,
    ComparisonStatus,
    CompiledComparisonManifest,
    CompiledComparisonRunSpec,
)
from Thesis_ML.config.framework_mode import FrameworkMode
from Thesis_ML.experiments.run_experiment import run_experiment


def _comparison_output_dir(comparison: ComparisonSpec, reports_root: Path | str) -> Path:
    root = Path(reports_root)
    return root / "comparison_runs" / f"{comparison.comparison_id}__{comparison.comparison_version}"


def _comparison_context_payload(spec: CompiledComparisonRunSpec) -> dict[str, Any]:
    return {
        "framework_mode": FrameworkMode.LOCKED_COMPARISON.value,
        "comparison_id": spec.comparison_id,
        "comparison_version": spec.comparison_version,
        "variant_id": spec.variant_id,
        "claim_ids": list(spec.claim_ids),
        "artifact_requirements": list(spec.artifact_requirements),
        "primary_metric": spec.primary_metric,
        "controls": spec.controls.model_dump(mode="json"),
        "interpretability_enabled": bool(spec.interpretability_enabled),
    }


def _to_run_result_success(
    spec: CompiledComparisonRunSpec,
    run_payload: dict[str, Any],
) -> ComparisonRunResult:
    metrics_payload = run_payload.get("metrics", {}) if isinstance(run_payload, dict) else {}
    metrics: dict[str, float | int | str | bool | None] = {}
    if isinstance(metrics_payload, dict):
        for key in ("balanced_accuracy", "macro_f1", "accuracy", "n_folds"):
            value = metrics_payload.get(key)
            if isinstance(value, (float, int, str, bool)) or value is None:
                metrics[key] = value

    return ComparisonRunResult(
        run_id=spec.run_id,
        framework_mode=FrameworkMode.LOCKED_COMPARISON.value,
        comparison_id=spec.comparison_id,
        comparison_version=spec.comparison_version,
        variant_id=spec.variant_id,
        status="completed",
        report_dir=(str(run_payload.get("report_dir")) if run_payload.get("report_dir") else None),
        config_path=(
            str(run_payload.get("config_path")) if run_payload.get("config_path") else None
        ),
        metrics_path=(
            str(run_payload.get("metrics_path")) if run_payload.get("metrics_path") else None
        ),
        metrics=metrics,
    )


def execute_compiled_comparison(
    *,
    comparison: ComparisonSpec,
    compiled_manifest: CompiledComparisonManifest,
    index_csv: Path | str,
    data_root: Path | str,
    cache_dir: Path | str,
    reports_root: Path | str,
    force: bool,
    resume: bool,
    dry_run: bool,
) -> dict[str, Any]:
    run_results: list[ComparisonRunResult] = []
    reports_root_path = Path(reports_root)
    reports_root_path.mkdir(parents=True, exist_ok=True)

    for spec in compiled_manifest.runs:
        if dry_run:
            run_results.append(
                ComparisonRunResult(
                    run_id=spec.run_id,
                    framework_mode=FrameworkMode.LOCKED_COMPARISON.value,
                    comparison_id=spec.comparison_id,
                    comparison_version=spec.comparison_version,
                    variant_id=spec.variant_id,
                    status="planned",
                )
            )
            continue

        try:
            payload = run_experiment(
                index_csv=Path(index_csv),
                data_root=Path(data_root),
                cache_dir=Path(cache_dir),
                target=spec.target,
                model=spec.model,
                cv=spec.cv_mode,
                subject=spec.subject,
                train_subject=spec.train_subject,
                test_subject=spec.test_subject,
                seed=int(spec.seed),
                filter_task=spec.filter_task,
                filter_modality=spec.filter_modality,
                n_permutations=int(spec.controls.n_permutations),
                run_id=spec.run_id,
                reports_root=reports_root_path,
                force=bool(force),
                resume=bool(resume),
                framework_mode=FrameworkMode.LOCKED_COMPARISON,
                primary_metric_name=spec.primary_metric,
                permutation_metric_name=spec.controls.permutation_metric,
                interpretability_enabled_override=bool(spec.interpretability_enabled),
                comparison_context=_comparison_context_payload(spec),
            )
            run_results.append(_to_run_result_success(spec, payload))
        except Exception as exc:
            run_results.append(
                ComparisonRunResult(
                    run_id=spec.run_id,
                    framework_mode=FrameworkMode.LOCKED_COMPARISON.value,
                    comparison_id=spec.comparison_id,
                    comparison_version=spec.comparison_version,
                    variant_id=spec.variant_id,
                    status="failed",
                    error=str(exc),
                )
            )

    comparison_output_dir = _comparison_output_dir(comparison, reports_root=reports_root_path)
    artifact_paths = write_comparison_artifacts(
        comparison=comparison,
        compiled_manifest=compiled_manifest,
        run_results=run_results,
        output_dir=comparison_output_dir,
        dry_run=dry_run,
    )

    n_completed = sum(result.status == "completed" for result in run_results)
    n_failed = sum(result.status == "failed" for result in run_results)
    n_planned = sum(result.status == "planned" for result in run_results)
    return {
        "comparison_id": comparison.comparison_id,
        "comparison_version": comparison.comparison_version,
        "comparison_output_dir": str(comparison_output_dir.resolve()),
        "compiled_manifest": compiled_manifest.model_dump(mode="json"),
        "run_results": [result.model_dump(mode="json") for result in run_results],
        "n_completed": int(n_completed),
        "n_failed": int(n_failed),
        "n_planned": int(n_planned),
        "artifact_paths": artifact_paths,
    }


def compile_and_run_comparison(
    *,
    comparison: ComparisonSpec,
    index_csv: Path | str,
    data_root: Path | str,
    cache_dir: Path | str,
    reports_root: Path | str,
    variant_ids: list[str] | None = None,
    force: bool = False,
    resume: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    if comparison.status == ComparisonStatus.RETIRED:
        raise ValueError("Comparison execution rejected: comparison status is 'retired'.")
    if comparison.status == ComparisonStatus.DRAFT:
        raise ValueError(
            "Comparison execution rejected: comparison status is 'draft'. "
            "Lock comparison spec before execution."
        )
    compiled_manifest = compile_comparison(
        comparison,
        index_csv=index_csv,
        variant_ids=variant_ids,
    )
    return execute_compiled_comparison(
        comparison=comparison,
        compiled_manifest=compiled_manifest,
        index_csv=index_csv,
        data_root=data_root,
        cache_dir=cache_dir,
        reports_root=reports_root,
        force=force,
        resume=resume,
        dry_run=dry_run,
    )
