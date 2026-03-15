from __future__ import annotations

from pathlib import Path
from typing import Any

from Thesis_ML.experiments.run_experiment import run_experiment
from Thesis_ML.protocols.artifacts import write_protocol_artifacts
from Thesis_ML.protocols.compiler import compile_protocol
from Thesis_ML.protocols.models import (
    CompiledProtocolManifest,
    CompiledRunSpec,
    ProtocolRunResult,
    ThesisProtocol,
)


def _protocol_output_dir(protocol: ThesisProtocol, reports_root: Path | str) -> Path:
    root = Path(reports_root)
    return root / "protocol_runs" / f"{protocol.protocol_id}__{protocol.protocol_version}"


def _protocol_context_payload(spec: CompiledRunSpec) -> dict[str, Any]:
    return {
        "canonical_run": bool(spec.canonical_run),
        "protocol_id": spec.protocol_id,
        "protocol_version": spec.protocol_version,
        "protocol_schema_version": spec.protocol_schema_version,
        "suite_id": spec.suite_id,
        "claim_ids": list(spec.claim_ids),
        "artifact_requirements": list(spec.artifact_requirements),
        "primary_metric": spec.primary_metric,
        "controls": spec.controls.model_dump(mode="json"),
        "interpretability_enabled": bool(spec.interpretability_enabled),
    }


def _to_run_result_success(
    spec: CompiledRunSpec,
    run_payload: dict[str, Any],
) -> ProtocolRunResult:
    metrics_payload = run_payload.get("metrics", {}) if isinstance(run_payload, dict) else {}
    metrics: dict[str, float | int | str | bool | None] = {}
    if isinstance(metrics_payload, dict):
        for key in ("balanced_accuracy", "macro_f1", "accuracy", "n_folds"):
            value = metrics_payload.get(key)
            if isinstance(value, (float, int, str, bool)) or value is None:
                metrics[key] = value

    return ProtocolRunResult(
        run_id=spec.run_id,
        suite_id=spec.suite_id,
        status="completed",
        report_dir=(str(run_payload.get("report_dir")) if run_payload.get("report_dir") else None),
        metrics_path=(
            str(run_payload.get("metrics_path")) if run_payload.get("metrics_path") else None
        ),
        config_path=(str(run_payload.get("config_path")) if run_payload.get("config_path") else None),
        metrics=metrics,
    )


def execute_compiled_protocol(
    *,
    protocol: ThesisProtocol,
    compiled_manifest: CompiledProtocolManifest,
    index_csv: Path | str,
    data_root: Path | str,
    cache_dir: Path | str,
    reports_root: Path | str,
    force: bool,
    resume: bool,
    dry_run: bool,
) -> dict[str, Any]:
    run_results: list[ProtocolRunResult] = []
    reports_root_path = Path(reports_root)
    reports_root_path.mkdir(parents=True, exist_ok=True)

    for spec in compiled_manifest.runs:
        if dry_run:
            run_results.append(
                ProtocolRunResult(
                    run_id=spec.run_id,
                    suite_id=spec.suite_id,
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
                primary_metric_name=spec.primary_metric,
                permutation_metric_name=spec.controls.permutation_metric,
                interpretability_enabled_override=bool(spec.interpretability_enabled),
                protocol_context=_protocol_context_payload(spec),
            )
            run_results.append(_to_run_result_success(spec, payload))
        except Exception as exc:
            run_results.append(
                ProtocolRunResult(
                    run_id=spec.run_id,
                    suite_id=spec.suite_id,
                    status="failed",
                    error=str(exc),
                )
            )

    protocol_output_dir = _protocol_output_dir(protocol, reports_root=reports_root_path)
    artifact_paths = write_protocol_artifacts(
        protocol=protocol,
        compiled_manifest=compiled_manifest,
        run_results=run_results,
        output_dir=protocol_output_dir,
        dry_run=dry_run,
    )

    n_completed = sum(result.status == "completed" for result in run_results)
    n_failed = sum(result.status == "failed" for result in run_results)
    n_planned = sum(result.status == "planned" for result in run_results)
    return {
        "protocol_id": protocol.protocol_id,
        "protocol_version": protocol.protocol_version,
        "protocol_output_dir": str(protocol_output_dir.resolve()),
        "compiled_manifest": compiled_manifest.model_dump(mode="json"),
        "run_results": [result.model_dump(mode="json") for result in run_results],
        "n_completed": int(n_completed),
        "n_failed": int(n_failed),
        "n_planned": int(n_planned),
        "artifact_paths": artifact_paths,
    }


def compile_and_run_protocol(
    *,
    protocol: ThesisProtocol,
    index_csv: Path | str,
    data_root: Path | str,
    cache_dir: Path | str,
    reports_root: Path | str,
    suite_ids: list[str] | None = None,
    force: bool = False,
    resume: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    compiled_manifest = compile_protocol(
        protocol,
        index_csv=index_csv,
        suite_ids=suite_ids,
    )
    return execute_compiled_protocol(
        protocol=protocol,
        compiled_manifest=compiled_manifest,
        index_csv=index_csv,
        data_root=data_root,
        cache_dir=cache_dir,
        reports_root=reports_root,
        force=force,
        resume=resume,
        dry_run=dry_run,
    )

