from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from Thesis_ML.release.adapter import build_release_adapter_plan
from Thesis_ML.release.evidence import verify_release_evidence
from Thesis_ML.release.manifests import (
    build_release_manifest,
    build_run_manifest,
    write_json,
)
from Thesis_ML.release.models import RunClass, RunStatus
from Thesis_ML.release.paths import ensure_run_directories, resolve_run_paths
from Thesis_ML.release.runtime_protocol.runner import compile_and_run_protocol
from Thesis_ML.release.scope import (
    compile_release_scope,
    verify_scope_execution_alignment,
)
from Thesis_ML.release.states import is_runner_allowed_run_class
from Thesis_ML.release.validator import validate_release


def _timestamp_run_id(prefix: str = "run") -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}_{stamp}"


def _emit_release_event(
    *,
    event_callback: Callable[[dict[str, Any]], None] | None,
    event_name: str,
    **payload: Any,
) -> None:
    if event_callback is None:
        return
    event_payload = {
        "event": str(event_name),
        "timestamp_utc": datetime.now(UTC).isoformat(),
        **dict(payload),
    }
    try:
        event_callback(event_payload)
    except Exception:
        # Terminal observability must not alter scientific execution behavior.
        return


def _write_dataset_snapshot(
    *,
    run_dir: Path,
    dataset_manifest_path: Path,
    dataset_fingerprint: str,
    index_csv_path: Path,
    data_root_path: Path,
    cache_dir_path: Path,
    compiled_scope_summary: dict[str, Any],
    exclusions_summary: dict[str, Any],
) -> Path:
    payload = {
        "schema_version": "release-dataset-snapshot-v1",
        "dataset_manifest_path": str(dataset_manifest_path.resolve()),
        "dataset_fingerprint": dataset_fingerprint,
        "index_csv_path": str(index_csv_path.resolve()),
        "data_root_path": str(data_root_path.resolve()),
        "cache_dir_path": str(cache_dir_path.resolve()),
        "exact_selected_scope_summary": dict(compiled_scope_summary),
        "exact_exclusions_summary": dict(exclusions_summary),
    }
    return write_json(run_dir / "dataset_snapshot.json", payload)


def _write_release_summary(
    *,
    run_dir: Path,
    protocol_result: dict[str, Any],
    adapter_alignment_report: dict[str, Any],
    compiled_scope_manifest: dict[str, Any],
    dry_run: bool,
) -> Path:
    scope_counts = compiled_scope_manifest.get("counts", {})
    payload = {
        "schema_version": "release-summary-v1",
        "release_mode": "release_runner",
        "dry_run": bool(dry_run),
        "protocol_id": protocol_result.get("protocol_id"),
        "protocol_version": protocol_result.get("protocol_version"),
        "protocol_output_dir": protocol_result.get("protocol_output_dir"),
        "n_success": protocol_result.get("n_success"),
        "n_failed": protocol_result.get("n_failed"),
        "n_timed_out": protocol_result.get("n_timed_out"),
        "n_skipped_due_to_policy": protocol_result.get("n_skipped_due_to_policy"),
        "n_planned": protocol_result.get("n_planned"),
        "max_parallel_runs_effective": protocol_result.get("max_parallel_runs_effective"),
        "artifact_paths": protocol_result.get("artifact_paths"),
        "artifact_verification": protocol_result.get("artifact_verification"),
        "alignment_report": adapter_alignment_report,
        "selected_row_count": int(compiled_scope_manifest.get("selected_row_count", 0)),
        "selected_sample_ids_sha256": str(
            compiled_scope_manifest.get("selected_sample_ids_sha256", "")
        ),
        "scope_task_counts": dict(scope_counts.get("by_task", {}))
        if isinstance(scope_counts, dict)
        else {},
        "scope_modality_counts": dict(scope_counts.get("by_modality", {}))
        if isinstance(scope_counts, dict)
        else {},
        "scope_subject_counts": dict(scope_counts.get("by_subject", {}))
        if isinstance(scope_counts, dict)
        else {},
        "scope_session_counts": dict(scope_counts.get("by_session", {}))
        if isinstance(scope_counts, dict)
        else {},
    }
    return write_json(run_dir / "release_summary.json", payload)


def _write_warnings_and_deviations(
    *,
    run_dir: Path,
    protocol_output_dir: Path | None,
    exception_message: str | None = None,
) -> tuple[Path, Path]:
    warnings_payload: dict[str, Any]
    deviations_payload: dict[str, Any]
    if protocol_output_dir is None:
        warnings_payload = {
            "warnings": [],
            "n_warnings": 0,
            "note": "protocol output unavailable",
        }
        deviations_payload = {
            "deviations": [],
            "n_total_deviations": 0,
            "exception": exception_message,
        }
    else:
        execution_status_path = protocol_output_dir / "execution_status.json"
        deviation_log_path = protocol_output_dir / "deviation_log.json"
        execution_status = (
            json.loads(execution_status_path.read_text(encoding="utf-8"))
            if execution_status_path.exists()
            else {}
        )
        deviation_log = (
            json.loads(deviation_log_path.read_text(encoding="utf-8"))
            if deviation_log_path.exists()
            else {}
        )
        warnings_payload = {
            "warnings": execution_status.get("warnings", []),
            "n_warnings": len(execution_status.get("warnings", []))
            if isinstance(execution_status.get("warnings"), list)
            else 0,
        }
        deviations_payload = {
            "deviations": deviation_log.get("deviations", [])
            if isinstance(deviation_log.get("deviations"), list)
            else [],
            "n_total_deviations": int(deviation_log.get("n_total_deviations", 0)),
            "science_critical_deviation_detected": bool(
                deviation_log.get("science_critical_deviation_detected", False)
            ),
        }
        if exception_message:
            deviations_payload["exception"] = exception_message
    warnings_path = write_json(run_dir / "warnings.json", warnings_payload)
    deviations_path = write_json(run_dir / "deviations.json", deviations_payload)
    return warnings_path, deviations_path


def _materialize_required_run_alias_artifacts(
    *,
    protocol_output_dir: Path,
) -> None:
    report_index_path = protocol_output_dir / "report_index.csv"
    if not report_index_path.exists():
        return
    index_df = pd.read_csv(report_index_path)
    if "status" not in index_df.columns or "report_dir" not in index_df.columns:
        return
    alias_map = {
        "subgroup_report.csv": "subgroup_metrics.csv",
        "spatial_report.json": "spatial_compatibility_report.json",
    }
    success_rows = index_df[index_df["status"].astype(str).str.lower() == "success"]
    for _, row in success_rows.iterrows():
        raw_report_dir = str(row["report_dir"]).strip()
        if not raw_report_dir:
            continue
        report_dir = Path(raw_report_dir)
        if not report_dir.is_absolute():
            report_dir = (protocol_output_dir / report_dir).resolve()
        if not report_dir.exists():
            continue
        for alias_name, source_name in alias_map.items():
            alias_path = report_dir / alias_name
            source_path = report_dir / source_name
            if alias_path.exists() or not source_path.exists():
                continue
            shutil.copy2(source_path, alias_path)


def run_release(
    *,
    release_ref: Path | str,
    dataset_manifest_path: Path | str,
    run_class: RunClass,
    force: bool = False,
    resume: bool = False,
    dry_run: bool = False,
    command_line: list[str] | None = None,
    run_id: str | None = None,
    event_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    _emit_release_event(
        event_callback=event_callback,
        event_name="release_runner.start",
        release_ref=str(release_ref),
        dataset_manifest_path=str(dataset_manifest_path),
        run_class=str(run_class.value),
        dry_run=bool(dry_run),
        force=bool(force),
        resume=bool(resume),
    )
    if not is_runner_allowed_run_class(run_class):
        raise ValueError(
            "Release runner only supports scratch/exploratory/candidate. "
            f"Received '{run_class.value}'."
        )

    validated = validate_release(
        release_ref=release_ref,
        dataset_manifest_path=dataset_manifest_path,
        strict_environment=False,
    )
    if not validated.passed:
        raise ValueError(
            "Release validation failed before execution: "
            + "; ".join(str(issue["message"]) for issue in validated.issues[:8])
        )

    release = validated.release
    dataset = validated.dataset
    _emit_release_event(
        event_callback=event_callback,
        event_name="release_runner.validation_passed",
        release_id=str(release.release.release_id),
        release_version=str(release.release.release_version),
    )

    if resume and run_class not in set(release.execution.allow_resume_for):
        raise ValueError(f"--resume is not allowed for run_class='{run_class.value}'.")
    if force and run_class not in set(release.execution.allow_force_for):
        raise ValueError(f"--force is not allowed for run_class='{run_class.value}'.")

    resolved_run_id = run_id or _timestamp_run_id(prefix=run_class.value)
    run_paths = resolve_run_paths(
        execution=release.execution,
        run_class=run_class,
        release_id=release.release.release_id,
        run_id=resolved_run_id,
    )
    if run_paths.root.exists():
        if force:
            shutil.rmtree(run_paths.root)
        else:
            raise FileExistsError(
                f"Run directory already exists: {run_paths.root}. Use --force to replace."
            )
    ensure_run_directories(run_paths)
    _emit_release_event(
        event_callback=event_callback,
        event_name="release_runner.run_directory_ready",
        run_dir=str(run_paths.root.resolve()),
    )

    compiled_scope_result = compile_release_scope(
        release_bundle=release,
        dataset_manifest=dataset,
        run_dir=run_paths.root,
    )
    compiled_scope_manifest_payload = json.loads(
        compiled_scope_result.scope_manifest_path.read_text(encoding="utf-8")
    )
    if not isinstance(compiled_scope_manifest_payload, dict):
        raise ValueError(
            "Compiled scope manifest must be a JSON object: "
            f"{compiled_scope_result.scope_manifest_path}"
        )
    _emit_release_event(
        event_callback=event_callback,
        event_name="release_runner.scope_compiled",
        selected_row_count=int(compiled_scope_manifest_payload.get("selected_row_count", 0)),
        selected_sample_ids_sha256=str(
            compiled_scope_manifest_payload.get("selected_sample_ids_sha256", "")
        ),
        scope_manifest_path=str(compiled_scope_result.scope_manifest_path.resolve()),
    )

    release_manifest = build_release_manifest(
        release_bundle=release.release,
        release_json_path=release.release_path,
        science_json_path=release.science_path,
        execution_json_path=release.execution_path,
        environment_json_path=release.environment_path,
        evidence_json_path=release.evidence_path,
        claims_json_path=release.claims_path,
        hashes=release.hashes,
    )
    write_json(run_paths.root / "release_manifest.json", release_manifest.model_dump(mode="json"))

    run_manifest = build_run_manifest(
        run_id=resolved_run_id,
        run_class=run_class,
        release_bundle=release.release,
        hashes=release.hashes,
        dataset_manifest_path=dataset.manifest_path,
        dataset_fingerprint=dataset.manifest.dataset_fingerprint,
        cache_policy=release.execution.cache_policy,
        command_line=command_line,
        parent_run_id=None,
        status=RunStatus.CREATED,
        promotable=False,
        official=False,
        evidence_verified=False,
        compiled_scope_manifest_path=str(compiled_scope_result.scope_manifest_path.resolve()),
        selected_samples_path=str(compiled_scope_result.selected_samples_path.resolve()),
        selected_sample_ids_sha256=str(
            compiled_scope_manifest_payload.get("selected_sample_ids_sha256", "")
        )
        or None,
        scope_alignment_passed=False,
    )
    write_json(run_paths.root / "run_manifest.json", run_manifest.model_dump(mode="json"))

    _write_dataset_snapshot(
        run_dir=run_paths.root,
        dataset_manifest_path=dataset.manifest_path,
        dataset_fingerprint=dataset.manifest.dataset_fingerprint,
        index_csv_path=dataset.index_csv_path,
        data_root_path=dataset.data_root_path,
        cache_dir_path=dataset.cache_dir_path,
        compiled_scope_summary={
            "selected_row_count": int(compiled_scope_manifest_payload.get("selected_row_count", 0)),
            "selected_sample_ids_sha256": str(
                compiled_scope_manifest_payload.get("selected_sample_ids_sha256", "")
            ),
            "scope_subjects": list(compiled_scope_manifest_payload.get("scope_subjects", [])),
            "scope_tasks": list(compiled_scope_manifest_payload.get("scope_tasks", [])),
            "scope_modality": str(compiled_scope_manifest_payload.get("scope_modality", "")),
            "counts": dict(compiled_scope_manifest_payload.get("counts", {})),
        },
        exclusions_summary=dict(compiled_scope_manifest_payload.get("exclusions_summary", {})),
    )

    protocol_output_dir: Path | None = None
    try:
        run_manifest = run_manifest.model_copy(
            update={"status": RunStatus.RUNNING},
        )
        write_json(run_paths.root / "run_manifest.json", run_manifest.model_dump(mode="json"))

        adapter_plan = build_release_adapter_plan(
            release=release,
            dataset=dataset,
            compiled_scope=compiled_scope_result,
        )
        _emit_release_event(
            event_callback=event_callback,
            event_name="release_runner.protocol_execution_start",
            n_compiled_runs=int(len(adapter_plan.compiled_manifest.runs)),
            max_parallel_runs=int(release.execution.max_parallel_runs),
        )
        protocol_result = compile_and_run_protocol(
            protocol=adapter_plan.protocol,
            index_csv=adapter_plan.scoped_index_csv,
            data_root=dataset.data_root_path,
            cache_dir=dataset.cache_dir_path,
            reports_root=run_paths.artifacts_dir,
            suite_ids=None,
            force=bool(force),
            resume=bool(resume),
            dry_run=bool(dry_run),
            max_parallel_runs=int(release.execution.max_parallel_runs),
            hardware_mode=str(release.execution.hardware_mode),
            gpu_device_id=None,
            deterministic_compute=bool(release.execution.deterministic_compute),
            allow_backend_fallback=bool(release.execution.allow_backend_fallback),
            protocol_context_overrides_by_run_id=adapter_plan.protocol_context_overrides_by_run_id,
            runtime_event_callback=event_callback,
        )
        _emit_release_event(
            event_callback=event_callback,
            event_name="release_runner.protocol_execution_complete",
            n_success=int(protocol_result.get("n_success", 0)),
            n_failed=int(protocol_result.get("n_failed", 0)),
            n_timed_out=int(protocol_result.get("n_timed_out", 0)),
            n_planned=int(protocol_result.get("n_planned", 0)),
        )
        protocol_output_dir = Path(str(protocol_result["protocol_output_dir"])).resolve()
        _materialize_required_run_alias_artifacts(protocol_output_dir=protocol_output_dir)
        scope_alignment_verification = verify_scope_execution_alignment(
            run_dir=run_paths.root,
            compiled_scope_manifest_path=compiled_scope_result.scope_manifest_path,
            expected_science_hash=release.hashes.science_hash,
            expected_target_mapping_hash=release.science.target.mapping_hash,
            write_output=True,
        )
        if not bool(scope_alignment_verification.get("passed", False)):
            raise ValueError(
                "Scope alignment verification failed: "
                + "; ".join(
                    str(issue.get("message"))
                    for issue in scope_alignment_verification.get("issues", [])[:8]
                    if isinstance(issue, dict)
                )
            )
        _emit_release_event(
            event_callback=event_callback,
            event_name="release_runner.scope_alignment_passed",
            selected_sample_ids_sha256=str(
                scope_alignment_verification.get("selected_sample_ids_sha256", "")
            ),
        )
        run_manifest = run_manifest.model_copy(update={"scope_alignment_passed": True})
        _write_release_summary(
            run_dir=run_paths.root,
            protocol_result=protocol_result,
            adapter_alignment_report=adapter_plan.alignment_report,
            compiled_scope_manifest=compiled_scope_manifest_payload,
            dry_run=bool(dry_run),
        )
        _write_warnings_and_deviations(
            run_dir=run_paths.root,
            protocol_output_dir=protocol_output_dir,
        )

        verification = verify_release_evidence(
            run_dir=run_paths.root,
            release=release,
            dataset=dataset,
            run_manifest=run_manifest,
            allow_missing_evidence_verification=True,
            write_output=True,
        )
        write_json(run_paths.root / "evidence_verification.json", verification)
        if not bool(verification.get("passed", False)):
            raise ValueError(
                "Release evidence verification failed: "
                + "; ".join(str(issue["message"]) for issue in verification.get("issues", [])[:8])
            )
        _emit_release_event(
            event_callback=event_callback,
            event_name="release_runner.evidence_verification_passed",
        )

        run_manifest = run_manifest.model_copy(
            update={
                "status": RunStatus.SUCCEEDED,
                "promotable": bool(run_class == RunClass.CANDIDATE),
                "official": False,
                "evidence_verified": True,
                "scope_alignment_passed": True,
            }
        )
        write_json(run_paths.root / "run_manifest.json", run_manifest.model_dump(mode="json"))
        _emit_release_event(
            event_callback=event_callback,
            event_name="release_runner.succeeded",
            run_id=str(run_manifest.run_id),
            run_dir=str(run_paths.root.resolve()),
            promotable=bool(run_manifest.promotable),
        )
        return {
            "passed": True,
            "run_id": run_manifest.run_id,
            "run_class": run_manifest.run_class.value,
            "run_dir": str(run_paths.root.resolve()),
            "promotable": bool(run_manifest.promotable),
            "official": bool(run_manifest.official),
            "release_id": run_manifest.release_id,
            "release_version": run_manifest.release_version,
        }
    except Exception as exc:
        _write_warnings_and_deviations(
            run_dir=run_paths.root,
            protocol_output_dir=protocol_output_dir,
            exception_message=str(exc),
        )
        failure_verification = verify_release_evidence(
            run_dir=run_paths.root,
            release=release,
            dataset=dataset,
            run_manifest=run_manifest,
            allow_missing_evidence_verification=True,
            write_output=True,
        )
        write_json(run_paths.root / "evidence_verification.json", failure_verification)

        run_manifest = run_manifest.model_copy(
            update={
                "status": RunStatus.FAILED,
                "promotable": False,
                "official": False,
                "evidence_verified": bool(failure_verification.get("passed", False)),
                "scope_alignment_passed": False,
            }
        )
        write_json(run_paths.root / "run_manifest.json", run_manifest.model_dump(mode="json"))
        _emit_release_event(
            event_callback=event_callback,
            event_name="release_runner.failed",
            run_id=str(run_manifest.run_id),
            run_dir=str(run_paths.root.resolve()),
            error=str(exc),
        )
        raise


__all__ = ["run_release"]
