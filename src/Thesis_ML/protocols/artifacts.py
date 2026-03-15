from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from Thesis_ML.config.framework_mode import FrameworkMode
from Thesis_ML.protocols.models import (
    CompiledProtocolManifest,
    ProtocolRunResult,
    ThesisProtocol,
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{json.dumps(payload, indent=2)}\n", encoding="utf-8")


def _suite_summary(
    compiled_manifest: CompiledProtocolManifest,
    run_results: list[ProtocolRunResult],
) -> dict[str, Any]:
    by_suite: dict[str, dict[str, int]] = {}
    for suite_id in compiled_manifest.suite_ids:
        by_suite[suite_id] = {"planned": 0, "completed": 0, "failed": 0}
    for result in run_results:
        status_counts = by_suite.setdefault(
            result.suite_id,
            {"planned": 0, "completed": 0, "failed": 0},
        )
        status_counts[result.status] = int(status_counts.get(result.status, 0)) + 1
    return {
        "framework_mode": FrameworkMode.CONFIRMATORY.value,
        "protocol_id": compiled_manifest.protocol_id,
        "protocol_version": compiled_manifest.protocol_version,
        "methodology_policy_name": compiled_manifest.methodology_policy.policy_name.value,
        "primary_metric": compiled_manifest.metric_policy.primary_metric,
        "subgroup_reporting_enabled": bool(compiled_manifest.subgroup_reporting_policy.enabled),
        "suite_status_counts": by_suite,
        "n_runs": int(len(run_results)),
    }


def _execution_status_payload(
    protocol: ThesisProtocol,
    compiled_manifest: CompiledProtocolManifest,
    run_results: list[ProtocolRunResult],
    *,
    dry_run: bool,
) -> dict[str, Any]:
    return {
        "framework_mode": FrameworkMode.CONFIRMATORY.value,
        "protocol_id": protocol.protocol_id,
        "protocol_version": protocol.protocol_version,
        "protocol_schema_version": protocol.protocol_schema_version,
        "compiled_schema_version": compiled_manifest.compiled_schema_version,
        "dry_run": bool(dry_run),
        "runs": [result.model_dump(mode="json") for result in run_results],
    }


def _report_index_rows(
    compiled_manifest: CompiledProtocolManifest,
    run_results: list[ProtocolRunResult],
) -> list[dict[str, Any]]:
    result_by_run_id = {result.run_id: result for result in run_results}
    rows: list[dict[str, Any]] = []
    for spec in compiled_manifest.runs:
        result = result_by_run_id.get(spec.run_id)
        metrics = result.metrics if result and result.metrics else {}
        rows.append(
            {
                "run_id": spec.run_id,
                "framework_mode": FrameworkMode.CONFIRMATORY.value,
                "suite_id": spec.suite_id,
                "claim_ids": "|".join(spec.claim_ids),
                "status": result.status if result is not None else "planned",
                "target": spec.target,
                "model": spec.model,
                "cv_mode": spec.cv_mode,
                "subject": spec.subject,
                "train_subject": spec.train_subject,
                "test_subject": spec.test_subject,
                "seed": int(spec.seed),
                "primary_metric": spec.primary_metric,
                "methodology_policy_name": spec.methodology_policy_name.value,
                "class_weight_policy": spec.class_weight_policy.value,
                "tuning_enabled": bool(spec.tuning_enabled),
                "permutation_enabled": bool(spec.controls.permutation_enabled),
                "n_permutations": int(spec.controls.n_permutations),
                "permutation_metric": spec.controls.permutation_metric,
                "dummy_baseline_run": bool(spec.controls.dummy_baseline_run),
                "interpretability_enabled": bool(spec.interpretability_enabled),
                "subgroup_reporting_enabled": bool(spec.subgroup_reporting_enabled),
                "canonical_run": bool(spec.canonical_run),
                "report_dir": result.report_dir if result is not None else None,
                "config_path": result.config_path if result is not None else None,
                "metrics_path": result.metrics_path if result is not None else None,
                "error": result.error if result is not None else None,
                "balanced_accuracy": metrics.get("balanced_accuracy"),
                "macro_f1": metrics.get("macro_f1"),
                "accuracy": metrics.get("accuracy"),
            }
        )
    return rows


def write_protocol_artifacts(
    *,
    protocol: ThesisProtocol,
    compiled_manifest: CompiledProtocolManifest,
    run_results: list[ProtocolRunResult],
    output_dir: Path | str,
    dry_run: bool,
) -> dict[str, str]:
    protocol_dir = Path(output_dir)
    protocol_dir.mkdir(parents=True, exist_ok=True)

    protocol_json_path = protocol_dir / "protocol.json"
    compiled_manifest_path = protocol_dir / "compiled_protocol_manifest.json"
    claim_map_path = protocol_dir / "claim_to_run_map.json"
    suite_summary_path = protocol_dir / "suite_summary.json"
    execution_status_path = protocol_dir / "execution_status.json"
    report_index_path = protocol_dir / "report_index.csv"

    _write_json(protocol_json_path, protocol.model_dump(mode="json"))
    _write_json(compiled_manifest_path, compiled_manifest.model_dump(mode="json"))
    _write_json(claim_map_path, compiled_manifest.claim_to_run_map)
    _write_json(suite_summary_path, _suite_summary(compiled_manifest, run_results))
    _write_json(
        execution_status_path,
        _execution_status_payload(
            protocol=protocol,
            compiled_manifest=compiled_manifest,
            run_results=run_results,
            dry_run=dry_run,
        ),
    )

    report_rows = _report_index_rows(compiled_manifest, run_results)
    with report_index_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(report_rows[0].keys()) if report_rows else ["run_id"],
        )
        writer.writeheader()
        for row in report_rows:
            writer.writerow(row)

    return {
        "protocol_json": str(protocol_json_path.resolve()),
        "compiled_protocol_manifest": str(compiled_manifest_path.resolve()),
        "claim_to_run_map": str(claim_map_path.resolve()),
        "suite_summary": str(suite_summary_path.resolve()),
        "execution_status": str(execution_status_path.resolve()),
        "report_index": str(report_index_path.resolve()),
    }
