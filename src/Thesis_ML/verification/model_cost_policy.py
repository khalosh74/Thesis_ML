from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from Thesis_ML.comparisons.compiler import compile_comparison
from Thesis_ML.comparisons.loader import load_comparison_spec
from Thesis_ML.comparisons.models import ComparisonSpec, CompiledComparisonManifest
from Thesis_ML.experiments.model_catalog import model_catalog_snapshot
from Thesis_ML.protocols.compiler import compile_protocol
from Thesis_ML.protocols.loader import load_protocol
from Thesis_ML.protocols.models import CompiledProtocolManifest, ThesisProtocol


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _issue(code: str, message: str, *, details: dict[str, Any] | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {"code": str(code), "message": str(message)}
    if details:
        payload["details"] = dict(details)
    return payload


def _validate_protocol_manifest(
    *,
    protocol: ThesisProtocol,
    manifest: CompiledProtocolManifest,
    issues: list[dict[str, Any]],
) -> dict[str, Any]:
    tier_counts: dict[str, int] = {}
    total_runtime_seconds = 0
    max_runtime_seconds = 0
    allowed_tiers = {tier.value for tier in protocol.model_cost_policy.allowed_tiers}
    max_allowed_runtime = int(protocol.model_cost_policy.max_projected_runtime_seconds_per_run)

    for run in manifest.runs:
        tier = str(run.model_cost_tier.value)
        projected_runtime = int(run.projected_runtime_seconds)
        tier_counts[tier] = int(tier_counts.get(tier, 0)) + 1
        total_runtime_seconds += projected_runtime
        max_runtime_seconds = max(max_runtime_seconds, projected_runtime)
        if tier not in allowed_tiers:
            issues.append(
                _issue(
                    "confirmatory_model_tier_disallowed",
                    "Confirmatory run uses disallowed model cost tier.",
                    details={
                        "run_id": run.run_id,
                        "model": run.model,
                        "model_cost_tier": tier,
                        "allowed_tiers": sorted(allowed_tiers),
                    },
                )
            )
        if projected_runtime > max_allowed_runtime:
            issues.append(
                _issue(
                    "confirmatory_projected_runtime_exceeds_policy",
                    "Confirmatory run projected runtime exceeds policy threshold.",
                    details={
                        "run_id": run.run_id,
                        "model": run.model,
                        "projected_runtime_seconds": projected_runtime,
                        "max_projected_runtime_seconds_per_run": max_allowed_runtime,
                    },
                )
            )

    return {
        "status": "passed",
        "protocol_id": protocol.protocol_id,
        "protocol_version": protocol.protocol_version,
        "n_runs": int(len(manifest.runs)),
        "allowed_tiers": sorted(allowed_tiers),
        "max_projected_runtime_seconds_per_run": int(max_allowed_runtime),
        "total_projected_runtime_seconds": int(total_runtime_seconds),
        "max_projected_runtime_seconds": int(max_runtime_seconds),
        "run_counts_by_cost_tier": dict(sorted(tier_counts.items())),
    }


def _validate_comparison_manifest(
    *,
    comparison_spec_path: Path,
    comparison: ComparisonSpec,
    manifest: CompiledComparisonManifest,
    issues: list[dict[str, Any]],
) -> dict[str, Any]:
    tier_counts: dict[str, int] = {}
    total_runtime_seconds = 0
    max_runtime_seconds = 0
    max_allowed_runtime = int(comparison.cost_policy.max_projected_runtime_seconds_per_run)
    explicit_expensive = sorted(comparison.cost_policy.explicit_benchmark_expensive_models)

    for run in manifest.runs:
        tier = str(run.model_cost_tier.value)
        projected_runtime = int(run.projected_runtime_seconds)
        tier_counts[tier] = int(tier_counts.get(tier, 0)) + 1
        total_runtime_seconds += projected_runtime
        max_runtime_seconds = max(max_runtime_seconds, projected_runtime)
        if projected_runtime > max_allowed_runtime:
            issues.append(
                _issue(
                    "comparison_projected_runtime_exceeds_policy",
                    "Comparison run projected runtime exceeds policy threshold.",
                    details={
                        "comparison_spec": str(comparison_spec_path),
                        "run_id": run.run_id,
                        "variant_id": run.variant_id,
                        "model": run.model,
                        "projected_runtime_seconds": projected_runtime,
                        "max_projected_runtime_seconds_per_run": max_allowed_runtime,
                    },
                )
            )

    return {
        "status": "passed",
        "comparison_spec": str(comparison_spec_path),
        "comparison_id": comparison.comparison_id,
        "comparison_version": comparison.comparison_version,
        "n_runs": int(len(manifest.runs)),
        "explicit_benchmark_expensive_models": explicit_expensive,
        "max_projected_runtime_seconds_per_run": int(max_allowed_runtime),
        "total_projected_runtime_seconds": int(total_runtime_seconds),
        "max_projected_runtime_seconds": int(max_runtime_seconds),
        "run_counts_by_cost_tier": dict(sorted(tier_counts.items())),
    }


def verify_model_cost_policy_precheck(
    *,
    index_csv: Path | str,
    confirmatory_protocol: Path | str,
    comparison_specs: list[Path | str],
) -> dict[str, Any]:
    index_csv_path = Path(index_csv).resolve()
    protocol_path = Path(confirmatory_protocol).resolve()
    comparison_paths = [Path(path).resolve() for path in comparison_specs]
    issues: list[dict[str, Any]] = []

    confirmatory_summary: dict[str, Any]
    try:
        protocol = load_protocol(protocol_path)
        protocol_manifest = compile_protocol(protocol, index_csv=index_csv_path)
        confirmatory_summary = _validate_protocol_manifest(
            protocol=protocol,
            manifest=protocol_manifest,
            issues=issues,
        )
    except Exception as exc:
        confirmatory_summary = {
            "status": "failed",
            "protocol_path": str(protocol_path),
            "error": str(exc),
        }
        issues.append(
            _issue(
                "confirmatory_compile_failed",
                "Confirmatory protocol cost-policy precheck compilation failed.",
                details={"protocol_path": str(protocol_path), "error": str(exc)},
            )
        )

    comparison_summaries: list[dict[str, Any]] = []
    for comparison_path in comparison_paths:
        try:
            comparison = load_comparison_spec(comparison_path)
            comparison_manifest = compile_comparison(comparison, index_csv=index_csv_path)
            summary = _validate_comparison_manifest(
                comparison_spec_path=comparison_path,
                comparison=comparison,
                manifest=comparison_manifest,
                issues=issues,
            )
            comparison_summaries.append(summary)
        except Exception as exc:
            comparison_summaries.append(
                {
                    "status": "failed",
                    "comparison_spec": str(comparison_path),
                    "error": str(exc),
                }
            )
            issues.append(
                _issue(
                    "comparison_compile_failed",
                    "Comparison cost-policy precheck compilation failed.",
                    details={"comparison_spec": str(comparison_path), "error": str(exc)},
                )
            )

    aggregate_tier_counts: dict[str, int] = {}
    aggregate_total_runtime_seconds = 0
    aggregate_max_runtime_seconds = 0
    aggregate_n_runs = 0

    for summary in [confirmatory_summary, *comparison_summaries]:
        if str(summary.get("status")) != "passed":
            continue
        aggregate_n_runs += int(summary.get("n_runs", 0))
        aggregate_total_runtime_seconds += int(summary.get("total_projected_runtime_seconds", 0))
        aggregate_max_runtime_seconds = max(
            aggregate_max_runtime_seconds,
            int(summary.get("max_projected_runtime_seconds", 0)),
        )
        tier_counts = summary.get("run_counts_by_cost_tier", {})
        if isinstance(tier_counts, dict):
            for key, value in tier_counts.items():
                tier_key = str(key)
                aggregate_tier_counts[tier_key] = int(aggregate_tier_counts.get(tier_key, 0)) + int(
                    value
                )

    return {
        "schema_version": "model-cost-policy-precheck-v1",
        "generated_at_utc": _utc_now(),
        "passed": bool(not issues),
        "inputs": {
            "index_csv": str(index_csv_path),
            "confirmatory_protocol": str(protocol_path),
            "comparison_specs": [str(path) for path in comparison_paths],
        },
        "model_catalog": model_catalog_snapshot(),
        "confirmatory": confirmatory_summary,
        "comparisons": comparison_summaries,
        "aggregate": {
            "n_runs": int(aggregate_n_runs),
            "total_projected_runtime_seconds": int(aggregate_total_runtime_seconds),
            "max_projected_runtime_seconds": int(aggregate_max_runtime_seconds),
            "run_counts_by_cost_tier": dict(sorted(aggregate_tier_counts.items())),
        },
        "issues": issues,
    }


__all__ = ["verify_model_cost_policy_precheck"]
