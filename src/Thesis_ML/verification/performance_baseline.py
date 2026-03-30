from __future__ import annotations

import csv
import hashlib
import json
import platform
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd
import sklearn

from Thesis_ML.experiments.provenance import collect_git_provenance

_BASELINE_SCHEMA_VERSION = "performance-baseline-bundle-v1"

_NON_SEMANTIC_KEYS = {
    "timestamp",
    "timestamp_utc",
    "generated_at_utc",
    "updated_at_utc",
    "duration_seconds",
    "elapsed_seconds",
    "wall_clock_elapsed_seconds",
    "resource_summary",
    "warning_summary",
    "warnings",
    "sample_count",
    "first_sample_at_utc",
    "last_sample_at_utc",
}

_PATH_KEYS = {
    "path",
    "report_dir",
    "reports_root",
    "cache_dir",
    "data_root",
    "index_csv",
    "run_status_path",
}


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _coerce_bundle_payload(
    value: BaselineBundle | Mapping[str, Any] | Path | str,
) -> dict[str, Any]:
    if isinstance(value, BaselineBundle):
        return value.to_payload()
    if isinstance(value, (Path, str)):
        candidate = Path(value)
        if candidate.is_dir():
            candidate = candidate / "baseline_bundle.json"
        return _load_json(candidate)
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError(f"Unsupported baseline bundle input type: {type(value)!r}")


def _json_ready(value: Any) -> Any:
    if is_dataclass(value):
        return _json_ready(asdict(value))
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, Mapping):
        return {str(key): _json_ready(raw) for key, raw in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _normalize_for_hash(
    value: Any,
    *,
    drop_timing_keys: bool = True,
) -> Any:
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for raw_key, raw_value in value.items():
            key = str(raw_key)
            key_lower = key.lower()
            if key_lower in _NON_SEMANTIC_KEYS and drop_timing_keys:
                continue
            if key_lower in _PATH_KEYS:
                continue
            if key_lower.endswith("_path") or key_lower.endswith("_path_relative"):
                continue
            if key_lower.endswith("_dir") or key_lower.endswith("_dir_relative"):
                continue
            if key_lower.endswith("_at_utc"):
                continue
            normalized[key] = _normalize_for_hash(
                raw_value,
                drop_timing_keys=drop_timing_keys,
            )
        return normalized
    if isinstance(value, list):
        return [_normalize_for_hash(item, drop_timing_keys=drop_timing_keys) for item in value]
    if isinstance(value, tuple):
        return tuple(
            _normalize_for_hash(item, drop_timing_keys=drop_timing_keys) for item in value
        )
    return value


def _hash_payload(
    payload: Any,
    *,
    drop_timing_keys: bool = True,
) -> str:
    normalized = _normalize_for_hash(payload, drop_timing_keys=drop_timing_keys)
    return _sha256_text(_stable_json(normalized))


def _resolve_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return Path(text).resolve()


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [{str(key): str(value) for key, value in row.items()} for row in reader]


def _hash_csv(
    path: Path,
    *,
    sort_by: Sequence[str] | None = None,
) -> str:
    rows = _read_csv_rows(path)
    normalized_rows = [{str(key): str(value) for key, value in row.items()} for row in rows]
    if sort_by:
        normalized_rows = sorted(
            normalized_rows,
            key=lambda row: tuple(str(row.get(column, "")) for column in sort_by),
        )
    return _sha256_text(_stable_json(normalized_rows))


def _safe_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _extract_numeric_metrics(metrics_payload: Mapping[str, Any]) -> dict[str, float]:
    collected: dict[str, float] = {}
    for key, value in metrics_payload.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            collected[str(key)] = float(value)
    for key in ("primary_metric_value", "accuracy", "balanced_accuracy", "macro_f1"):
        candidate = _safe_float(metrics_payload.get(key))
        if candidate is not None:
            collected[str(key)] = float(candidate)
    return collected


def _profiling_observability(
    *,
    run_status_payload: Mapping[str, Any] | None,
    process_samples_path: Path | None,
    process_profile_summary_path: Path | None,
) -> dict[str, Any]:
    expected_artifacts = []
    artifact_presence: dict[str, bool] = {}
    if process_samples_path is not None:
        expected_artifacts.append(str(process_samples_path))
        artifact_presence[str(process_samples_path)] = bool(process_samples_path.exists())
    if process_profile_summary_path is not None:
        expected_artifacts.append(str(process_profile_summary_path))
        artifact_presence[str(process_profile_summary_path)] = bool(
            process_profile_summary_path.exists()
        )
    schema_checks = {
        "run_status_has_process_profile_summary": bool(
            isinstance(run_status_payload, Mapping)
            and isinstance(run_status_payload.get("process_profile_summary"), Mapping)
        ),
        "run_status_has_process_profile_artifacts": bool(
            isinstance(run_status_payload, Mapping)
            and isinstance(run_status_payload.get("process_profile_artifacts"), Mapping)
        ),
    }
    return {
        "expected_artifacts": expected_artifacts,
        "artifact_presence": artifact_presence,
        "schema_checks": schema_checks,
    }


@dataclass(frozen=True)
class EnvironmentSnapshot:
    python_version: str
    platform: str
    numpy_version: str
    pandas_version: str
    sklearn_version: str
    nibabel_version: str
    git_commit: str | None = None
    git_branch: str | None = None
    git_dirty: bool | None = None

    @classmethod
    def capture(cls) -> EnvironmentSnapshot:
        git_payload = collect_git_provenance()
        return cls(
            python_version=platform.python_version(),
            platform=platform.platform(),
            numpy_version=str(np.__version__),
            pandas_version=str(pd.__version__),
            sklearn_version=str(sklearn.__version__),
            nibabel_version=str(nib.__version__),
            git_commit=(
                str(git_payload.get("git_commit"))
                if isinstance(git_payload.get("git_commit"), str)
                else None
            ),
            git_branch=(
                str(git_payload.get("git_branch"))
                if isinstance(git_payload.get("git_branch"), str)
                else None
            ),
            git_dirty=(
                bool(git_payload.get("git_dirty"))
                if isinstance(git_payload.get("git_dirty"), bool)
                else None
            ),
        )

    def to_payload(self) -> dict[str, Any]:
        return _json_ready(asdict(self))


@dataclass(frozen=True)
class BenchmarkCaseSpec:
    case_id: str
    mode: str
    description: str
    inputs: dict[str, Any] = field(default_factory=dict)
    config_identity: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        return _json_ready(asdict(self))


@dataclass(frozen=True)
class BenchmarkFingerprint:
    config_fingerprint: str | None = None
    selected_sample_fingerprint: str | None = None
    fold_split_fingerprint: str | None = None
    prediction_fingerprint: str | None = None
    metrics_fingerprint: str | None = None
    tuning_fingerprint: str | None = None
    stage_execution_fingerprint: str | None = None

    def to_payload(self) -> dict[str, Any]:
        return _json_ready(asdict(self))


@dataclass(frozen=True)
class BenchmarkCaseResult:
    case_id: str
    status: str
    elapsed_seconds: float
    artifact_refs: dict[str, str] = field(default_factory=dict)
    fingerprints: BenchmarkFingerprint = field(default_factory=BenchmarkFingerprint)
    scientific_metrics: dict[str, float] = field(default_factory=dict)
    observability: dict[str, Any] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        payload = _json_ready(asdict(self))
        payload["elapsed_seconds"] = float(self.elapsed_seconds)
        return payload


@dataclass(frozen=True)
class BaselineBundle:
    schema_version: str
    generated_at_utc: str
    baseline_mode: str
    suite_id: str
    output_root: str
    environment: EnvironmentSnapshot
    case_specs: list[BenchmarkCaseSpec]
    case_results: list[BenchmarkCaseResult]
    aggregate_summary: dict[str, Any]
    comparison: dict[str, Any] | None = None

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": str(self.schema_version),
            "generated_at_utc": str(self.generated_at_utc),
            "baseline_mode": str(self.baseline_mode),
            "suite_id": str(self.suite_id),
            "output_root": str(self.output_root),
            "environment": self.environment.to_payload(),
            "case_specs": [case.to_payload() for case in self.case_specs],
            "case_results": [case.to_payload() for case in self.case_results],
            "aggregate_summary": _json_ready(dict(self.aggregate_summary)),
            "comparison": _json_ready(self.comparison) if self.comparison is not None else None,
        }


def build_benchmark_fingerprint(
    *,
    config_payload: Mapping[str, Any] | None = None,
    selected_samples_path: Path | str | None = None,
    fold_splits_path: Path | str | None = None,
    predictions_path: Path | str | None = None,
    metrics_payload: Mapping[str, Any] | None = None,
    tuning_payload: Mapping[str, Any] | None = None,
    stage_execution_payload: Mapping[str, Any] | None = None,
) -> BenchmarkFingerprint:
    selected_samples_resolved = _resolve_path(selected_samples_path)
    fold_splits_resolved = _resolve_path(fold_splits_path)
    predictions_resolved = _resolve_path(predictions_path)
    return BenchmarkFingerprint(
        config_fingerprint=(
            _hash_payload(config_payload, drop_timing_keys=True)
            if isinstance(config_payload, Mapping)
            else None
        ),
        selected_sample_fingerprint=(
            _hash_csv(selected_samples_resolved, sort_by=("sample_id",))
            if selected_samples_resolved is not None and selected_samples_resolved.exists()
            else None
        ),
        fold_split_fingerprint=(
            _hash_csv(
                fold_splits_resolved,
                sort_by=("fold", "sample_id", "split"),
            )
            if fold_splits_resolved is not None and fold_splits_resolved.exists()
            else None
        ),
        prediction_fingerprint=(
            _hash_csv(
                predictions_resolved,
                sort_by=("fold", "sample_id", "y_true", "y_pred"),
            )
            if predictions_resolved is not None and predictions_resolved.exists()
            else None
        ),
        metrics_fingerprint=(
            _hash_payload(metrics_payload, drop_timing_keys=True)
            if isinstance(metrics_payload, Mapping)
            else None
        ),
        tuning_fingerprint=(
            _hash_payload(tuning_payload, drop_timing_keys=True)
            if isinstance(tuning_payload, Mapping)
            else None
        ),
        stage_execution_fingerprint=(
            _hash_payload(stage_execution_payload, drop_timing_keys=True)
            if isinstance(stage_execution_payload, Mapping)
            else None
        ),
    )


def run_baseline_suite(
    *,
    suite_id: str,
    baseline_mode: str,
    output_root: Path | str,
    case_specs: Sequence[BenchmarkCaseSpec],
    case_runner: Any,
    compare_against: BaselineBundle | Mapping[str, Any] | Path | str | None = None,
    metrics_tolerance: float = 1e-8,
) -> BaselineBundle:
    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    case_results: list[BenchmarkCaseResult] = []

    for case_spec in case_specs:
        case_dir = root / str(case_spec.case_id)
        case_dir.mkdir(parents=True, exist_ok=True)
        case_start = perf_counter()
        result = case_runner(case_spec, case_dir)
        if isinstance(result, Mapping):
            result = BenchmarkCaseResult(
                case_id=str(result.get("case_id", case_spec.case_id)),
                status=str(result.get("status", "failed")),
                elapsed_seconds=float(
                    result.get("elapsed_seconds", perf_counter() - case_start)
                ),
                artifact_refs=dict(result.get("artifact_refs", {})),
                fingerprints=(
                    result.get("fingerprints")
                    if isinstance(result.get("fingerprints"), BenchmarkFingerprint)
                    else BenchmarkFingerprint(**dict(result.get("fingerprints", {})))
                ),
                scientific_metrics={
                    str(key): float(value)
                    for key, value in dict(result.get("scientific_metrics", {})).items()
                    if isinstance(value, (int, float))
                },
                observability=dict(result.get("observability", {})),
                summary=dict(result.get("summary", {})),
            )
        if not isinstance(result, BenchmarkCaseResult):
            raise TypeError(
                "case_runner must return BenchmarkCaseResult or Mapping-compatible payload."
            )
        if str(result.case_id) != str(case_spec.case_id):
            raise ValueError(
                f"Case runner returned mismatched case_id: expected={case_spec.case_id} got={result.case_id}"
            )
        case_results.append(result)

    passed_cases = [row for row in case_results if str(row.status) == "passed"]
    failed_cases = [row for row in case_results if str(row.status) != "passed"]
    aggregate_summary = {
        "n_cases": int(len(case_results)),
        "n_passed": int(len(passed_cases)),
        "n_failed": int(len(failed_cases)),
        "failed_case_ids": [str(row.case_id) for row in failed_cases],
    }

    comparison_payload = (
        compare_baseline_bundles(
            left_bundle=compare_against,
            right_bundle=BaselineBundle(
                schema_version=_BASELINE_SCHEMA_VERSION,
                generated_at_utc=_utc_now(),
                baseline_mode=str(baseline_mode),
                suite_id=str(suite_id),
                output_root=str(root),
                environment=EnvironmentSnapshot.capture(),
                case_specs=list(case_specs),
                case_results=list(case_results),
                aggregate_summary=dict(aggregate_summary),
                comparison=None,
            ),
            metrics_tolerance=float(metrics_tolerance),
        )
        if compare_against is not None
        else None
    )

    return BaselineBundle(
        schema_version=_BASELINE_SCHEMA_VERSION,
        generated_at_utc=_utc_now(),
        baseline_mode=str(baseline_mode),
        suite_id=str(suite_id),
        output_root=str(root),
        environment=EnvironmentSnapshot.capture(),
        case_specs=list(case_specs),
        case_results=list(case_results),
        aggregate_summary=aggregate_summary,
        comparison=comparison_payload,
    )


def write_baseline_bundle(
    bundle: BaselineBundle | Mapping[str, Any],
    output_path: Path | str,
) -> Path:
    path = Path(output_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(bundle, BaselineBundle):
        payload = bundle.to_payload()
    elif isinstance(bundle, Mapping):
        payload = _json_ready(dict(bundle))
    else:
        raise TypeError(f"Unsupported bundle type: {type(bundle)!r}")
    path.write_text(f"{json.dumps(payload, indent=2)}\n", encoding="utf-8")
    return path


def compare_baseline_bundles(
    *,
    left_bundle: BaselineBundle | Mapping[str, Any] | Path | str,
    right_bundle: BaselineBundle | Mapping[str, Any] | Path | str,
    metrics_tolerance: float = 1e-8,
) -> dict[str, Any]:
    left_payload = _coerce_bundle_payload(left_bundle)
    right_payload = _coerce_bundle_payload(right_bundle)

    left_cases = {
        str(case.get("case_id")): case
        for case in list(left_payload.get("case_results", []))
        if isinstance(case, Mapping) and case.get("case_id") is not None
    }
    right_cases = {
        str(case.get("case_id")): case
        for case in list(right_payload.get("case_results", []))
        if isinstance(case, Mapping) and case.get("case_id") is not None
    }

    left_case_ids = sorted(left_cases.keys())
    right_case_ids = sorted(right_cases.keys())
    common_case_ids = sorted(set(left_case_ids) & set(right_case_ids))

    scientific_mismatches: list[dict[str, Any]] = []
    observability_mismatches: list[dict[str, Any]] = []

    if left_case_ids != right_case_ids:
        scientific_mismatches.append(
            {
                "code": "case_set_mismatch",
                "left_case_ids": left_case_ids,
                "right_case_ids": right_case_ids,
            }
        )

    scientific_keys = (
        "selected_sample_fingerprint",
        "fold_split_fingerprint",
        "prediction_fingerprint",
        "stage_execution_fingerprint",
    )

    for case_id in common_case_ids:
        left_case = dict(left_cases[case_id])
        right_case = dict(right_cases[case_id])
        left_fp = dict(left_case.get("fingerprints", {}))
        right_fp = dict(right_case.get("fingerprints", {}))

        for key in scientific_keys:
            left_value = left_fp.get(key)
            right_value = right_fp.get(key)
            if left_value is None and right_value is None:
                continue
            if left_value != right_value:
                scientific_mismatches.append(
                    {
                        "code": "scientific_fingerprint_mismatch",
                        "case_id": case_id,
                        "field": key,
                        "left": left_value,
                        "right": right_value,
                    }
                )

        left_tuning = left_fp.get("tuning_fingerprint")
        right_tuning = right_fp.get("tuning_fingerprint")
        if left_tuning is not None or right_tuning is not None:
            if left_tuning != right_tuning:
                scientific_mismatches.append(
                    {
                        "code": "tuning_choice_mismatch",
                        "case_id": case_id,
                        "left": left_tuning,
                        "right": right_tuning,
                    }
                )

        left_metrics = {
            str(key): float(value)
            for key, value in dict(left_case.get("scientific_metrics", {})).items()
            if isinstance(value, (int, float))
        }
        right_metrics = {
            str(key): float(value)
            for key, value in dict(right_case.get("scientific_metrics", {})).items()
            if isinstance(value, (int, float))
        }
        metric_keys = sorted(set(left_metrics.keys()) | set(right_metrics.keys()))
        for metric_name in metric_keys:
            left_value = left_metrics.get(metric_name)
            right_value = right_metrics.get(metric_name)
            if left_value is None or right_value is None:
                scientific_mismatches.append(
                    {
                        "code": "scientific_metric_missing",
                        "case_id": case_id,
                        "metric": metric_name,
                        "left": left_value,
                        "right": right_value,
                    }
                )
                continue
            absolute_delta = abs(float(left_value) - float(right_value))
            if absolute_delta > float(metrics_tolerance):
                scientific_mismatches.append(
                    {
                        "code": "scientific_metric_drift",
                        "case_id": case_id,
                        "metric": metric_name,
                        "left": float(left_value),
                        "right": float(right_value),
                        "absolute_delta": float(absolute_delta),
                        "tolerance": float(metrics_tolerance),
                    }
                )

        for bundle_side, case_payload in (("left", left_case), ("right", right_case)):
            observability = dict(case_payload.get("observability", {}))
            expected_artifacts = [
                str(item)
                for item in list(observability.get("expected_artifacts", []))
                if str(item).strip()
            ]
            artifact_presence = {
                str(key): bool(value)
                for key, value in dict(observability.get("artifact_presence", {})).items()
            }
            schema_checks = {
                str(key): bool(value)
                for key, value in dict(observability.get("schema_checks", {})).items()
            }

            missing_artifacts = [
                artifact for artifact in expected_artifacts if not bool(artifact_presence.get(artifact))
            ]
            if missing_artifacts:
                observability_mismatches.append(
                    {
                        "code": "missing_observability_artifact",
                        "case_id": case_id,
                        "bundle_side": bundle_side,
                        "missing_artifacts": missing_artifacts,
                    }
                )

            invalid_schema_checks = [
                name for name, passed in schema_checks.items() if not bool(passed)
            ]
            if invalid_schema_checks:
                observability_mismatches.append(
                    {
                        "code": "observability_schema_invalid",
                        "case_id": case_id,
                        "bundle_side": bundle_side,
                        "failed_schema_checks": invalid_schema_checks,
                    }
                )

    return {
        "schema_version": "performance-baseline-compare-v1",
        "generated_at_utc": _utc_now(),
        "left_bundle_source": (
            str(Path(left_bundle).resolve())
            if isinstance(left_bundle, (Path, str))
            else "in_memory"
        ),
        "right_bundle_source": (
            str(Path(right_bundle).resolve())
            if isinstance(right_bundle, (Path, str))
            else "in_memory"
        ),
        "scientific_parity": bool(not scientific_mismatches),
        "observability_parity": bool(not observability_mismatches),
        "mismatches": {
            "scientific": scientific_mismatches,
            "observability": observability_mismatches,
        },
        "metrics_tolerance": float(metrics_tolerance),
    }


def build_case_result_from_run_payload(
    *,
    case_id: str,
    elapsed_seconds: float,
    run_payload: Mapping[str, Any],
) -> BenchmarkCaseResult:
    metrics_path = _resolve_path(str(run_payload.get("metrics_path", "")))
    config_path = _resolve_path(str(run_payload.get("config_path", "")))
    fold_splits_path = _resolve_path(str(run_payload.get("fold_splits_path", "")))
    predictions_path = _resolve_path(str(run_payload.get("predictions_path", "")))
    tuning_summary_path = _resolve_path(str(run_payload.get("tuning_summary_path", "")))
    selected_samples_path = _resolve_path(str(run_payload.get("feature_qc_selected_samples_path", "")))
    run_status_path = _resolve_path(str(run_payload.get("run_status_path", "")))

    metrics_payload = _load_json(metrics_path) if metrics_path is not None and metrics_path.exists() else {}
    config_payload = _load_json(config_path) if config_path is not None and config_path.exists() else {}
    tuning_payload = (
        _load_json(tuning_summary_path)
        if tuning_summary_path is not None and tuning_summary_path.exists()
        else {}
    )
    run_status_payload = (
        _load_json(run_status_path)
        if run_status_path is not None and run_status_path.exists()
        else {}
    )

    process_profile_artifacts = run_payload.get("process_profile_artifacts")
    if isinstance(process_profile_artifacts, Mapping):
        process_samples_path = _resolve_path(process_profile_artifacts.get("process_samples_path"))
        process_profile_summary_path = _resolve_path(
            process_profile_artifacts.get("process_profile_summary_path")
        )
    else:
        process_samples_path = _resolve_path(run_payload.get("process_samples_path"))
        process_profile_summary_path = _resolve_path(run_payload.get("process_profile_summary_path"))

    fingerprints = build_benchmark_fingerprint(
        config_payload=config_payload,
        selected_samples_path=selected_samples_path,
        fold_splits_path=fold_splits_path,
        predictions_path=predictions_path,
        metrics_payload=metrics_payload,
        tuning_payload=tuning_payload,
        stage_execution_payload=(
            metrics_payload.get("stage_execution")
            if isinstance(metrics_payload.get("stage_execution"), Mapping)
            else config_payload.get("stage_execution")
        ),
    )

    observability = _profiling_observability(
        run_status_payload=run_status_payload,
        process_samples_path=process_samples_path,
        process_profile_summary_path=process_profile_summary_path,
    )

    artifact_refs = {
        key: str(value)
        for key, value in {
            "run_status_path": run_status_path,
            "metrics_path": metrics_path,
            "config_path": config_path,
            "fold_splits_path": fold_splits_path,
            "predictions_path": predictions_path,
            "tuning_summary_path": tuning_summary_path,
            "feature_qc_selected_samples_path": selected_samples_path,
            "process_samples_path": process_samples_path,
            "process_profile_summary_path": process_profile_summary_path,
        }.items()
        if isinstance(value, Path)
    }

    return BenchmarkCaseResult(
        case_id=str(case_id),
        status="passed",
        elapsed_seconds=float(elapsed_seconds),
        artifact_refs=artifact_refs,
        fingerprints=fingerprints,
        scientific_metrics=_extract_numeric_metrics(metrics_payload),
        observability=observability,
        summary={
            "run_id": str(run_payload.get("run_id")),
            "framework_mode": str(run_payload.get("framework_mode")),
            "status": str(run_status_payload.get("status", "unknown")),
        },
    )


__all__ = [
    "BaselineBundle",
    "BenchmarkCaseResult",
    "BenchmarkCaseSpec",
    "BenchmarkFingerprint",
    "EnvironmentSnapshot",
    "build_benchmark_fingerprint",
    "build_case_result_from_run_payload",
    "compare_baseline_bundles",
    "run_baseline_suite",
    "write_baseline_bundle",
]
