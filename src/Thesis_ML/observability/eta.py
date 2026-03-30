from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, median
from typing import Any


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _coerce_positive_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        resolved = float(value)
    except Exception:
        return None
    if not math.isfinite(resolved) or resolved <= 0.0:
        return None
    return float(resolved)


def _normalized_text(value: Any) -> str:
    if value is None:
        return "__na__"
    text = str(value).strip().lower()
    return text or "__na__"


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value in (None, ""):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _safe_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(str(value))
    except Exception:
        return None


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    try:
        return str(value)
    except Exception:
        return repr(value)


def load_runtime_history(path: Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except Exception:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
    except Exception:
        return []
    return rows


def append_runtime_history(path: Path, record: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"{json.dumps(_json_safe(dict(record)), ensure_ascii=True)}\n")


def load_runtime_profile_summary(path: Path) -> dict[str, Any] | None:
    path = Path(path)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    return {
        "cohort_estimates": payload.get("cohort_estimates"),
        "phase_estimates": payload.get("phase_estimates"),
        "model_estimates": payload.get("model_estimates"),
        "estimated_total_wall_time_seconds": payload.get("estimated_total_wall_time_seconds"),
    }


def build_runtime_keys(metadata: dict[str, Any]) -> dict[str, str]:
    exact_fields = [
        "experiment_id",
        "framework_mode",
        "model_cost_tier",
        "feature_space",
        "preprocessing_strategy",
        "dimensionality_strategy",
        "tuning_enabled",
        "cv_mode",
        "n_permutations",
        "subject",
        "train_subject",
        "test_subject",
        "task",
        "modality",
    ]
    exact_parts = [f"{field}={_normalized_text(metadata.get(field))}" for field in exact_fields]

    backoff_1_fields = ["phase_name", "framework_mode", "model_cost_tier", "cv_mode"]
    backoff_1_parts = [
        f"{field}={_normalized_text(metadata.get(field))}" for field in backoff_1_fields
    ]

    backoff_2_fields = ["framework_mode", "model_cost_tier"]
    backoff_2_parts = [
        f"{field}={_normalized_text(metadata.get(field))}" for field in backoff_2_fields
    ]

    return {
        "exact": "exact|" + "|".join(exact_parts),
        "backoff_1": "backoff_1|" + "|".join(backoff_1_parts),
        "backoff_2": "backoff_2|" + "|".join(backoff_2_parts),
    }


_SOURCE_MULTIPLIERS: dict[str, float] = {
    "live_exact": 1.25,
    "history_exact": 1.40,
    "history_backoff": 1.60,
    "runtime_profile_cohort": 1.75,
    "runtime_profile_phase": 1.75,
    "runtime_profile_model": 1.75,
    "projected_runtime": 2.00,
}

_SOURCE_CONFIDENCE: dict[str, str] = {
    "live_exact": "high",
    "history_exact": "medium",
    "history_backoff": "medium",
    "runtime_profile_cohort": "low",
    "runtime_profile_phase": "low",
    "runtime_profile_model": "low",
    "projected_runtime": "low",
}


class EtaEstimator:
    def __init__(
        self,
        campaign_root: Path,
        campaign_id: str,
        history_path: Path,
        runtime_profile_summary_path: Path | None = None,
    ) -> None:
        self.campaign_root = Path(campaign_root)
        self.campaign_id = str(campaign_id)
        self.history_path = Path(history_path)
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        self.history_path.touch(exist_ok=True)
        self.runtime_profile_summary_path = (
            Path(runtime_profile_summary_path) if runtime_profile_summary_path is not None else None
        )
        self.eta_state_path = self.campaign_root / "eta_state.json"
        self.calibration_path = self.campaign_root / "campaign_eta_calibration.json"

        self._current_phase: str | None = None
        self._planned_runs: dict[str, dict[str, Any]] = {}
        self._started_run_ids: set[str] = set()
        self._terminal_run_ids: set[str] = set()
        self._history_written_run_ids: set[str] = set()
        self._measured_rows: list[dict[str, Any]] = []

        self._history_rows = load_runtime_history(self.history_path)
        self._runtime_profile = (
            load_runtime_profile_summary(self.runtime_profile_summary_path)
            if self.runtime_profile_summary_path is not None
            else None
        )
        self._campaign_dry_run = False

        self._live_exact_samples: dict[str, list[float]] = defaultdict(list)
        self._history_exact_samples: dict[str, list[float]] = defaultdict(list)
        self._history_backoff_1_samples: dict[str, list[float]] = defaultdict(list)
        self._history_backoff_2_samples: dict[str, list[float]] = defaultdict(list)
        self._runtime_profile_cohort_samples: dict[str, list[float]] = defaultdict(list)
        self._runtime_profile_phase_samples: dict[str, float] = {}
        self._runtime_profile_model_samples: dict[str, float] = {}

        self._build_history_indexes()
        self._build_runtime_profile_indexes()

    def _build_history_indexes(self) -> None:
        for row in self._history_rows:
            if not isinstance(row, dict):
                continue
            actual = _coerce_positive_float(row.get("actual_runtime_seconds"))
            if actual is None:
                continue
            exact = str(row.get("runtime_key_exact") or "").strip()
            if exact:
                self._history_exact_samples[exact].append(float(actual))
            backoff_1 = str(row.get("runtime_key_backoff_1") or "").strip()
            if backoff_1:
                self._history_backoff_1_samples[backoff_1].append(float(actual))
            backoff_2 = str(row.get("runtime_key_backoff_2") or "").strip()
            if backoff_2:
                self._history_backoff_2_samples[backoff_2].append(float(actual))

    def _build_runtime_profile_indexes(self) -> None:
        if not isinstance(self._runtime_profile, dict):
            return
        cohort_estimates = self._runtime_profile.get("cohort_estimates")
        if isinstance(cohort_estimates, list):
            for item in cohort_estimates:
                if not isinstance(item, dict):
                    continue
                estimated_seconds = _coerce_positive_float(
                    item.get("estimated_seconds_per_full_run")
                )
                if estimated_seconds is None:
                    total_seconds = _coerce_positive_float(item.get("estimated_total_seconds"))
                    planned_runs = _safe_int(item.get("n_planned_runs")) or 0
                    if total_seconds is not None and planned_runs > 0:
                        estimated_seconds = float(total_seconds) / float(planned_runs)
                if estimated_seconds is None:
                    continue
                cohort_descriptor = item.get("cohort")
                metadata: dict[str, Any] = {}
                if isinstance(cohort_descriptor, dict):
                    metadata = {
                        "phase_name": cohort_descriptor.get("phase"),
                        "framework_mode": cohort_descriptor.get("framework_mode_source"),
                        "model_cost_tier": cohort_descriptor.get("model_cost_tier"),
                        "feature_space": cohort_descriptor.get("feature_space"),
                        "preprocessing_strategy": cohort_descriptor.get("preprocessing_strategy"),
                        "dimensionality_strategy": cohort_descriptor.get("dimensionality_strategy"),
                        "tuning_enabled": cohort_descriptor.get("tuning_enabled"),
                        "cv_mode": cohort_descriptor.get("cv_mode"),
                        "n_permutations": cohort_descriptor.get("controls_n_permutations"),
                        "task": cohort_descriptor.get("filter_task"),
                        "modality": cohort_descriptor.get("filter_modality"),
                    }
                metadata["experiment_id"] = item.get("experiment_id")
                keys = build_runtime_keys(metadata)
                self._runtime_profile_cohort_samples[keys["exact"]].append(float(estimated_seconds))

        phase_estimates = self._runtime_profile.get("phase_estimates")
        if isinstance(phase_estimates, dict):
            for phase_name, phase_payload in phase_estimates.items():
                if not isinstance(phase_payload, dict):
                    continue
                total_seconds = _coerce_positive_float(phase_payload.get("estimated_total_seconds"))
                if total_seconds is None:
                    continue
                planned_runs = _safe_int(phase_payload.get("n_planned_runs")) or 0
                per_run_seconds = (
                    float(total_seconds) / float(planned_runs)
                    if planned_runs > 0
                    else float(total_seconds)
                )
                self._runtime_profile_phase_samples[str(phase_name).strip().lower()] = float(
                    per_run_seconds
                )

        model_estimates = self._runtime_profile.get("model_estimates")
        if isinstance(model_estimates, dict):
            for model_name, model_payload in model_estimates.items():
                if not isinstance(model_payload, dict):
                    continue
                seconds = _coerce_positive_float(model_payload.get("estimated_total_seconds"))
                if seconds is None:
                    continue
                self._runtime_profile_model_samples[str(model_name).strip().lower()] = float(
                    seconds
                )

    def _resolve_run_id(self, metadata: dict[str, Any]) -> str:
        run_id_value = metadata.get("run_id")
        if run_id_value is not None and str(run_id_value).strip():
            return str(run_id_value)
        keys = build_runtime_keys(metadata)
        return f"unknown::{keys['exact']}"

    def _normalized_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]:
        resolved = dict(metadata)
        resolved["campaign_id"] = str(resolved.get("campaign_id") or self.campaign_id)
        resolved["phase_name"] = (
            str(resolved.get("phase_name")) if resolved.get("phase_name") is not None else None
        )
        resolved["experiment_id"] = (
            str(resolved.get("experiment_id"))
            if resolved.get("experiment_id") is not None
            else None
        )
        resolved["run_id"] = self._resolve_run_id(resolved)
        resolved["framework_mode"] = str(resolved.get("framework_mode") or "exploratory")
        resolved["feature_space"] = str(resolved.get("feature_space") or "whole_brain_masked")
        resolved["preprocessing_strategy"] = (
            str(resolved.get("preprocessing_strategy"))
            if resolved.get("preprocessing_strategy") not in (None, "")
            else "none"
        )
        resolved["dimensionality_strategy"] = (
            str(resolved.get("dimensionality_strategy"))
            if resolved.get("dimensionality_strategy") not in (None, "")
            else "none"
        )
        resolved["tuning_enabled"] = bool(_coerce_bool(resolved.get("tuning_enabled")))
        resolved["cv_mode"] = (
            str(resolved.get("cv_mode")) if resolved.get("cv_mode") not in (None, "") else None
        )
        resolved["n_permutations"] = _safe_int(resolved.get("n_permutations")) or 0
        resolved["projected_runtime_seconds"] = _coerce_positive_float(
            resolved.get("projected_runtime_seconds")
        )
        runtime_keys = build_runtime_keys(resolved)
        resolved["runtime_key_exact"] = runtime_keys["exact"]
        resolved["runtime_key_backoff_1"] = runtime_keys["backoff_1"]
        resolved["runtime_key_backoff_2"] = runtime_keys["backoff_2"]
        return resolved

    def register_planned_run(self, metadata: dict[str, Any]) -> None:
        normalized = self._normalized_metadata(metadata)
        run_id = str(normalized["run_id"])
        self._planned_runs[run_id] = normalized

    def mark_run_started(self, metadata: dict[str, Any]) -> None:
        normalized = self._normalized_metadata(metadata)
        run_id = str(normalized["run_id"])
        if run_id not in self._planned_runs:
            self._planned_runs[run_id] = normalized
        self._started_run_ids.add(run_id)

    def _estimate_from_samples(self, source: str, samples: list[float]) -> dict[str, Any] | None:
        valid_samples = [float(value) for value in samples if _coerce_positive_float(value)]
        if not valid_samples:
            return None
        p50 = float(median(valid_samples))
        if len(valid_samples) >= 3:
            ordered = sorted(valid_samples)
            index = max(0, int(math.ceil(0.8 * len(ordered))) - 1)
            p80 = float(ordered[index])
        else:
            p80 = float(p50 * _SOURCE_MULTIPLIERS[source])
        return {
            "p50_seconds": p50,
            "p80_seconds": p80,
            "source": source,
            "confidence": _SOURCE_CONFIDENCE[source],
            "sample_count": int(len(valid_samples)),
        }

    def _estimate_runtime(self, metadata: dict[str, Any]) -> dict[str, Any]:
        normalized = self._normalized_metadata(metadata)
        exact_key = str(normalized["runtime_key_exact"])
        backoff_1_key = str(normalized["runtime_key_backoff_1"])
        backoff_2_key = str(normalized["runtime_key_backoff_2"])

        live_exact = self._estimate_from_samples(
            "live_exact", self._live_exact_samples.get(exact_key, [])
        )
        if live_exact is not None:
            return live_exact

        history_exact = self._estimate_from_samples(
            "history_exact", self._history_exact_samples.get(exact_key, [])
        )
        if history_exact is not None:
            return history_exact

        history_backoff_samples = list(self._history_backoff_1_samples.get(backoff_1_key, []))
        history_backoff_samples.extend(self._history_backoff_2_samples.get(backoff_2_key, []))
        history_backoff = self._estimate_from_samples("history_backoff", history_backoff_samples)
        if history_backoff is not None:
            return history_backoff

        runtime_profile_cohort = self._estimate_from_samples(
            "runtime_profile_cohort", self._runtime_profile_cohort_samples.get(exact_key, [])
        )
        if runtime_profile_cohort is not None:
            return runtime_profile_cohort

        phase_name = str(normalized.get("phase_name") or "").strip().lower()
        phase_seconds = _coerce_positive_float(self._runtime_profile_phase_samples.get(phase_name))
        if phase_seconds is not None:
            p50 = float(phase_seconds)
            return {
                "p50_seconds": p50,
                "p80_seconds": float(p50 * _SOURCE_MULTIPLIERS["runtime_profile_phase"]),
                "source": "runtime_profile_phase",
                "confidence": _SOURCE_CONFIDENCE["runtime_profile_phase"],
                "sample_count": 1,
            }

        model_name = str(normalized.get("model") or "").strip().lower()
        model_seconds = _coerce_positive_float(self._runtime_profile_model_samples.get(model_name))
        if model_seconds is not None:
            p50 = float(model_seconds)
            return {
                "p50_seconds": p50,
                "p80_seconds": float(p50 * _SOURCE_MULTIPLIERS["runtime_profile_model"]),
                "source": "runtime_profile_model",
                "confidence": _SOURCE_CONFIDENCE["runtime_profile_model"],
                "sample_count": 1,
            }

        projected_runtime = _coerce_positive_float(normalized.get("projected_runtime_seconds"))
        if projected_runtime is not None:
            p50 = float(projected_runtime)
        else:
            p50 = 1.0
        return {
            "p50_seconds": p50,
            "p80_seconds": float(p50 * _SOURCE_MULTIPLIERS["projected_runtime"]),
            "source": "projected_runtime",
            "confidence": _SOURCE_CONFIDENCE["projected_runtime"],
            "sample_count": 0,
        }

    def mark_run_finished(self, metadata: dict[str, Any]) -> None:
        normalized = self._normalized_metadata(metadata)
        run_id = str(normalized["run_id"])
        planned = dict(self._planned_runs.get(run_id, {}))
        planned.update(normalized)
        self._planned_runs[run_id] = planned
        self._terminal_run_ids.add(run_id)
        self._started_run_ids.discard(run_id)

        actual_runtime = _coerce_positive_float(planned.get("actual_runtime_seconds"))
        if actual_runtime is None:
            return

        estimate = self._estimate_runtime(planned)
        source = str(estimate["source"])
        estimated_p50 = float(estimate["p50_seconds"])
        self._live_exact_samples[str(planned["runtime_key_exact"])].append(float(actual_runtime))

        measured_row = {
            "run_id": run_id,
            "phase_name": planned.get("phase_name"),
            "eta_source": source,
            "estimated_runtime_seconds": float(estimated_p50),
            "actual_runtime_seconds": float(actual_runtime),
        }
        self._measured_rows.append(measured_row)

        if run_id not in self._history_written_run_ids:
            history_record = {
                "completed_at_utc": _utc_now(),
                "campaign_id": str(planned.get("campaign_id") or self.campaign_id),
                "phase_name": planned.get("phase_name"),
                "experiment_id": planned.get("experiment_id"),
                "run_id": run_id,
                "runtime_key_exact": str(planned["runtime_key_exact"]),
                "runtime_key_backoff_1": str(planned["runtime_key_backoff_1"]),
                "runtime_key_backoff_2": str(planned["runtime_key_backoff_2"]),
                "projected_runtime_seconds": planned.get("projected_runtime_seconds"),
                "actual_runtime_seconds": float(actual_runtime),
                "framework_mode": planned.get("framework_mode"),
                "model_cost_tier": planned.get("model_cost_tier"),
                "feature_space": planned.get("feature_space"),
                "preprocessing_strategy": planned.get("preprocessing_strategy"),
                "dimensionality_strategy": planned.get("dimensionality_strategy"),
                "tuning_enabled": bool(planned.get("tuning_enabled")),
                "cv_mode": planned.get("cv_mode"),
                "n_permutations": int(planned.get("n_permutations") or 0),
            }
            append_runtime_history(self.history_path, history_record)
            self._history_written_run_ids.add(run_id)

    def mark_run_terminal_nonmeasured(self, metadata: dict[str, Any]) -> None:
        normalized = self._normalized_metadata(metadata)
        run_id = str(normalized["run_id"])
        if run_id not in self._planned_runs:
            self._planned_runs[run_id] = normalized
        self._terminal_run_ids.add(run_id)
        self._started_run_ids.discard(run_id)

    def ingest_event(self, event: dict[str, Any]) -> dict[str, Any] | None:
        if not isinstance(event, dict):
            return None
        event_name = str(event.get("event_name") or "").strip().lower()
        if not event_name:
            return None

        metadata = event.get("metadata")
        event_metadata = dict(metadata) if isinstance(metadata, dict) else {}
        event_metadata.setdefault("campaign_id", event.get("campaign_id") or self.campaign_id)
        event_metadata.setdefault("phase_name", event.get("phase_name"))
        event_metadata.setdefault("experiment_id", event.get("experiment_id"))
        event_metadata.setdefault("variant_id", event.get("variant_id"))
        event_metadata.setdefault("run_id", event.get("run_id"))
        if event_name == "campaign_started":
            self._campaign_dry_run = bool(event_metadata.get("dry_run"))
        elif bool(event_metadata.get("dry_run")):
            self._campaign_dry_run = True

        if event_name == "phase_started":
            self._current_phase = (
                str(event.get("phase_name")) if event.get("phase_name") is not None else None
            )
        elif event_name == "phase_finished":
            self._current_phase = (
                str(event.get("phase_name")) if event.get("phase_name") is not None else None
            )

        if event_name == "run_planned":
            if bool(event_metadata.get("supported", True)):
                self.register_planned_run(event_metadata)
        elif event_name == "run_started":
            self.mark_run_started(event_metadata)
        elif event_name in {"run_finished", "run_failed"}:
            if _coerce_positive_float(event_metadata.get("actual_runtime_seconds")) is not None:
                self.mark_run_finished(event_metadata)
            else:
                self.mark_run_terminal_nonmeasured(event_metadata)
        elif event_name in {"run_blocked", "run_dry_run"}:
            self.mark_run_terminal_nonmeasured(event_metadata)

        payload = self.current_eta_payload()
        self.write_eta_state()
        return payload

    def current_eta_payload(self) -> dict[str, Any]:
        remaining_runs = [
            metadata
            for run_id, metadata in self._planned_runs.items()
            if run_id not in self._terminal_run_ids
        ]

        remaining_estimates: list[dict[str, Any]] = []
        for metadata in remaining_runs:
            estimate = self._estimate_runtime(metadata)
            remaining_estimates.append({**metadata, **estimate})

        campaign_eta_p50 = float(sum(item["p50_seconds"] for item in remaining_estimates))
        campaign_eta_p80 = float(sum(item["p80_seconds"] for item in remaining_estimates))
        campaign_source_counts = Counter(str(item["source"]) for item in remaining_estimates)
        campaign_source = (
            str(campaign_source_counts.most_common(1)[0][0]) if campaign_source_counts else None
        )
        campaign_confidence = (
            _SOURCE_CONFIDENCE.get(campaign_source, "low") if campaign_source is not None else None
        )

        phase_name = self._current_phase
        phase_remaining = [
            item
            for item in remaining_estimates
            if phase_name is not None and str(item.get("phase_name")) == str(phase_name)
        ]
        phase_eta_p50 = float(sum(item["p50_seconds"] for item in phase_remaining))
        phase_eta_p80 = float(sum(item["p80_seconds"] for item in phase_remaining))
        phase_source_counts = Counter(str(item["source"]) for item in phase_remaining)
        phase_source = (
            str(phase_source_counts.most_common(1)[0][0]) if phase_source_counts else None
        )
        phase_confidence = (
            _SOURCE_CONFIDENCE.get(phase_source, "low") if phase_source is not None else None
        )

        buckets: dict[str, dict[str, Any]] = {}
        for item in remaining_estimates:
            key = str(item.get("runtime_key_backoff_1") or item.get("runtime_key_exact"))
            if key not in buckets:
                buckets[key] = {
                    "bucket": key,
                    "run_count": 0,
                    "eta_p50_seconds": 0.0,
                    "eta_p80_seconds": 0.0,
                }
            buckets[key]["run_count"] = int(buckets[key]["run_count"]) + 1
            buckets[key]["eta_p50_seconds"] = float(buckets[key]["eta_p50_seconds"]) + float(
                item["p50_seconds"]
            )
            buckets[key]["eta_p80_seconds"] = float(buckets[key]["eta_p80_seconds"]) + float(
                item["p80_seconds"]
            )

        top_buckets = sorted(
            buckets.values(),
            key=lambda row: float(row["eta_p50_seconds"]),
            reverse=True,
        )[:5]

        if bool(self._campaign_dry_run):
            payload = {
                "campaign_id": self.campaign_id,
                "counts": {
                    "runs_planned": int(len(self._planned_runs)),
                    "runs_completed": int(len(self._terminal_run_ids)),
                    "runs_remaining": int(len(remaining_runs)),
                },
                "current_phase": phase_name,
                "campaign_eta": {
                    "eta_p50_seconds": None,
                    "eta_p80_seconds": None,
                    "eta_confidence": "low",
                    "eta_source": "planning_only_dry_run",
                },
                "phase_eta": {
                    "eta_p50_seconds": None,
                    "eta_p80_seconds": None,
                    "eta_confidence": "low",
                    "eta_source": "planning_only_dry_run",
                },
                "eta_p50_seconds": None,
                "eta_p80_seconds": None,
                "eta_confidence": "low",
                "eta_source": "planning_only_dry_run",
                "dry_run_planning_only": True,
                "source_counts": {},
                "top_remaining_runtime_buckets": top_buckets,
            }
            return payload

        payload = {
            "campaign_id": self.campaign_id,
            "counts": {
                "runs_planned": int(len(self._planned_runs)),
                "runs_completed": int(len(self._terminal_run_ids)),
                "runs_remaining": int(len(remaining_runs)),
            },
            "current_phase": phase_name,
            "campaign_eta": {
                "eta_p50_seconds": campaign_eta_p50,
                "eta_p80_seconds": campaign_eta_p80,
                "eta_confidence": campaign_confidence,
                "eta_source": campaign_source,
            },
            "phase_eta": {
                "eta_p50_seconds": phase_eta_p50,
                "eta_p80_seconds": phase_eta_p80,
                "eta_confidence": phase_confidence,
                "eta_source": phase_source,
            },
            "eta_p50_seconds": campaign_eta_p50,
            "eta_p80_seconds": campaign_eta_p80,
            "eta_confidence": campaign_confidence,
            "eta_source": campaign_source,
            "source_counts": {
                str(source): int(count) for source, count in sorted(campaign_source_counts.items())
            },
            "top_remaining_runtime_buckets": top_buckets,
        }
        return payload

    def write_eta_state(self) -> None:
        payload = self.current_eta_payload()
        self.eta_state_path.parent.mkdir(parents=True, exist_ok=True)
        self.eta_state_path.write_text(
            f"{json.dumps(_json_safe(payload), indent=2)}\n", encoding="utf-8"
        )

    def finalize(self) -> dict[str, Any]:
        rows = list(self._measured_rows)
        estimated_values = [float(row["estimated_runtime_seconds"]) for row in rows]
        actual_values = [float(row["actual_runtime_seconds"]) for row in rows]
        abs_errors = [
            abs(actual - estimated)
            for actual, estimated in zip(actual_values, estimated_values, strict=False)
        ]

        total_estimated = float(sum(estimated_values))
        total_actual = float(sum(actual_values))
        absolute_error = abs(total_actual - total_estimated)
        mean_absolute_error = float(mean(abs_errors)) if abs_errors else 0.0
        median_absolute_error = float(median(abs_errors)) if abs_errors else 0.0

        mape: float | None = None
        if rows and all(actual > 0.0 for actual in actual_values):
            mape = float(
                mean(
                    abs(actual - estimated) / actual
                    for actual, estimated in zip(actual_values, estimated_values, strict=False)
                )
            )

        source_counts = Counter(str(row["eta_source"]) for row in rows)
        phase_errors: dict[str, dict[str, Any]] = {}
        for row in rows:
            phase_name = str(row.get("phase_name") or "__na__")
            entry = phase_errors.setdefault(
                phase_name,
                {
                    "phase_name": phase_name,
                    "count": 0,
                    "estimated_seconds": 0.0,
                    "actual_seconds": 0.0,
                    "absolute_error_seconds": 0.0,
                },
            )
            estimated = float(row["estimated_runtime_seconds"])
            actual = float(row["actual_runtime_seconds"])
            entry["count"] = int(entry["count"]) + 1
            entry["estimated_seconds"] = float(entry["estimated_seconds"]) + estimated
            entry["actual_seconds"] = float(entry["actual_seconds"]) + actual
            entry["absolute_error_seconds"] = float(entry["absolute_error_seconds"]) + abs(
                actual - estimated
            )

        for phase_payload in phase_errors.values():
            count = int(phase_payload["count"])
            phase_payload["mean_absolute_error_seconds"] = (
                float(phase_payload["absolute_error_seconds"]) / float(count) if count > 0 else 0.0
            )

        calibration_payload = {
            "campaign_id": self.campaign_id,
            "generated_at_utc": _utc_now(),
            "total_estimated_seconds_for_measured_runs": total_estimated,
            "total_actual_seconds_for_measured_runs": total_actual,
            "absolute_error_seconds": absolute_error,
            "mean_absolute_error_seconds": mean_absolute_error,
            "median_absolute_error_seconds": median_absolute_error,
            "mape": mape,
            "counts_by_eta_source": {
                str(source): int(count) for source, count in sorted(source_counts.items())
            },
            "error_by_phase": {
                key: value for key, value in sorted(phase_errors.items(), key=lambda item: item[0])
            },
            "history_path_used": str(self.history_path.resolve()),
            "runtime_profile_summary_path_used": (
                str(self.runtime_profile_summary_path.resolve())
                if self.runtime_profile_summary_path is not None
                else None
            ),
        }
        self.calibration_path.parent.mkdir(parents=True, exist_ok=True)
        self.calibration_path.write_text(
            f"{json.dumps(_json_safe(calibration_payload), indent=2)}\n",
            encoding="utf-8",
        )
        return calibration_payload


__all__ = [
    "EtaEstimator",
    "append_runtime_history",
    "build_runtime_keys",
    "load_runtime_history",
    "load_runtime_profile_summary",
]
