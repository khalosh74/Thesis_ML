from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_SEVERITY_ORDER = {"info": 0, "warning": 1, "error": 2}
_ALLOWED_SEVERITIES = set(_SEVERITY_ORDER.keys())
_ALLOWED_CATEGORIES = {"runtime", "scheduler", "scientific_execution", "data_config"}
_ALLOWED_SOURCES = {"event_rule", "terminal_run_rule", "eta_rule", "timeout_rule"}


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


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


def _safe_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except Exception:
        return None


def _safe_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(str(value))
    except Exception:
        return None


def _normalized_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _severity_rank(value: str) -> int:
    return _SEVERITY_ORDER.get(str(value).strip().lower(), -1)


def _is_lock_phase(phase_name: str | None) -> bool:
    if phase_name is None:
        return False
    normalized = str(phase_name).strip().lower()
    return (
        normalized.startswith("stage 1")
        or normalized.startswith("stage 2")
        or normalized.startswith("stage 3")
        or normalized.startswith("stage 4")
    )


def _contains_timeout(value: Any) -> bool:
    if value is None:
        return False
    text = str(value).strip().lower()
    return "timeout" in text or "watchdog_timeout" in text


def _has_stage_timing_key(stage_timings: dict[str, Any] | None, needle: str) -> bool:
    if not isinstance(stage_timings, dict):
        return False
    key_lower = str(needle).strip().lower()
    for key, value in stage_timings.items():
        key_text = str(key).strip().lower()
        if key_lower in key_text:
            numeric = _safe_float(value)
            if numeric is None:
                continue
            if numeric > 0.0:
                return True
    return False


def build_anomaly_id(
    *,
    campaign_id: str,
    code: str,
    phase_name: str | None,
    experiment_id: str | None,
    run_id: str | None,
    severity: str,
) -> str:
    phase = str(phase_name or "__na__").strip()
    experiment = str(experiment_id or "__na__").strip()
    run = str(run_id or "__na__").strip()
    return (
        f"{str(campaign_id).strip()}::{str(code).strip()}::{phase}::{experiment}::{run}::"
        f"{str(severity).strip().lower()}"
    )


def append_anomaly(path: Path, anomaly: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"{json.dumps(_json_safe(dict(anomaly)), ensure_ascii=True)}\n")


class AnomalyEngine:
    def __init__(
        self,
        campaign_root: Path,
        campaign_id: str,
        high_memory_peak_mb: float = 8192.0,
        lock_phase_blocked_ratio_error: float = 0.10,
    ) -> None:
        self.campaign_root = Path(campaign_root)
        self.campaign_id = str(campaign_id)
        self.high_memory_peak_mb = float(high_memory_peak_mb)
        self.lock_phase_blocked_ratio_error = float(lock_phase_blocked_ratio_error)
        self.anomalies_path = self.campaign_root / "anomalies.jsonl"
        self.report_path = self.campaign_root / "campaign_anomaly_report.json"
        self.anomalies_path.parent.mkdir(parents=True, exist_ok=True)
        self.anomalies_path.touch(exist_ok=True)

        self._dedup_by_id: set[str] = set()
        self._max_severity_by_group: dict[tuple[str, str, str], str] = {}
        self._special_warning_emitted = False
        self._anomalies: list[dict[str, Any]] = []
        self._phase_counts: dict[str, dict[str, int]] = defaultdict(
            lambda: {
                "runs_planned": 0,
                "runs_completed": 0,
                "runs_failed": 0,
                "runs_blocked": 0,
                "runs_dry_run": 0,
            }
        )

    def _emit(
        self,
        *,
        severity: str,
        category: str,
        code: str,
        message: str,
        source: str,
        phase_name: str | None = None,
        experiment_id: str | None = None,
        run_id: str | None = None,
        evidence: dict[str, Any] | None = None,
        recommended_action: str | None = None,
        blocking: bool = False,
        dedup_group: str | None = None,
    ) -> dict[str, Any] | None:
        sev = str(severity).strip().lower()
        cat = str(category).strip().lower()
        src = str(source).strip().lower()
        if sev not in _ALLOWED_SEVERITIES:
            return None
        if cat not in _ALLOWED_CATEGORIES:
            return None
        if src not in _ALLOWED_SOURCES:
            return None

        group_key = (
            str(code).strip(),
            str(dedup_group or ""),
            str(phase_name or "__na__"),
        )
        previous_severity = self._max_severity_by_group.get(group_key)
        if previous_severity is not None and _severity_rank(sev) <= _severity_rank(
            previous_severity
        ):
            return None
        self._max_severity_by_group[group_key] = sev

        anomaly_id = build_anomaly_id(
            campaign_id=self.campaign_id,
            code=str(code),
            phase_name=phase_name,
            experiment_id=experiment_id,
            run_id=run_id,
            severity=sev,
        )
        if anomaly_id in self._dedup_by_id:
            return None
        self._dedup_by_id.add(anomaly_id)

        anomaly = {
            "anomaly_id": anomaly_id,
            "detected_at_utc": _utc_now(),
            "campaign_id": self.campaign_id,
            "phase_name": _normalized_text(phase_name),
            "experiment_id": _normalized_text(experiment_id),
            "run_id": _normalized_text(run_id),
            "severity": sev,
            "category": cat,
            "code": str(code),
            "message": str(message),
            "evidence": _json_safe(dict(evidence or {})),
            "recommended_action": (
                str(recommended_action)
                if recommended_action is not None
                else "Inspect campaign artifacts and rerun affected units after fixes."
            ),
            "blocking": bool(blocking),
            "source": src,
        }
        append_anomaly(self.anomalies_path, anomaly)
        self._anomalies.append(anomaly)
        return anomaly

    def _update_phase_counts_from_event(self, event: dict[str, Any]) -> None:
        phase_name = _normalized_text(event.get("phase_name"))
        if phase_name is None:
            return
        event_name = str(event.get("event_name") or "").strip().lower()
        phase_counts = self._phase_counts[phase_name]
        if event_name == "run_planned":
            phase_counts["runs_planned"] = int(phase_counts["runs_planned"]) + 1
        elif event_name == "run_finished":
            phase_counts["runs_completed"] = int(phase_counts["runs_completed"]) + 1
        elif event_name == "run_failed":
            phase_counts["runs_failed"] = int(phase_counts["runs_failed"]) + 1
        elif event_name == "run_blocked":
            phase_counts["runs_blocked"] = int(phase_counts["runs_blocked"]) + 1
        elif event_name == "run_dry_run":
            phase_counts["runs_dry_run"] = int(phase_counts["runs_dry_run"]) + 1

    def ingest_event(self, event: dict[str, Any]) -> list[dict[str, Any]]:
        emitted: list[dict[str, Any]] = []
        if not isinstance(event, dict):
            return emitted
        self._update_phase_counts_from_event(event)

        event_name = str(event.get("event_name") or "").strip().lower()
        status = str(event.get("status") or "").strip().lower()
        phase_name = _normalized_text(event.get("phase_name"))
        experiment_id = _normalized_text(event.get("experiment_id"))
        run_id = _normalized_text(event.get("run_id"))
        metadata = event.get("metadata")
        event_metadata = dict(metadata) if isinstance(metadata, dict) else {}
        error_text = event_metadata.get("error")

        if event_name == "run_failed" or status == "failed":
            anomaly = self._emit(
                severity="error",
                category="runtime",
                code="RUN_FAILED",
                message="Run finished with failed status.",
                source="event_rule",
                phase_name=phase_name,
                experiment_id=experiment_id,
                run_id=run_id,
                evidence={"status": status, "error": error_text},
                recommended_action="Inspect run_status.json and stderr artifacts for failure root cause.",
                blocking=True,
                dedup_group="run_failed",
            )
            if anomaly is not None:
                emitted.append(anomaly)

        if event_name == "run_timeout" or status == "timeout" or _contains_timeout(error_text):
            anomaly = self._emit(
                severity="error",
                category="runtime",
                code="RUN_TIMEOUT",
                message="Run exceeded watchdog timeout policy.",
                source="timeout_rule",
                phase_name=phase_name,
                experiment_id=experiment_id,
                run_id=run_id,
                evidence={"status": status, "error": error_text},
                recommended_action="Review timeout policy and fit_timing_summary before retrying.",
                blocking=True,
                dedup_group="run_timeout",
            )
            if anomaly is not None:
                emitted.append(anomaly)

        if event_name == "run_blocked" and _is_lock_phase(phase_name):
            phase_counts = self._phase_counts[str(phase_name)]
            planned = max(1, int(phase_counts["runs_planned"]))
            blocked = int(phase_counts["runs_blocked"])
            blocked_ratio = float(blocked) / float(planned)
            severity = (
                "error" if blocked_ratio > float(self.lock_phase_blocked_ratio_error) else "warning"
            )
            anomaly = self._emit(
                severity=severity,
                category="scheduler",
                code="HIGH_BLOCKED_RATIO_LOCK_PHASE",
                message="Blocked run(s) detected in lock phase.",
                source="event_rule",
                phase_name=phase_name,
                experiment_id=experiment_id,
                run_id=run_id,
                evidence={
                    "runs_planned": int(planned),
                    "runs_blocked": int(blocked),
                    "blocked_ratio": float(blocked_ratio),
                    "threshold_error": float(self.lock_phase_blocked_ratio_error),
                    "blocked_reason": event_metadata.get("blocked_reason"),
                },
                recommended_action=(
                    "Review blocked reason and unblock lock-phase runs before accepting lock outcome."
                ),
                blocking=(severity == "error"),
                dedup_group="lock_phase_blocked_ratio",
            )
            if anomaly is not None:
                emitted.append(anomaly)

        if event_name == "run_failed" and _is_lock_phase(phase_name):
            anomaly = self._emit(
                severity="error",
                category="scientific_execution",
                code="LOCK_PHASE_HAS_FAILURES",
                message="Lock phase contains failed run(s).",
                source="event_rule",
                phase_name=phase_name,
                experiment_id=experiment_id,
                run_id=run_id,
                evidence={"error": error_text},
                recommended_action="Resolve lock-phase failures before downstream confirmatory execution.",
                blocking=True,
                dedup_group="lock_phase_failures",
            )
            if anomaly is not None:
                emitted.append(anomaly)

        if event_name == "phase_finished":
            phase_counts = self._phase_counts.get(str(phase_name or ""), {})
            planned = int(phase_counts.get("runs_planned", 0))
            completed = int(phase_counts.get("runs_completed", 0))
            if planned > 0 and completed == 0:
                anomaly = self._emit(
                    severity="error",
                    category="scheduler",
                    code="PHASE_WITH_NO_COMPLETIONS",
                    message="Phase ended with planned work but zero completed runs.",
                    source="event_rule",
                    phase_name=phase_name,
                    experiment_id=experiment_id,
                    run_id=run_id,
                    evidence={
                        "runs_planned": int(planned),
                        "runs_completed": int(completed),
                        "runs_failed": int(phase_counts.get("runs_failed", 0)),
                        "runs_blocked": int(phase_counts.get("runs_blocked", 0)),
                        "runs_dry_run": int(phase_counts.get("runs_dry_run", 0)),
                    },
                    recommended_action="Inspect phase skip/block/failure causes before proceeding.",
                    blocking=True,
                    dedup_group="phase_no_completions",
                )
                if anomaly is not None:
                    emitted.append(anomaly)

        if (
            not self._special_warning_emitted
            and event_name == "run_blocked"
            and experiment_id in {"E23"}
        ):
            self._special_warning_emitted = True
            anomaly = self._emit(
                severity="warning",
                category="data_config",
                code="UNSUPPORTED_SPECIAL_EXPERIMENT_BLOCKED",
                message="Special experiment blocked due to unsupported/missing data configuration.",
                source="event_rule",
                phase_name=phase_name,
                experiment_id=experiment_id,
                run_id=run_id,
                evidence={"blocked_reason": event_metadata.get("blocked_reason")},
                recommended_action=(
                    "Validate registry/search-space and dataset metadata for special experiment."
                ),
                blocking=False,
                dedup_group="special_experiment_blocked",
            )
            if anomaly is not None:
                emitted.append(anomaly)

        return emitted

    def inspect_terminal_run(self, metadata: dict[str, Any]) -> list[dict[str, Any]]:
        emitted: list[dict[str, Any]] = []
        if not isinstance(metadata, dict):
            return emitted
        terminal_status = str(metadata.get("status") or "").strip().lower()
        if terminal_status in {"blocked", "dry_run"}:
            return emitted
        phase_name = _normalized_text(metadata.get("phase_name"))
        experiment_id = _normalized_text(metadata.get("experiment_id"))
        run_id = _normalized_text(metadata.get("run_id"))

        actual_runtime_seconds = _safe_float(metadata.get("actual_runtime_seconds"))
        eta_p80_seconds = _safe_float(metadata.get("eta_p80_seconds"))
        projected_runtime_seconds = _safe_float(metadata.get("projected_runtime_seconds"))
        if (
            actual_runtime_seconds is not None
            and actual_runtime_seconds > 0.0
            and (
                (eta_p80_seconds is not None and eta_p80_seconds > 0.0)
                or (projected_runtime_seconds is not None and projected_runtime_seconds > 0.0)
            )
        ):
            threshold = (
                float(eta_p80_seconds)
                if eta_p80_seconds is not None and eta_p80_seconds > 0.0
                else float(projected_runtime_seconds) * 2.0
            )
            if actual_runtime_seconds > threshold:
                anomaly = self._emit(
                    severity="warning",
                    category="runtime",
                    code="RUN_EXCEEDS_ETA_P80",
                    message="Run exceeded expected high-percentile runtime bound.",
                    source="eta_rule",
                    phase_name=phase_name,
                    experiment_id=experiment_id,
                    run_id=run_id,
                    evidence={
                        "actual_runtime_seconds": float(actual_runtime_seconds),
                        "eta_p80_seconds": (
                            float(eta_p80_seconds) if eta_p80_seconds is not None else None
                        ),
                        "projected_runtime_seconds": (
                            float(projected_runtime_seconds)
                            if projected_runtime_seconds is not None
                            else None
                        ),
                        "threshold_seconds": float(threshold),
                    },
                    recommended_action="Inspect stage timings and consider compute policy adjustment.",
                    blocking=False,
                    dedup_group="run_exceeds_eta",
                )
                if anomaly is not None:
                    emitted.append(anomaly)

        process_profile_summary = metadata.get("process_profile_summary")
        if isinstance(process_profile_summary, dict):
            peak_rss_mb = _safe_float(process_profile_summary.get("peak_rss_mb"))
            if peak_rss_mb is not None and peak_rss_mb > float(self.high_memory_peak_mb):
                anomaly = self._emit(
                    severity="warning",
                    category="runtime",
                    code="HIGH_MEMORY_PEAK",
                    message="Run exceeded high memory peak threshold.",
                    source="terminal_run_rule",
                    phase_name=phase_name,
                    experiment_id=experiment_id,
                    run_id=run_id,
                    evidence={
                        "peak_rss_mb": float(peak_rss_mb),
                        "threshold_mb": float(self.high_memory_peak_mb),
                    },
                    recommended_action="Review memory-heavy stages and reduce per-process workload.",
                    blocking=False,
                    dedup_group="high_memory_peak",
                )
                if anomaly is not None:
                    emitted.append(anomaly)

        stage_timings = metadata.get("stage_timings_seconds")
        tuning_enabled = bool(metadata.get("tuning_enabled"))
        if tuning_enabled and not _has_stage_timing_key(stage_timings, "tuning"):
            anomaly = self._emit(
                severity="warning",
                category="scientific_execution",
                code="TUNING_EXPECTED_BUT_MISSING",
                message="Tuning was expected but no tuning timing evidence was found.",
                source="terminal_run_rule",
                phase_name=phase_name,
                experiment_id=experiment_id,
                run_id=run_id,
                evidence={
                    "tuning_enabled": bool(tuning_enabled),
                    "stage_timing_keys": (
                        list(stage_timings.keys()) if isinstance(stage_timings, dict) else []
                    ),
                },
                recommended_action="Verify grouped nested tuning execution path and timing stamps.",
                blocking=False,
                dedup_group="tuning_missing",
            )
            if anomaly is not None:
                emitted.append(anomaly)

        n_permutations = _safe_int(metadata.get("n_permutations")) or 0
        if n_permutations > 0 and not _has_stage_timing_key(stage_timings, "permutation"):
            anomaly = self._emit(
                severity="error",
                category="scientific_execution",
                code="PERMUTATION_EXPECTED_BUT_MISSING",
                message="Permutation control expected but no permutation timing evidence was found.",
                source="terminal_run_rule",
                phase_name=phase_name,
                experiment_id=experiment_id,
                run_id=run_id,
                evidence={
                    "n_permutations": int(n_permutations),
                    "stage_timing_keys": (
                        list(stage_timings.keys()) if isinstance(stage_timings, dict) else []
                    ),
                },
                recommended_action="Inspect permutation execution path and control artifact generation.",
                blocking=True,
                dedup_group="permutation_missing",
            )
            if anomaly is not None:
                emitted.append(anomaly)

        feature_space = _normalized_text(metadata.get("feature_space"))
        roi_spec_path = _normalized_text(metadata.get("roi_spec_path"))
        if feature_space == "roi_mean_predefined":
            missing_or_proxy = False
            proxy_reason = None
            if roi_spec_path is None:
                missing_or_proxy = True
                proxy_reason = "missing roi_spec_path"
            else:
                roi_path = Path(roi_spec_path)
                if not roi_path.exists():
                    missing_or_proxy = True
                    proxy_reason = "roi_spec_path does not exist"
                elif "proxy" in roi_path.name.lower() or "lightweight" in roi_path.name.lower():
                    missing_or_proxy = True
                    proxy_reason = "roi_spec_path appears to be a proxy/lightweight artifact"
            if missing_or_proxy:
                anomaly = self._emit(
                    severity="warning",
                    category="data_config",
                    code="ROI_FEATURE_SPACE_PROXY_WARNING",
                    message="ROI feature space appears to use proxy/missing ROI assets.",
                    source="terminal_run_rule",
                    phase_name=phase_name,
                    experiment_id=experiment_id,
                    run_id=run_id,
                    evidence={"roi_spec_path": roi_spec_path, "reason": proxy_reason},
                    recommended_action="Validate ROI spec and required ROI assets before lock decisions.",
                    blocking=False,
                    dedup_group="roi_proxy_warning",
                )
                if anomaly is not None:
                    emitted.append(anomaly)

        return emitted

    def current_anomaly_payload(self) -> dict[str, Any]:
        severity_counts = Counter(str(item.get("severity")) for item in self._anomalies)
        code_counts = Counter(str(item.get("code")) for item in self._anomalies)
        category_counts = Counter(str(item.get("category")) for item in self._anomalies)
        return {
            "anomalies": list(self._anomalies[-20:]),
            "anomaly_counts": {
                "total": int(len(self._anomalies)),
                "by_severity": {
                    str(key): int(value) for key, value in sorted(severity_counts.items())
                },
                "by_code": {str(key): int(value) for key, value in sorted(code_counts.items())},
                "by_category": {
                    str(key): int(value) for key, value in sorted(category_counts.items())
                },
            },
            "latest_anomaly": (dict(self._anomalies[-1]) if self._anomalies else None),
        }

    def finalize(self) -> dict[str, Any]:
        payload = self.current_anomaly_payload()
        by_phase: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0})
        for anomaly in self._anomalies:
            phase_name = str(anomaly.get("phase_name") or "__na__")
            by_phase_entry = by_phase[phase_name]
            by_phase_entry["total"] = int(by_phase_entry.get("total", 0)) + 1
            severity = str(anomaly.get("severity") or "info")
            by_phase_entry[severity] = int(by_phase_entry.get(severity, 0)) + 1

        report = {
            "campaign_id": self.campaign_id,
            "generated_at_utc": _utc_now(),
            "anomalies_path": str(self.anomalies_path.resolve()),
            "anomaly_counts": payload.get("anomaly_counts", {}),
            "latest_anomaly": payload.get("latest_anomaly"),
            "anomalies_by_phase": {str(key): value for key, value in sorted(by_phase.items())},
            "codes": sorted(
                {str(anomaly.get("code")) for anomaly in self._anomalies if anomaly.get("code")}
            ),
        }
        self.report_path.write_text(
            f"{json.dumps(_json_safe(report), indent=2)}\n", encoding="utf-8"
        )
        return report


__all__ = ["AnomalyEngine", "append_anomaly", "build_anomaly_id"]
