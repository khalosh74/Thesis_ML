from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from typing import Any

from Thesis_ML.experiments.progress import ProgressCallback, emit_progress
from Thesis_ML.experiments.stage_execution import StageKey

_STAGE_EVENTS_FILENAME = "stage_events.jsonl"
_STAGE_OBSERVED_EVIDENCE_FILENAME = "stage_observed_evidence.json"


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


def _stage_key_value(value: StageKey | str) -> str:
    if isinstance(value, StageKey):
        return value.value
    candidate = str(value).strip()
    if not candidate:
        raise ValueError("stage key must be non-empty.")
    try:
        return StageKey(candidate).value
    except ValueError:
        return candidate


def _parse_utc(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except Exception:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _coerce_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _coerce_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return int(value)
    return None


def _normalize_stage_context(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(metadata, Mapping):
        return {}
    payload = {str(key): value for key, value in dict(metadata).items()}

    # Canonical planned keys.
    if "planned_backend_family" not in payload and "assigned_backend_family" in payload:
        payload["planned_backend_family"] = payload.get("assigned_backend_family")
    if "planned_compute_lane" not in payload and "assigned_compute_lane" in payload:
        payload["planned_compute_lane"] = payload.get("assigned_compute_lane")
    if "planned_executor_id" not in payload and "assigned_executor_id" in payload:
        payload["planned_executor_id"] = payload.get("assigned_executor_id")

    # Canonical observed keys.
    if "observed_backend_family" not in payload:
        if "actual_estimator_backend_family" in payload:
            payload["observed_backend_family"] = payload.get("actual_estimator_backend_family")
        elif "backend_family" in payload:
            payload["observed_backend_family"] = payload.get("backend_family")
    if "observed_compute_lane" not in payload and "compute_lane" in payload:
        payload["observed_compute_lane"] = payload.get("compute_lane")
    if "observed_executor_id" not in payload:
        if "executor_id" in payload:
            payload["observed_executor_id"] = payload.get("executor_id")
        elif "actual_executor_id" in payload:
            payload["observed_executor_id"] = payload.get("actual_executor_id")

    artifacts = payload.get("primary_artifacts")
    if isinstance(artifacts, Sequence) and not isinstance(artifacts, (str, bytes)):
        payload["primary_artifacts"] = [str(item) for item in artifacts if str(item).strip()]
    elif artifacts is None:
        pass
    else:
        payload["primary_artifacts"] = [str(artifacts)]

    return _json_safe(payload)


class StageBoundaryRecorder:
    """Records stage boundary events and keeps additive observed stage evidence."""

    def __init__(
        self,
        *,
        report_dir: Path,
        run_id: str,
        progress_callback: ProgressCallback | None = None,
    ) -> None:
        self.report_dir = Path(report_dir)
        self.run_id = str(run_id)
        self.progress_callback = progress_callback
        self.stage_events_path = self.report_dir / _STAGE_EVENTS_FILENAME
        self.stage_observed_evidence_path = self.report_dir / _STAGE_OBSERVED_EVIDENCE_FILENAME
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.stage_events_path.touch(exist_ok=True)
        self._stage_rows: dict[str, dict[str, Any]] = self._load_existing_rows()

    def _load_existing_rows(self) -> dict[str, dict[str, Any]]:
        if not self.stage_observed_evidence_path.exists():
            return {}
        try:
            payload = json.loads(self.stage_observed_evidence_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        if not isinstance(payload, dict):
            return {}
        rows_payload = payload.get("stages")
        if not isinstance(rows_payload, list):
            return {}
        rows: dict[str, dict[str, Any]] = {}
        for row in rows_payload:
            if not isinstance(row, dict):
                continue
            key = row.get("stage_key")
            if not isinstance(key, str) or not key.strip():
                continue
            rows[str(key)] = dict(row)
        return rows

    def _stage_row(self, stage_key: str) -> dict[str, Any]:
        row = self._stage_rows.get(stage_key)
        if row is None:
            row = {
                "stage_key": str(stage_key),
                "run_id": str(self.run_id),
                "observed_status": "planned",
                "status_source": "stage_observability_default",
                "started_at_utc": None,
                "ended_at_utc": None,
                "duration_seconds": None,
                "event_count": 0,
                "primary_artifacts": [],
            }
            self._stage_rows[stage_key] = row
        return row

    def _write_stage_event(
        self,
        *,
        event_type: str,
        stage_key: str,
        status: str,
        timestamp_utc: str,
        metadata: dict[str, Any],
    ) -> None:
        payload = {
            "event_type": str(event_type),
            "stage_key": str(stage_key),
            "status": str(status),
            "timestamp_utc": str(timestamp_utc),
            "run_id": str(self.run_id),
            "metadata": _json_safe(dict(metadata)),
        }
        with self.stage_events_path.open("a", encoding="utf-8") as handle:
            handle.write(f"{json.dumps(payload, ensure_ascii=True)}\n")

        emit_progress(
            self.progress_callback,
            stage="stage",
            message=f"{event_type} stage {stage_key}",
            metadata={
                "run_id": str(self.run_id),
                "stage_key": str(stage_key),
                "status": str(status),
                **dict(metadata),
            },
        )

    @staticmethod
    def _duration_between(started_at_utc: str | None, ended_at_utc: str | None) -> float | None:
        started_dt = _parse_utc(started_at_utc)
        ended_dt = _parse_utc(ended_at_utc)
        if started_dt is None or ended_dt is None:
            return None
        return max(0.0, float((ended_dt - started_dt).total_seconds()))

    def _merge_context(self, row: dict[str, Any], metadata: Mapping[str, Any] | None) -> None:
        normalized = _normalize_stage_context(metadata)
        for key, value in normalized.items():
            if key == "primary_artifacts":
                existing = row.get("primary_artifacts")
                merged: list[str] = []
                if isinstance(existing, list):
                    merged.extend(str(item) for item in existing if str(item).strip())
                if isinstance(value, list):
                    merged.extend(str(item) for item in value if str(item).strip())
                # Keep deterministic order and uniqueness.
                deduped: list[str] = []
                seen: set[str] = set()
                for item in merged:
                    if item in seen:
                        continue
                    seen.add(item)
                    deduped.append(item)
                row["primary_artifacts"] = deduped
                continue
            row[str(key)] = value

    def _persist(self) -> None:
        payload = {
            "schema_version": "stage-observed-evidence-v1",
            "generated_at_utc": _utc_now(),
            "run_id": str(self.run_id),
            "stage_events_path": str(self.stage_events_path.resolve()),
            "stages": [
                self._stage_rows[key]
                for key in sorted(self._stage_rows)
            ],
        }
        self.stage_observed_evidence_path.write_text(
            f"{json.dumps(_json_safe(payload), indent=2)}\n",
            encoding="utf-8",
        )

    def stage_started(
        self,
        stage_key: StageKey | str,
        *,
        metadata: Mapping[str, Any] | None = None,
        timestamp_utc: str | None = None,
    ) -> None:
        key = _stage_key_value(stage_key)
        now_utc = str(timestamp_utc or _utc_now())
        row = self._stage_row(key)
        if row.get("started_at_utc") is None:
            row["started_at_utc"] = now_utc
        row["observed_status"] = "started"
        row["status_source"] = "stage_event_started"
        row["event_count"] = int(row.get("event_count", 0)) + 1
        self._merge_context(row, metadata)
        self._write_stage_event(
            event_type="stage_started",
            stage_key=key,
            status="started",
            timestamp_utc=now_utc,
            metadata=_normalize_stage_context(metadata),
        )
        self._persist()

    def stage_finished(
        self,
        stage_key: StageKey | str,
        *,
        metadata: Mapping[str, Any] | None = None,
        timestamp_utc: str | None = None,
        status: str = "executed",
    ) -> None:
        key = _stage_key_value(stage_key)
        now_utc = str(timestamp_utc or _utc_now())
        row = self._stage_row(key)
        if row.get("started_at_utc") is None:
            row["started_at_utc"] = now_utc
        row["ended_at_utc"] = now_utc
        row["observed_status"] = str(status)
        row["status_source"] = "stage_event_finished"
        row["event_count"] = int(row.get("event_count", 0)) + 1
        self._merge_context(row, metadata)
        row["duration_seconds"] = self._duration_between(
            row.get("started_at_utc"),
            row.get("ended_at_utc"),
        )
        self._write_stage_event(
            event_type="stage_finished",
            stage_key=key,
            status=str(status),
            timestamp_utc=now_utc,
            metadata=_normalize_stage_context(metadata),
        )
        self._persist()

    def stage_skipped(
        self,
        stage_key: StageKey | str,
        *,
        metadata: Mapping[str, Any] | None = None,
        reason: str | None = None,
        derived_from_stage: StageKey | str | None = None,
        timestamp_utc: str | None = None,
    ) -> None:
        key = _stage_key_value(stage_key)
        now_utc = str(timestamp_utc or _utc_now())
        payload = dict(metadata or {})
        if reason is not None:
            payload["skip_reason"] = str(reason)
        if derived_from_stage is not None:
            payload["derived_from_stage"] = _stage_key_value(derived_from_stage)
        row = self._stage_row(key)
        row["started_at_utc"] = row.get("started_at_utc") or now_utc
        row["ended_at_utc"] = now_utc
        row["observed_status"] = "skipped"
        row["status_source"] = "stage_event_skipped"
        row["event_count"] = int(row.get("event_count", 0)) + 1
        self._merge_context(row, payload)
        row["duration_seconds"] = self._duration_between(
            row.get("started_at_utc"),
            row.get("ended_at_utc"),
        )
        self._write_stage_event(
            event_type="stage_skipped",
            stage_key=key,
            status="skipped",
            timestamp_utc=now_utc,
            metadata=_normalize_stage_context(payload),
        )
        self._persist()

    def stage_reused(
        self,
        stage_key: StageKey | str,
        *,
        metadata: Mapping[str, Any] | None = None,
        derived_from_stage: StageKey | str | None = None,
        timestamp_utc: str | None = None,
    ) -> None:
        key = _stage_key_value(stage_key)
        now_utc = str(timestamp_utc or _utc_now())
        payload = dict(metadata or {})
        if derived_from_stage is not None:
            payload["derived_from_stage"] = _stage_key_value(derived_from_stage)
        row = self._stage_row(key)
        row["started_at_utc"] = row.get("started_at_utc") or now_utc
        row["ended_at_utc"] = now_utc
        row["observed_status"] = "reused"
        row["status_source"] = "stage_event_reused"
        row["event_count"] = int(row.get("event_count", 0)) + 1
        self._merge_context(row, payload)
        row["duration_seconds"] = self._duration_between(
            row.get("started_at_utc"),
            row.get("ended_at_utc"),
        )
        self._write_stage_event(
            event_type="stage_reused",
            stage_key=key,
            status="reused",
            timestamp_utc=now_utc,
            metadata=_normalize_stage_context(payload),
        )
        self._persist()

    def update_stage_context(
        self,
        stage_key: StageKey | str,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        key = _stage_key_value(stage_key)
        row = self._stage_row(key)
        row["event_count"] = int(row.get("event_count", 0))
        self._merge_context(row, metadata)
        self._persist()



def load_stage_observed_evidence(*, report_dir: Path) -> dict[str, dict[str, Any]]:
    path = Path(report_dir) / _STAGE_OBSERVED_EVIDENCE_FILENAME
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    rows = payload.get("stages")
    if not isinstance(rows, list):
        return {}
    observed: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        stage_key = row.get("stage_key")
        if not isinstance(stage_key, str) or not stage_key.strip():
            continue
        observed[str(stage_key)] = dict(row)
    return observed


def load_stage_events(*, report_dir: Path) -> list[dict[str, Any]]:
    return _load_jsonl(Path(report_dir) / _STAGE_EVENTS_FILENAME)


def _sample_interval_from_samples(samples: list[dict[str, Any]]) -> float | None:
    timestamps: list[float] = []
    for sample in samples:
        parsed = _parse_utc(sample.get("timestamp_utc"))
        if parsed is None:
            continue
        timestamps.append(parsed.timestamp())
    if len(timestamps) < 2:
        return None
    timestamps.sort()
    deltas = [timestamps[idx] - timestamps[idx - 1] for idx in range(1, len(timestamps))]
    positive = [delta for delta in deltas if delta > 0.0]
    if not positive:
        return None
    return float(median(positive))


def _resource_coverage_quality(
    *,
    duration_seconds: float | None,
    sample_count: int,
    sample_interval_seconds: float | None,
) -> tuple[str, float]:
    if sample_count <= 0:
        return "none", 0.0
    if duration_seconds is None or duration_seconds <= 0.0:
        return ("partial", 0.25)
    if sample_interval_seconds is None or sample_interval_seconds <= 0.0:
        return ("partial", 0.5)

    expected_samples = int(max(1.0, (duration_seconds / sample_interval_seconds))) + 1
    ratio = float(sample_count) / float(max(expected_samples, 1))
    if ratio >= 0.75 and sample_count >= 2:
        return "high", min(1.0, ratio)
    if ratio >= 0.25:
        return "partial", min(1.0, ratio)
    return "none", min(1.0, ratio)


def attribute_stage_resources(
    *,
    report_dir: Path,
    process_profile_summary: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    observed = load_stage_observed_evidence(report_dir=report_dir)
    if not observed:
        return {}

    samples_path = Path(report_dir) / "process_samples.jsonl"
    samples = _load_jsonl(samples_path)
    if not samples:
        return {}

    sample_interval_seconds: float | None = None
    if isinstance(process_profile_summary, Mapping):
        sample_interval_seconds = _coerce_float(process_profile_summary.get("sample_interval_seconds"))
    if sample_interval_seconds is None:
        sample_interval_seconds = _sample_interval_from_samples(samples)

    parsed_samples: list[tuple[datetime, dict[str, Any]]] = []
    for sample in samples:
        ts = _parse_utc(sample.get("timestamp_utc"))
        if ts is None:
            continue
        parsed_samples.append((ts, sample))
    parsed_samples.sort(key=lambda item: item[0])
    if not parsed_samples:
        return {}

    summaries: dict[str, dict[str, Any]] = {}
    for stage_key, row in observed.items():
        started = _parse_utc(row.get("started_at_utc"))
        ended = _parse_utc(row.get("ended_at_utc"))
        if started is None or ended is None:
            continue
        if ended < started:
            continue

        stage_samples = [
            sample
            for ts, sample in parsed_samples
            if ts >= started and ts <= ended
        ]
        sample_count = int(len(stage_samples))
        duration_seconds = float(max(0.0, (ended - started).total_seconds()))
        coverage, coverage_ratio = _resource_coverage_quality(
            duration_seconds=duration_seconds,
            sample_count=sample_count,
            sample_interval_seconds=sample_interval_seconds,
        )

        cpu_values = [
            float(value)
            for value in (_coerce_float(sample.get("cpu_percent")) for sample in stage_samples)
            if value is not None
        ]
        rss_values = [
            float(value)
            for value in (_coerce_float(sample.get("rss_mb")) for sample in stage_samples)
            if value is not None
        ]
        vms_values = [
            float(value)
            for value in (_coerce_float(sample.get("vms_mb")) for sample in stage_samples)
            if value is not None
        ]
        thread_values = [
            int(value)
            for value in (_coerce_int(sample.get("num_threads")) for sample in stage_samples)
            if value is not None
        ]

        read_start = _coerce_int(stage_samples[0].get("read_bytes")) if stage_samples else None
        read_end = _coerce_int(stage_samples[-1].get("read_bytes")) if stage_samples else None
        write_start = _coerce_int(stage_samples[0].get("write_bytes")) if stage_samples else None
        write_end = _coerce_int(stage_samples[-1].get("write_bytes")) if stage_samples else None

        gpu_util_values = [
            float(value)
            for value in (
                _coerce_float(sample.get("gpu_utilization_percent")) for sample in stage_samples
            )
            if value is not None
        ]
        gpu_memory_values = [
            float(value)
            for value in (_coerce_float(sample.get("gpu_memory_mb")) for sample in stage_samples)
            if value is not None
        ]

        summaries[stage_key] = {
            "duration_seconds": float(round(duration_seconds, 6)),
            "sample_count": int(sample_count),
            "sample_interval_seconds": (
                float(round(sample_interval_seconds, 6))
                if isinstance(sample_interval_seconds, float)
                else None
            ),
            "mean_cpu_percent": (
                float(round(sum(cpu_values) / len(cpu_values), 6)) if cpu_values else None
            ),
            "peak_cpu_percent": (float(round(max(cpu_values), 6)) if cpu_values else None),
            "peak_rss_mb": (float(round(max(rss_values), 6)) if rss_values else None),
            "peak_vms_mb": (float(round(max(vms_values), 6)) if vms_values else None),
            "peak_thread_count": (int(max(thread_values)) if thread_values else None),
            "read_bytes_delta": (
                int(max(0, int(read_end) - int(read_start)))
                if read_start is not None and read_end is not None
                else None
            ),
            "write_bytes_delta": (
                int(max(0, int(write_end) - int(write_start)))
                if write_start is not None and write_end is not None
                else None
            ),
            "gpu_sample_count": int(max(len(gpu_util_values), len(gpu_memory_values))),
            "peak_gpu_utilization_percent": (
                float(round(max(gpu_util_values), 6)) if gpu_util_values else None
            ),
            "peak_gpu_memory_mb": (
                float(round(max(gpu_memory_values), 6)) if gpu_memory_values else None
            ),
            "mean_gpu_utilization_percent": (
                float(round(sum(gpu_util_values) / len(gpu_util_values), 6))
                if gpu_util_values
                else None
            ),
            "resource_coverage": str(coverage),
            "resource_coverage_ratio": float(round(coverage_ratio, 6)),
            "evidence_quality": (
                "high" if coverage == "high" else ("medium" if coverage == "partial" else "low")
            ),
        }
    return summaries


def merge_stage_resource_attribution(
    *,
    report_dir: Path,
    process_profile_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    observed = load_stage_observed_evidence(report_dir=report_dir)
    if not observed:
        return {
            "schema_version": "stage-resource-attribution-v1",
            "generated_at_utc": _utc_now(),
            "stage_resource_summaries": {},
            "resource_coverage_counts": {},
            "process_samples_path": str((Path(report_dir) / "process_samples.jsonl").resolve()),
        }

    stage_resource_summaries = attribute_stage_resources(
        report_dir=report_dir,
        process_profile_summary=process_profile_summary,
    )
    for stage_key, stage_summary in stage_resource_summaries.items():
        row = observed.get(stage_key)
        if row is None:
            continue
        row["resource_summary"] = dict(stage_summary)
        row["resource_coverage"] = stage_summary.get("resource_coverage")
        row["evidence_quality"] = stage_summary.get("evidence_quality")

    coverage_counter = Counter(
        str(summary.get("resource_coverage") or "none")
        for summary in stage_resource_summaries.values()
    )

    evidence_payload = {
        "schema_version": "stage-observed-evidence-v1",
        "generated_at_utc": _utc_now(),
        "run_id": (
            str(next(iter(observed.values())).get("run_id"))
            if observed
            else None
        ),
        "stage_events_path": str((Path(report_dir) / _STAGE_EVENTS_FILENAME).resolve()),
        "stages": [observed[key] for key in sorted(observed)],
    }
    (Path(report_dir) / _STAGE_OBSERVED_EVIDENCE_FILENAME).write_text(
        f"{json.dumps(_json_safe(evidence_payload), indent=2)}\n",
        encoding="utf-8",
    )

    return {
        "schema_version": "stage-resource-attribution-v1",
        "generated_at_utc": _utc_now(),
        "stage_resource_summaries": {
            key: stage_resource_summaries[key] for key in sorted(stage_resource_summaries)
        },
        "resource_coverage_counts": {
            str(key): int(value) for key, value in sorted(coverage_counter.items())
        },
        "process_samples_path": str((Path(report_dir) / "process_samples.jsonl").resolve()),
        "stage_events_path": str((Path(report_dir) / _STAGE_EVENTS_FILENAME).resolve()),
    }


__all__ = [
    "StageBoundaryRecorder",
    "attribute_stage_resources",
    "load_stage_events",
    "load_stage_observed_evidence",
    "merge_stage_resource_attribution",
]
