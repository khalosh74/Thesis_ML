from __future__ import annotations

import json
import os
import sys
import time
from datetime import UTC, datetime
from typing import Any, Protocol, TextIO

try:  # pragma: no cover - import path is environment dependent
    from rich.console import Console as _RichConsole
    from rich.progress import BarColumn as _RichBarColumn
    from rich.progress import MofNCompleteColumn as _RichMofNCompleteColumn
    from rich.progress import Progress as _RichProgress
    from rich.progress import TextColumn as _RichTextColumn

    _RICH_AVAILABLE = True
except Exception:  # pragma: no cover - import path is environment dependent
    _RichConsole = None
    _RichProgress = None
    _RichTextColumn = None
    _RichBarColumn = None
    _RichMofNCompleteColumn = None
    _RICH_AVAILABLE = False


PROGRESS_UI_CHOICES = ("auto", "legacy", "bar")
PROGRESS_DETAIL_CHOICES = ("summary", "experiment_stage", "verbose")
_STAGE_COMPLETION_EVENT_TYPES = {"stage_finished", "stage_skipped", "stage_reused"}
_STAGE_VERBOSE_EVENT_TYPES = _STAGE_COMPLETION_EVENT_TYPES | {"stage_started"}
_RUN_TERMINAL_EVENT_TYPES = {"run_finished", "run_failed", "run_blocked", "run_dry_run"}


def _utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _parse_iso_timestamp(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        if text.endswith("Z"):
            text = f"{text[:-1]}+00:00"
        parsed = datetime.fromisoformat(text)
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=UTC)
        return parsed.astimezone(UTC)
    except Exception:
        return None


def _format_seconds(value: float | None) -> str:
    if value is None:
        return "n/a"
    total = max(0, int(round(float(value))))
    minutes, seconds = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def _coerce_optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except Exception:
        return None


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


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


def _progress_percent(completed_units: float | None, total_units: float | None) -> float | None:
    if completed_units is None or total_units is None:
        return None
    if total_units <= 0.0:
        return None
    percent = (float(completed_units) / float(total_units)) * 100.0
    return float(min(max(percent, 0.0), 100.0))


def _shorten_message(message: str, *, max_chars: int = 72) -> str:
    normalized = str(message or "").strip()
    if len(normalized) <= int(max_chars):
        return normalized
    if int(max_chars) <= 3:
        return normalized[: int(max_chars)]
    return f"{normalized[: int(max_chars) - 3]}..."


def _normalize_progress_ui(value: str) -> str:
    normalized = str(value or "auto").strip().lower()
    if normalized not in PROGRESS_UI_CHOICES:
        raise ValueError(f"progress_ui must be one of {PROGRESS_UI_CHOICES}, got '{value}'.")
    return normalized


def _normalize_progress_detail(value: str) -> str:
    normalized = str(value or "experiment_stage").strip().lower()
    if normalized not in PROGRESS_DETAIL_CHOICES:
        raise ValueError(
            f"progress_detail must be one of {PROGRESS_DETAIL_CHOICES}, got '{value}'."
        )
    return normalized


def _detail_rank(value: str) -> int:
    normalized = _normalize_progress_detail(value)
    if normalized == "summary":
        return 0
    if normalized == "experiment_stage":
        return 1
    return 2


def _stream_is_tty(stream: TextIO) -> bool:
    isatty = getattr(stream, "isatty", None)
    if not callable(isatty):
        return False
    try:
        return bool(isatty())
    except Exception:
        return False


def rich_available() -> bool:
    return bool(_RICH_AVAILABLE)


class EventReporter(Protocol):
    def handle_event(self, event: dict[str, Any], live_status: dict[str, Any]) -> None: ...


class LegacyLineReporter:
    def __init__(
        self,
        stream: TextIO | None = None,
        interval_seconds: float = 15.0,
        quiet: bool = False,
        progress_detail: str = "experiment_stage",
    ) -> None:
        self.stream = stream if stream is not None else sys.stdout
        self.interval_seconds = max(0.1, float(interval_seconds))
        self.quiet = bool(quiet)
        self.progress_detail = _normalize_progress_detail(progress_detail)
        self._last_summary_at: float = 0.0
        self._last_error_anomaly_id: str | None = None
        self._dry_run_blocked_limit_per_experiment = 3
        self._dry_run_blocked_counts: dict[tuple[str, str], int] = {}
        self._dry_run_blocked_suppressed_notified: set[tuple[str, str]] = set()
        self._emitted_experiment_finished_ids: set[str] = set()
        self._emitted_stage_completion_keys: set[tuple[str, str, str, str, str]] = set()
        self._operation_state_by_run_id: dict[str, dict[str, Any]] = {}

    def _write_line(self, line: str) -> None:
        self.stream.write(f"{line}\n")
        self.stream.flush()

    def _elapsed_seconds(self, live_status: dict[str, Any]) -> float | None:
        started = _parse_iso_timestamp(live_status.get("started_at_utc"))
        updated = _parse_iso_timestamp(
            live_status.get("last_updated_at_utc")
        ) or _parse_iso_timestamp(_utc_now_iso())
        if started is None or updated is None:
            return None
        return float(max(0.0, (updated - started).total_seconds()))

    def _counts(self, live_status: dict[str, Any]) -> dict[str, int]:
        raw = live_status.get("counts")
        if not isinstance(raw, dict):
            return {}
        return {
            str(key): int(value) for key, value in raw.items() if isinstance(value, (int, float))
        }

    def _maybe_emit_latest_error_anomaly(self, live_status: dict[str, Any]) -> None:
        latest = live_status.get("latest_anomaly")
        if not isinstance(latest, dict):
            return
        severity = str(latest.get("severity") or "").strip().lower()
        if severity != "error":
            return
        anomaly_id = str(latest.get("anomaly_id") or "").strip()
        if not anomaly_id or anomaly_id == self._last_error_anomaly_id:
            return
        self._last_error_anomaly_id = anomaly_id
        code = str(latest.get("code") or "UNKNOWN")
        message = str(latest.get("message") or "").strip()
        self._write_line(f"[anomaly:error] {code}: {message}")

    def _supports_completion_feed(self) -> bool:
        return _detail_rank(self.progress_detail) >= 1

    def _supports_verbose_feed(self) -> bool:
        return _detail_rank(self.progress_detail) >= 2

    def _extract_stage_completion_payload(
        self,
        event: dict[str, Any],
    ) -> tuple[tuple[str, str, str, str, str], str] | None:
        if not self._supports_completion_feed():
            return None
        if str(event.get("event_name") or "").strip() != "progress":
            return None
        if str(event.get("stage") or "").strip() != "stage":
            return None
        metadata = event.get("metadata")
        event_metadata = dict(metadata) if isinstance(metadata, dict) else {}
        event_type = str(event_metadata.get("event_type") or "").strip().lower()
        allowed_event_types = (
            _STAGE_VERBOSE_EVENT_TYPES
            if self._supports_verbose_feed()
            else _STAGE_COMPLETION_EVENT_TYPES
        )
        if event_type not in allowed_event_types:
            return None
        timestamp = str(event.get("timestamp_utc") or _utc_now_iso())
        phase_name = str(event.get("phase_name") or event_metadata.get("phase_name") or "")
        experiment_id = str(event.get("experiment_id") or event_metadata.get("experiment_id") or "")
        run_id = str(event.get("run_id") or event_metadata.get("run_id") or "")
        stage_key = str(event_metadata.get("stage_key") or "")
        status = str(event.get("status") or event_metadata.get("status") or "").strip()
        if not status:
            if event_type == "stage_skipped":
                status = "skipped"
            elif event_type == "stage_started":
                status = "started"
            else:
                status = "executed"
        dedupe_key = (
            experiment_id,
            run_id,
            stage_key,
            event_type,
            status,
        )
        if dedupe_key in self._emitted_stage_completion_keys:
            return None
        self._emitted_stage_completion_keys.add(dedupe_key)

        message = str(event.get("message") or event_type.replace("_", " ")).strip()
        parts = [f"[event:{event_type}] {timestamp}", f"status={status}"]
        if phase_name:
            parts.append(f"phase={phase_name}")
        if experiment_id:
            parts.append(f"experiment={experiment_id}")
        if run_id:
            parts.append(f"run={run_id}")
        if stage_key:
            parts.append(f"stage_key={stage_key}")
        if message:
            parts.append(f"msg={message}")
        return dedupe_key, " ".join(parts)

    def _operation_elapsed_seconds(
        self,
        *,
        started_at_utc: str | None,
        event_timestamp_utc: str | None,
    ) -> float | None:
        started_dt = _parse_iso_timestamp(started_at_utc)
        current_dt = _parse_iso_timestamp(event_timestamp_utc) or _parse_iso_timestamp(
            _utc_now_iso()
        )
        if started_dt is None or current_dt is None:
            return None
        return float(max(0.0, (current_dt - started_dt).total_seconds()))

    def _format_operation_line(
        self,
        *,
        kind: str,
        timestamp_utc: str,
        run_id: str,
        experiment_id: str,
        stage: str,
        message: str,
        elapsed_seconds: float | None,
        percent_complete: float | None,
    ) -> str:
        parts = [f"[operation:{kind}] {timestamp_utc}", f"run={run_id}"]
        if experiment_id:
            parts.append(f"experiment={experiment_id}")
        if stage:
            parts.append(f"stage={stage}")
        if elapsed_seconds is not None:
            parts.append(f"elapsed={_format_seconds(elapsed_seconds)}")
        if percent_complete is not None:
            parts.append(f"percent={int(round(percent_complete))}%")
        if message:
            parts.append(f"msg={_shorten_message(message)}")
        return " ".join(parts)

    def _extract_operation_progress_lines(self, event: dict[str, Any]) -> list[str]:
        if not self._supports_verbose_feed():
            return []
        event_name = str(event.get("event_name") or "").strip().lower()
        timestamp_utc = str(event.get("timestamp_utc") or _utc_now_iso())
        run_id = _normalize_text(event.get("run_id"))
        if not run_id:
            return []
        experiment_id = _normalize_text(event.get("experiment_id"))

        if event_name in _RUN_TERMINAL_EVENT_TYPES:
            state = self._operation_state_by_run_id.pop(run_id, None)
            if not isinstance(state, dict):
                return []
            if bool(state.get("finish_emitted", False)):
                return []
            stage = _normalize_text(state.get("stage"))
            message = _normalize_text(state.get("message")) or _normalize_text(event.get("message"))
            percent_complete = _coerce_optional_float(state.get("last_percent_complete"))
            elapsed_seconds = self._operation_elapsed_seconds(
                started_at_utc=(
                    str(state.get("started_at_utc")) if state.get("started_at_utc") else None
                ),
                event_timestamp_utc=timestamp_utc,
            )
            return [
                self._format_operation_line(
                    kind="finish",
                    timestamp_utc=timestamp_utc,
                    run_id=run_id,
                    experiment_id=experiment_id or _normalize_text(state.get("experiment_id")),
                    stage=stage,
                    message=message,
                    elapsed_seconds=elapsed_seconds,
                    percent_complete=percent_complete,
                )
            ]

        if event_name != "progress":
            return []

        stage = _normalize_text(event.get("stage"))
        message = _normalize_text(event.get("message"))
        metadata = event.get("metadata")
        event_metadata = dict(metadata) if isinstance(metadata, dict) else {}
        status = _normalize_text(event.get("status")) or _normalize_text(
            event_metadata.get("status")
        )
        completed_units = _coerce_optional_float(event.get("completed_units"))
        total_units = _coerce_optional_float(event.get("total_units"))
        percent_complete = _progress_percent(completed_units, total_units)
        milestone_bucket = (
            max(0, min(10, int(percent_complete // 10.0))) if percent_complete is not None else None
        )

        state = self._operation_state_by_run_id.setdefault(
            run_id,
            {
                "started_at_utc": timestamp_utc,
                "last_bucket": -1,
                "finish_emitted": False,
            },
        )
        if stage:
            state["stage"] = stage
        if message:
            state["message"] = message
        if experiment_id:
            state["experiment_id"] = experiment_id
        if percent_complete is not None:
            state["last_percent_complete"] = float(percent_complete)

        elapsed_seconds = self._operation_elapsed_seconds(
            started_at_utc=str(state.get("started_at_utc") or timestamp_utc),
            event_timestamp_utc=timestamp_utc,
        )

        lines: list[str] = []
        if not bool(state.get("start_emitted", False)):
            state["start_emitted"] = True
            lines.append(
                self._format_operation_line(
                    kind="start",
                    timestamp_utc=timestamp_utc,
                    run_id=run_id,
                    experiment_id=experiment_id,
                    stage=stage,
                    message=message,
                    elapsed_seconds=elapsed_seconds,
                    percent_complete=percent_complete,
                )
            )

        finish_from_stage_event = (
            _normalize_text(event_metadata.get("event_type")).lower()
            in _STAGE_COMPLETION_EVENT_TYPES
        )
        finish_from_percent = percent_complete is not None and percent_complete >= 100.0
        finish_from_status = status.lower() in {
            "completed",
            "finished",
            "executed",
            "reused",
            "skipped",
        }
        should_emit_finish = finish_from_stage_event or finish_from_percent or finish_from_status

        last_bucket = int(state.get("last_bucket", -1))
        if (
            milestone_bucket is not None
            and milestone_bucket > last_bucket
            and milestone_bucket > 0
            and not (should_emit_finish and milestone_bucket >= 10)
        ):
            state["last_bucket"] = milestone_bucket
            lines.append(
                self._format_operation_line(
                    kind="progress",
                    timestamp_utc=timestamp_utc,
                    run_id=run_id,
                    experiment_id=experiment_id,
                    stage=stage,
                    message=message,
                    elapsed_seconds=elapsed_seconds,
                    percent_complete=percent_complete,
                )
            )

        if should_emit_finish and not bool(state.get("finish_emitted", False)):
            state["finish_emitted"] = True
            state["last_bucket"] = max(int(state.get("last_bucket", -1)), 10)
            lines.append(
                self._format_operation_line(
                    kind="finish",
                    timestamp_utc=timestamp_utc,
                    run_id=run_id,
                    experiment_id=experiment_id,
                    stage=stage,
                    message=message,
                    elapsed_seconds=elapsed_seconds,
                    percent_complete=percent_complete,
                )
            )
            self._operation_state_by_run_id.pop(run_id, None)
        else:
            self._operation_state_by_run_id[run_id] = state

        return lines

    def handle_event(self, event: dict[str, Any], live_status: dict[str, Any]) -> None:
        self.emit_immediate_line(event, live_status)
        self._maybe_emit_latest_error_anomaly(live_status)
        now = time.monotonic()
        if self.quiet:
            return
        if (now - self._last_summary_at) >= self.interval_seconds:
            self.emit_summary_line(live_status)
            self._last_summary_at = now

    def emit_summary_line(self, live_status: dict[str, Any]) -> None:
        counts = self._counts(live_status)
        elapsed = _format_seconds(self._elapsed_seconds(live_status))
        current_phase = str(live_status.get("current_phase") or "n/a")
        active_runs = live_status.get("active_runs")
        active_runs_count = len(active_runs) if isinstance(active_runs, list) else 0
        eta_p50 = _format_seconds(
            None
            if live_status.get("eta_p50_seconds") in (None, "")
            else float(live_status.get("eta_p50_seconds"))
        )
        eta_p80 = _format_seconds(
            None
            if live_status.get("eta_p80_seconds") in (None, "")
            else float(live_status.get("eta_p80_seconds"))
        )
        anomaly_counts = live_status.get("anomaly_counts")
        by_severity = {}
        if isinstance(anomaly_counts, dict) and isinstance(anomaly_counts.get("by_severity"), dict):
            by_severity = anomaly_counts.get("by_severity", {})
        anomalies_text = (
            f"anom(i/w/e)="
            f"{int(by_severity.get('info', 0))}/"
            f"{int(by_severity.get('warning', 0))}/"
            f"{int(by_severity.get('error', 0))}"
        )
        self._write_line(
            "[progress] "
            f"elapsed={elapsed} "
            f"phase={current_phase} "
            f"runs(c/p/s/b/f/d)="
            f"{int(counts.get('runs_completed', 0))}/"
            f"{int(counts.get('runs_planned', 0))}/"
            f"{int(counts.get('runs_started', 0))}/"
            f"{int(counts.get('runs_blocked', 0))}/"
            f"{int(counts.get('runs_failed', 0))}/"
            f"{int(counts.get('runs_dry_run', 0))} "
            f"active_runs={active_runs_count} "
            f"eta_p50={eta_p50} "
            f"eta_p80={eta_p80} "
            f"{anomalies_text}"
        )

    def emit_immediate_line(self, event: dict[str, Any], live_status: dict[str, Any]) -> None:
        _ = live_status
        if self.quiet and str(event.get("event_name") or "").strip() not in {
            "campaign_finished",
            "run_failed",
        }:
            return

        operation_lines = self._extract_operation_progress_lines(event)
        for line in operation_lines:
            self._write_line(line)

        stage_completion_payload = self._extract_stage_completion_payload(event)
        if stage_completion_payload is not None:
            self._write_line(stage_completion_payload[1])
            return

        event_name = str(event.get("event_name") or "").strip()
        important = {
            "campaign_started",
            "campaign_finished",
            "phase_started",
            "phase_finished",
            "experiment_skipped",
            "run_failed",
            "run_blocked",
            "run_dry_run",
        }
        if self._supports_completion_feed():
            important.add("experiment_finished")
        if self._supports_verbose_feed():
            important.update(
                {
                    "experiment_started",
                    "run_planned",
                    "run_dispatched",
                    "run_started",
                    "run_finished",
                }
            )

        if event_name not in important:
            return

        timestamp = str(event.get("timestamp_utc") or _utc_now_iso())
        phase_name = str(event.get("phase_name") or "")
        experiment_id = str(event.get("experiment_id") or "")
        run_id = str(event.get("run_id") or "")
        status = str(event.get("status") or "")
        message = str(event.get("message") or "").strip()
        metadata = event.get("metadata")
        event_metadata = dict(metadata) if isinstance(metadata, dict) else {}

        if event_name == "experiment_finished":
            experiment_key = experiment_id.strip()
            if experiment_key and experiment_key in self._emitted_experiment_finished_ids:
                return
            if experiment_key:
                self._emitted_experiment_finished_ids.add(experiment_key)

        if event_name == "run_blocked" and bool(event_metadata.get("dry_run")):
            key = (phase_name, experiment_id)
            current = int(self._dry_run_blocked_counts.get(key, 0)) + 1
            self._dry_run_blocked_counts[key] = current
            if current > int(self._dry_run_blocked_limit_per_experiment):
                if key not in self._dry_run_blocked_suppressed_notified:
                    self._dry_run_blocked_suppressed_notified.add(key)
                    self._write_line(
                        "[event:run_blocked] "
                        f"{timestamp} status=blocked phase={phase_name or 'n/a'} "
                        f"experiment={experiment_id or 'n/a'} "
                        f"msg=additional blocked dry-run events suppressed after "
                        f"{self._dry_run_blocked_limit_per_experiment}"
                    )
                return
        parts = [f"[event:{event_name}] {timestamp}", f"status={status}"]
        if phase_name:
            parts.append(f"phase={phase_name}")
        if experiment_id:
            parts.append(f"experiment={experiment_id}")
        if run_id:
            parts.append(f"run={run_id}")
        if message:
            parts.append(f"msg={message}")
        self._write_line(" ".join(parts))


class RichLiveReporter:
    def __init__(
        self,
        stream: TextIO | None = None,
        interval_seconds: float = 15.0,
        quiet: bool = False,
        progress_detail: str = "experiment_stage",
    ) -> None:
        if not _RICH_AVAILABLE:
            raise RuntimeError("rich is not available in this environment.")
        self.stream = stream if stream is not None else sys.stdout
        self.interval_seconds = max(0.1, float(interval_seconds))
        self.quiet = bool(quiet)
        self.progress_detail = _normalize_progress_detail(progress_detail)
        self._last_error_anomaly_id: str | None = None
        self._emitted_experiment_finished_ids: set[str] = set()
        self._emitted_stage_completion_keys: set[tuple[str, str, str, str, str]] = set()
        self._max_total_units_seen = 1
        self._progress_started = False
        self._campaign_task_id: int | None = None
        self._phase_task_id: int | None = None
        self._operation_task_ids: dict[str, int] = {}

        self._console = _RichConsole(file=self.stream, soft_wrap=True)
        self._progress = _RichProgress(
            _RichTextColumn("{task.description}"),
            _RichBarColumn(),
            _RichMofNCompleteColumn(),
            _RichTextColumn("{task.fields[eta]}"),
            _RichTextColumn("{task.fields[meta]}"),
            auto_refresh=False,
            transient=False,
            console=self._console,
        )

    def _start_progress(self) -> None:
        if self.quiet or self._progress_started:
            return
        self._progress.start()
        self._campaign_task_id = self._progress.add_task(
            "campaign phase=n/a elapsed=00:00 active_runs=0 active_ops=0",
            total=1.0,
            completed=0.0,
            eta="camp_eta p50=n/a p80=n/a phase_eta p50=n/a p80=n/a",
            meta="anom(i/w/e)=0/0/0",
        )
        self._phase_task_id = self._progress.add_task(
            "phase=n/a elapsed=00:00 active_ops=0",
            total=None,
            completed=0.0,
            eta="phase_eta p50=n/a p80=n/a",
            meta="",
        )
        self._progress.refresh()
        self._progress_started = True

    def _stop_progress(self) -> None:
        if not self._progress_started:
            return
        try:
            self._progress.stop()
        finally:
            self._progress_started = False
            self._campaign_task_id = None
            self._phase_task_id = None
            self._operation_task_ids.clear()

    def _write_line(self, line: str) -> None:
        if self._progress_started:
            self._progress.console.print(line)
            self._progress.refresh()
            return
        self.stream.write(f"{line}\n")
        self.stream.flush()

    def _counts(self, live_status: dict[str, Any]) -> dict[str, int]:
        raw = live_status.get("counts")
        if not isinstance(raw, dict):
            return {}
        return {
            str(key): int(value) for key, value in raw.items() if isinstance(value, (int, float))
        }

    def _elapsed_seconds(self, live_status: dict[str, Any]) -> float | None:
        started = _parse_iso_timestamp(live_status.get("started_at_utc"))
        updated = _parse_iso_timestamp(
            live_status.get("last_updated_at_utc")
        ) or _parse_iso_timestamp(_utc_now_iso())
        if started is None or updated is None:
            return None
        return float(max(0.0, (updated - started).total_seconds()))

    def _render_anomalies(self, live_status: dict[str, Any]) -> str:
        anomaly_counts = live_status.get("anomaly_counts")
        by_severity = {}
        if isinstance(anomaly_counts, dict) and isinstance(anomaly_counts.get("by_severity"), dict):
            by_severity = anomaly_counts.get("by_severity", {})
        return (
            f"{int(by_severity.get('info', 0))}/"
            f"{int(by_severity.get('warning', 0))}/"
            f"{int(by_severity.get('error', 0))}"
        )

    def _operation_elapsed_seconds(
        self,
        operation: dict[str, Any],
        live_status: dict[str, Any],
    ) -> float | None:
        started = _parse_iso_timestamp(operation.get("started_at_utc"))
        updated = _parse_iso_timestamp(
            operation.get("last_updated_at_utc")
        ) or _parse_iso_timestamp(live_status.get("last_updated_at_utc"))
        if started is None or updated is None:
            return None
        return float(max(0.0, (updated - started).total_seconds()))

    def _sync_operation_tasks(self, live_status: dict[str, Any]) -> None:
        if self.quiet:
            return
        active_operations_raw = live_status.get("active_operations")
        active_operations = (
            dict(active_operations_raw) if isinstance(active_operations_raw, dict) else {}
        )
        desired_run_ids = {str(run_id) for run_id in active_operations}
        stale_run_ids = set(self._operation_task_ids) - desired_run_ids
        for run_id in sorted(stale_run_ids):
            task_id = self._operation_task_ids.pop(run_id)
            try:
                self._progress.remove_task(task_id)
            except Exception:
                pass

        for run_id in sorted(desired_run_ids):
            operation_raw = active_operations.get(run_id)
            operation = dict(operation_raw) if isinstance(operation_raw, dict) else {}
            task_id = self._operation_task_ids.get(run_id)
            if task_id is None:
                task_id = self._progress.add_task(
                    f"op exp=n/a run={run_id}",
                    total=None,
                    completed=0.0,
                    eta="progress=indeterminate",
                    meta="",
                )
                self._operation_task_ids[run_id] = task_id

            experiment_id = _normalize_text(operation.get("experiment_id")) or "n/a"
            stage_name = _normalize_text(operation.get("stage")) or "n/a"
            message = _shorten_message(_normalize_text(operation.get("message")))
            elapsed = _format_seconds(self._operation_elapsed_seconds(operation, live_status))
            description = (
                f"op exp={experiment_id} run={run_id} stage={stage_name} "
                f"elapsed={elapsed} msg={message or 'n/a'}"
            )

            completed_units = _coerce_optional_float(operation.get("completed_units"))
            total_units = _coerce_optional_float(operation.get("total_units"))
            percent_complete = _coerce_optional_float(operation.get("percent_complete"))
            if completed_units is not None and total_units is not None and total_units > 0.0:
                resolved_total = float(max(total_units, 1.0))
                resolved_completed = float(min(max(completed_units, 0.0), resolved_total))
            else:
                resolved_total = None
                resolved_completed = 0.0
            eta = (
                f"progress={int(round(percent_complete))}%"
                if percent_complete is not None
                else "progress=indeterminate"
            )
            self._progress.update(
                task_id,
                total=resolved_total,
                completed=resolved_completed,
                description=description,
                eta=eta,
                meta="",
            )

    def _update_progress_bar(self, live_status: dict[str, Any]) -> None:
        if self.quiet:
            return
        self._start_progress()
        if self._campaign_task_id is None or self._phase_task_id is None:
            return
        counts = self._counts(live_status)
        done_units = int(
            counts.get("runs_completed", 0)
            + counts.get("runs_failed", 0)
            + counts.get("runs_blocked", 0)
            + counts.get("runs_dry_run", 0)
        )
        active_runs = live_status.get("active_runs")
        active_runs_count = len(active_runs) if isinstance(active_runs, list) else 0
        active_operations_raw = live_status.get("active_operations")
        active_operations_count = (
            len(active_operations_raw) if isinstance(active_operations_raw, dict) else 0
        )
        planned_units = int(counts.get("runs_planned", 0))
        candidate_total = max(planned_units, done_units + active_runs_count, 1)
        self._max_total_units_seen = max(int(self._max_total_units_seen), int(candidate_total))
        total_units = int(self._max_total_units_seen)
        completed_units = int(min(done_units, total_units))
        phase_name = str(live_status.get("current_phase") or "n/a")
        elapsed = _format_seconds(self._elapsed_seconds(live_status))
        campaign_eta = live_status.get("campaign_eta")
        campaign_eta_payload = dict(campaign_eta) if isinstance(campaign_eta, dict) else {}
        phase_eta = live_status.get("phase_eta")
        phase_eta_payload = dict(phase_eta) if isinstance(phase_eta, dict) else {}
        campaign_eta_p50 = _coerce_optional_float(
            campaign_eta_payload.get("eta_p50_seconds", live_status.get("eta_p50_seconds"))
        )
        campaign_eta_p80 = _coerce_optional_float(
            campaign_eta_payload.get("eta_p80_seconds", live_status.get("eta_p80_seconds"))
        )
        phase_eta_p50 = _coerce_optional_float(phase_eta_payload.get("eta_p50_seconds"))
        phase_eta_p80 = _coerce_optional_float(phase_eta_payload.get("eta_p80_seconds"))
        description = (
            f"campaign phase={phase_name} elapsed={elapsed} "
            f"active_runs={active_runs_count} active_ops={active_operations_count} "
            f"runs(c/p/s/b/f/d)="
            f"{int(counts.get('runs_completed', 0))}/"
            f"{int(counts.get('runs_planned', 0))}/"
            f"{int(counts.get('runs_started', 0))}/"
            f"{int(counts.get('runs_blocked', 0))}/"
            f"{int(counts.get('runs_failed', 0))}/"
            f"{int(counts.get('runs_dry_run', 0))}"
        )
        self._progress.update(
            self._campaign_task_id,
            total=float(total_units),
            completed=float(completed_units),
            description=description,
            eta=(
                f"camp_eta p50={_format_seconds(campaign_eta_p50)} "
                f"p80={_format_seconds(campaign_eta_p80)} "
                f"phase_eta p50={_format_seconds(phase_eta_p50)} "
                f"p80={_format_seconds(phase_eta_p80)}"
            ),
            meta=f"anom(i/w/e)={self._render_anomalies(live_status)}",
        )

        phase_progress_raw = live_status.get("phase_progress")
        phase_progress = dict(phase_progress_raw) if isinstance(phase_progress_raw, dict) else {}
        phase_entry_raw = phase_progress.get(phase_name)
        phase_entry = dict(phase_entry_raw) if isinstance(phase_entry_raw, dict) else {}
        phase_avg_percent = _coerce_optional_float(phase_entry.get("avg_percent_complete"))
        phase_active_ops = int(phase_entry.get("active_operation_count") or 0)
        phase_eta_source = _normalize_text(phase_eta_payload.get("eta_source")) or "n/a"
        phase_eta_confidence = _normalize_text(phase_eta_payload.get("eta_confidence")) or "n/a"
        self._progress.update(
            self._phase_task_id,
            total=(100.0 if phase_avg_percent is not None else None),
            completed=(float(phase_avg_percent or 0.0)),
            description=(
                f"phase={phase_name} elapsed={elapsed} "
                f"active_ops={phase_active_ops if phase_active_ops > 0 else active_operations_count}"
            ),
            eta=(
                f"phase_eta p50={_format_seconds(phase_eta_p50)} "
                f"p80={_format_seconds(phase_eta_p80)}"
            ),
            meta=f"source={phase_eta_source} conf={phase_eta_confidence}",
        )
        self._sync_operation_tasks(live_status)
        self._progress.refresh()

    def _maybe_emit_latest_error_anomaly(self, live_status: dict[str, Any]) -> None:
        latest = live_status.get("latest_anomaly")
        if not isinstance(latest, dict):
            return
        severity = str(latest.get("severity") or "").strip().lower()
        if severity != "error":
            return
        anomaly_id = str(latest.get("anomaly_id") or "").strip()
        if not anomaly_id or anomaly_id == self._last_error_anomaly_id:
            return
        self._last_error_anomaly_id = anomaly_id
        code = str(latest.get("code") or "UNKNOWN")
        message = str(latest.get("message") or "").strip()
        self._write_line(f"[anomaly:error] {code}: {message}")

    def _supports_completion_feed(self) -> bool:
        return _detail_rank(self.progress_detail) >= 1

    def _supports_verbose_feed(self) -> bool:
        return _detail_rank(self.progress_detail) >= 2

    def _extract_stage_completion_line(self, event: dict[str, Any]) -> str | None:
        if not self._supports_completion_feed():
            return None
        if str(event.get("event_name") or "").strip() != "progress":
            return None
        if str(event.get("stage") or "").strip() != "stage":
            return None
        metadata = event.get("metadata")
        event_metadata = dict(metadata) if isinstance(metadata, dict) else {}
        event_type = str(event_metadata.get("event_type") or "").strip().lower()
        allowed_event_types = (
            _STAGE_VERBOSE_EVENT_TYPES
            if self._supports_verbose_feed()
            else _STAGE_COMPLETION_EVENT_TYPES
        )
        if event_type not in allowed_event_types:
            return None
        phase_name = str(event.get("phase_name") or event_metadata.get("phase_name") or "")
        experiment_id = str(event.get("experiment_id") or event_metadata.get("experiment_id") or "")
        run_id = str(event.get("run_id") or event_metadata.get("run_id") or "")
        stage_key = str(event_metadata.get("stage_key") or "")
        status = str(event.get("status") or event_metadata.get("status") or "").strip()
        if not status:
            if event_type == "stage_skipped":
                status = "skipped"
            elif event_type == "stage_started":
                status = "started"
            else:
                status = "executed"
        dedupe_key = (experiment_id, run_id, stage_key, event_type, status)
        if dedupe_key in self._emitted_stage_completion_keys:
            return None
        self._emitted_stage_completion_keys.add(dedupe_key)

        timestamp = str(event.get("timestamp_utc") or _utc_now_iso())
        message = str(event.get("message") or event_type.replace("_", " ")).strip()
        parts = [f"[event:{event_type}] {timestamp}", f"status={status}"]
        if phase_name:
            parts.append(f"phase={phase_name}")
        if experiment_id:
            parts.append(f"experiment={experiment_id}")
        if run_id:
            parts.append(f"run={run_id}")
        if stage_key:
            parts.append(f"stage_key={stage_key}")
        if message:
            parts.append(f"msg={message}")
        return " ".join(parts)

    def _emit_immediate_line(self, event: dict[str, Any]) -> None:
        event_name = str(event.get("event_name") or "").strip()
        if self.quiet and event_name not in {"campaign_finished", "run_failed"}:
            return

        stage_completion_line = self._extract_stage_completion_line(event)
        if stage_completion_line is not None:
            self._write_line(stage_completion_line)
            return

        important = {
            "campaign_started",
            "campaign_finished",
            "phase_started",
            "phase_finished",
            "experiment_skipped",
            "run_failed",
            "run_blocked",
            "run_dry_run",
        }
        if self._supports_completion_feed():
            important.add("experiment_finished")
        if self._supports_verbose_feed():
            important.update(
                {
                    "experiment_started",
                    "run_planned",
                    "run_dispatched",
                    "run_started",
                    "run_finished",
                }
            )
        if event_name not in important:
            return

        timestamp = str(event.get("timestamp_utc") or _utc_now_iso())
        phase_name = str(event.get("phase_name") or "")
        experiment_id = str(event.get("experiment_id") or "")
        run_id = str(event.get("run_id") or "")
        status = str(event.get("status") or "")
        message = str(event.get("message") or "").strip()

        if event_name == "experiment_finished":
            experiment_key = experiment_id.strip()
            if experiment_key and experiment_key in self._emitted_experiment_finished_ids:
                return
            if experiment_key:
                self._emitted_experiment_finished_ids.add(experiment_key)

        parts = [f"[event:{event_name}] {timestamp}", f"status={status}"]
        if phase_name:
            parts.append(f"phase={phase_name}")
        if experiment_id:
            parts.append(f"experiment={experiment_id}")
        if run_id:
            parts.append(f"run={run_id}")
        if message:
            parts.append(f"msg={message}")
        self._write_line(" ".join(parts))

    def handle_event(self, event: dict[str, Any], live_status: dict[str, Any]) -> None:
        try:
            self._update_progress_bar(live_status)
            self._emit_immediate_line(event)
            self._maybe_emit_latest_error_anomaly(live_status)
        finally:
            if str(event.get("event_name") or "").strip() == "campaign_finished":
                self._stop_progress()


ConsoleReporter = LegacyLineReporter


class JSONLineReporter:
    def __init__(self, stream: TextIO | None = None, quiet: bool = False) -> None:
        self.stream = stream if stream is not None else sys.stdout
        self.quiet = bool(quiet)

    def handle_event(self, event: dict[str, Any], live_status: dict[str, Any]) -> None:
        if self.quiet:
            # preserve the legacy quiet behaviour for critical events
            event_name = str(event.get("event_name") or "").strip()
            if event_name not in {"campaign_finished", "run_failed"}:
                return
        payload = {
            "timestamp_utc": str(
                event.get("timestamp_utc") or datetime.now(UTC).replace(microsecond=0).isoformat()
            ),
            "event": _json_safe(event),
            "live_status": _json_safe(live_status),
        }
        try:
            self.stream.write(json.dumps(payload, ensure_ascii=False) + "\n")
            self.stream.flush()
        except Exception:
            try:
                self.stream.write('{"error": "json_serialize_failed"}\n')
                self.stream.flush()
            except Exception:
                pass


def build_progress_reporter(
    *,
    stream: TextIO | None = None,
    interval_seconds: float = 15.0,
    quiet: bool = False,
    progress_ui: str = "auto",
    progress_detail: str = "experiment_stage",
) -> EventReporter:
    reporter_stream = stream if stream is not None else sys.stdout
    # If JSON log format requested globally, or output is not a TTY (CI),
    # emit structured JSON progress events.
    env_val = str(os.environ.get("THESIS_ML_LOG_FORMAT") or "").strip().lower()
    if env_val == "json" or not _stream_is_tty(reporter_stream):
        return JSONLineReporter(stream=reporter_stream, quiet=bool(quiet))
    normalized_ui = _normalize_progress_ui(progress_ui)
    normalized_detail = _normalize_progress_detail(progress_detail)

    if bool(quiet):
        return LegacyLineReporter(
            stream=reporter_stream,
            interval_seconds=interval_seconds,
            quiet=True,
            progress_detail=normalized_detail,
        )

    if normalized_ui == "legacy":
        return LegacyLineReporter(
            stream=reporter_stream,
            interval_seconds=interval_seconds,
            quiet=False,
            progress_detail=normalized_detail,
        )

    if normalized_ui == "bar":
        if _RICH_AVAILABLE:
            return RichLiveReporter(
                stream=reporter_stream,
                interval_seconds=interval_seconds,
                quiet=False,
                progress_detail=normalized_detail,
            )
        return LegacyLineReporter(
            stream=reporter_stream,
            interval_seconds=interval_seconds,
            quiet=False,
            progress_detail=normalized_detail,
        )

    if _RICH_AVAILABLE and _stream_is_tty(reporter_stream):
        return RichLiveReporter(
            stream=reporter_stream,
            interval_seconds=interval_seconds,
            quiet=False,
            progress_detail=normalized_detail,
        )
    return LegacyLineReporter(
        stream=reporter_stream,
        interval_seconds=interval_seconds,
        quiet=False,
        progress_detail=normalized_detail,
    )


__all__ = [
    "ConsoleReporter",
    "LegacyLineReporter",
    "RichLiveReporter",
    "EventReporter",
    "build_progress_reporter",
    "PROGRESS_UI_CHOICES",
    "PROGRESS_DETAIL_CHOICES",
    "rich_available",
]
