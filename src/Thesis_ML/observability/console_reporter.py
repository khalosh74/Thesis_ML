from __future__ import annotations

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


def _normalize_progress_ui(value: str) -> str:
    normalized = str(value or "auto").strip().lower()
    if normalized not in PROGRESS_UI_CHOICES:
        raise ValueError(
            f"progress_ui must be one of {PROGRESS_UI_CHOICES}, got '{value}'."
        )
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

    def _write_line(self, line: str) -> None:
        self.stream.write(f"{line}\n")
        self.stream.flush()

    def _elapsed_seconds(self, live_status: dict[str, Any]) -> float | None:
        started = _parse_iso_timestamp(live_status.get("started_at_utc"))
        updated = _parse_iso_timestamp(live_status.get("last_updated_at_utc")) or _parse_iso_timestamp(
            _utc_now_iso()
        )
        if started is None or updated is None:
            return None
        return float(max(0.0, (updated - started).total_seconds()))

    def _counts(self, live_status: dict[str, Any]) -> dict[str, int]:
        raw = live_status.get("counts")
        if not isinstance(raw, dict):
            return {}
        return {str(key): int(value) for key, value in raw.items() if isinstance(value, (int, float))}

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
        if event_type not in _STAGE_COMPLETION_EVENT_TYPES:
            return None
        timestamp = str(event.get("timestamp_utc") or _utc_now_iso())
        phase_name = str(event.get("phase_name") or event_metadata.get("phase_name") or "")
        experiment_id = str(event.get("experiment_id") or event_metadata.get("experiment_id") or "")
        run_id = str(event.get("run_id") or event_metadata.get("run_id") or "")
        stage_key = str(event_metadata.get("stage_key") or "")
        status = str(event.get("status") or event_metadata.get("status") or "").strip() or (
            "skipped" if event_type == "stage_skipped" else "executed"
        )
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

        self._console = _RichConsole(file=self.stream, soft_wrap=True)
        self._progress = _RichProgress(
            _RichTextColumn("{task.description}"),
            _RichBarColumn(),
            _RichMofNCompleteColumn(),
            _RichTextColumn("eta p50={task.fields[eta_p50]} p80={task.fields[eta_p80]}"),
            _RichTextColumn("anom(i/w/e)={task.fields[anomalies]}"),
            auto_refresh=False,
            transient=False,
            console=self._console,
        )
        self._task_id: int | None = None

    def _start_progress(self) -> None:
        if self.quiet or self._progress_started:
            return
        self._progress.start()
        self._task_id = self._progress.add_task(
            "phase=n/a elapsed=00:00 active=0",
            total=1.0,
            completed=0.0,
            eta_p50="n/a",
            eta_p80="n/a",
            anomalies="0/0/0",
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
        return {str(key): int(value) for key, value in raw.items() if isinstance(value, (int, float))}

    def _elapsed_seconds(self, live_status: dict[str, Any]) -> float | None:
        started = _parse_iso_timestamp(live_status.get("started_at_utc"))
        updated = _parse_iso_timestamp(live_status.get("last_updated_at_utc")) or _parse_iso_timestamp(
            _utc_now_iso()
        )
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

    def _update_progress_bar(self, live_status: dict[str, Any]) -> None:
        if self.quiet:
            return
        self._start_progress()
        if self._task_id is None:
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
        planned_units = int(counts.get("runs_planned", 0))
        candidate_total = max(planned_units, done_units + active_runs_count, 1)
        self._max_total_units_seen = max(int(self._max_total_units_seen), int(candidate_total))
        total_units = int(self._max_total_units_seen)
        completed_units = int(min(done_units, total_units))
        phase_name = str(live_status.get("current_phase") or "n/a")
        elapsed = _format_seconds(self._elapsed_seconds(live_status))
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
        description = (
            f"phase={phase_name} elapsed={elapsed} active={active_runs_count} "
            f"runs(c/p/s/b/f/d)="
            f"{int(counts.get('runs_completed', 0))}/"
            f"{int(counts.get('runs_planned', 0))}/"
            f"{int(counts.get('runs_started', 0))}/"
            f"{int(counts.get('runs_blocked', 0))}/"
            f"{int(counts.get('runs_failed', 0))}/"
            f"{int(counts.get('runs_dry_run', 0))}"
        )
        self._progress.update(
            self._task_id,
            total=float(total_units),
            completed=float(completed_units),
            description=description,
            eta_p50=eta_p50,
            eta_p80=eta_p80,
            anomalies=self._render_anomalies(live_status),
        )
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
        if event_type not in _STAGE_COMPLETION_EVENT_TYPES:
            return None
        phase_name = str(event.get("phase_name") or event_metadata.get("phase_name") or "")
        experiment_id = str(event.get("experiment_id") or event_metadata.get("experiment_id") or "")
        run_id = str(event.get("run_id") or event_metadata.get("run_id") or "")
        stage_key = str(event_metadata.get("stage_key") or "")
        status = str(event.get("status") or event_metadata.get("status") or "").strip() or (
            "skipped" if event_type == "stage_skipped" else "executed"
        )
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


def build_progress_reporter(
    *,
    stream: TextIO | None = None,
    interval_seconds: float = 15.0,
    quiet: bool = False,
    progress_ui: str = "auto",
    progress_detail: str = "experiment_stage",
) -> EventReporter:
    reporter_stream = stream if stream is not None else sys.stdout
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
