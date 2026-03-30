from __future__ import annotations

import sys
import time
from datetime import UTC, datetime
from typing import Any, TextIO


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


class ConsoleReporter:
    def __init__(
        self,
        stream: TextIO | None = None,
        interval_seconds: float = 15.0,
        quiet: bool = False,
    ) -> None:
        self.stream = stream if stream is not None else sys.stdout
        self.interval_seconds = max(0.1, float(interval_seconds))
        self.quiet = bool(quiet)
        self._last_summary_at: float = 0.0
        self._last_error_anomaly_id: str | None = None

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
        event_name = str(event.get("event_name") or "").strip()
        important = {
            "campaign_started",
            "campaign_finished",
            "phase_started",
            "phase_finished",
            "run_failed",
            "run_blocked",
            "run_dry_run",
        }
        if event_name not in important:
            return
        if self.quiet and event_name not in {"campaign_finished", "run_failed"}:
            return
        timestamp = str(event.get("timestamp_utc") or _utc_now_iso())
        phase_name = str(event.get("phase_name") or "")
        experiment_id = str(event.get("experiment_id") or "")
        run_id = str(event.get("run_id") or "")
        status = str(event.get("status") or "")
        message = str(event.get("message") or "").strip()
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


__all__ = ["ConsoleReporter"]
