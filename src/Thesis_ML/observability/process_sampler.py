from __future__ import annotations

import json
import threading
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

try:  # pragma: no cover - environment dependent import guard
    import psutil
except Exception:  # pragma: no cover - environment dependent import guard
    psutil = None  # type: ignore[assignment]


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


class ProcessSampler:
    def __init__(
        self,
        *,
        pid: int,
        report_dir: Path,
        sample_interval_seconds: float = 10.0,
        include_io_counters: bool = True,
    ) -> None:
        self.pid = int(pid)
        self.report_dir = Path(report_dir)
        self.sample_interval_seconds = float(sample_interval_seconds)
        if self.sample_interval_seconds <= 0:
            self.sample_interval_seconds = 10.0
        self.include_io_counters = bool(include_io_counters)
        self.samples_path = self.report_dir / "process_samples.jsonl"
        self.summary_path = self.report_dir / "process_profile_summary.json"

        self._sampling_enabled = psutil is not None
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._start_perf: float | None = None
        self._first_sample_at_utc: str | None = None
        self._last_sample_at_utc: str | None = None
        self._sample_count = 0
        self._peak_rss_mb = 0.0
        self._peak_vms_mb = 0.0
        self._peak_thread_count = 0
        self._peak_cpu_percent = 0.0
        self._cpu_sum = 0.0
        self._peak_child_process_count = 0
        self._peak_read_bytes = 0
        self._peak_write_bytes = 0
        self._sampling_errors: list[str] = []
        self._process = None

    def _record_sampling_error(self, message: str) -> None:
        text = str(message).strip()
        if not text:
            return
        self._sampling_errors.append(text)

    def _append_sample(self, sample: dict[str, Any]) -> None:
        sample_payload = _json_safe(sample)
        try:
            with self.samples_path.open("a", encoding="utf-8") as handle:
                handle.write(f"{json.dumps(sample_payload, ensure_ascii=True)}\n")
        except Exception as exc:  # pragma: no cover - file system fault path
            self._record_sampling_error(f"sample_write_failed: {exc}")
            return
        timestamp = str(sample_payload.get("timestamp_utc") or _utc_now())
        if self._first_sample_at_utc is None:
            self._first_sample_at_utc = timestamp
        self._last_sample_at_utc = timestamp
        self._sample_count += 1
        cpu_percent = float(sample_payload.get("cpu_percent") or 0.0)
        rss_mb = float(sample_payload.get("rss_mb") or 0.0)
        vms_mb = float(sample_payload.get("vms_mb") or 0.0)
        num_threads = int(sample_payload.get("num_threads") or 0)
        child_process_count = int(sample_payload.get("child_process_count") or 0)
        read_bytes = int(sample_payload.get("read_bytes") or 0)
        write_bytes = int(sample_payload.get("write_bytes") or 0)
        self._cpu_sum += cpu_percent
        self._peak_cpu_percent = max(self._peak_cpu_percent, cpu_percent)
        self._peak_rss_mb = max(self._peak_rss_mb, rss_mb)
        self._peak_vms_mb = max(self._peak_vms_mb, vms_mb)
        self._peak_thread_count = max(self._peak_thread_count, num_threads)
        self._peak_child_process_count = max(
            self._peak_child_process_count,
            child_process_count,
        )
        self._peak_read_bytes = max(self._peak_read_bytes, read_bytes)
        self._peak_write_bytes = max(self._peak_write_bytes, write_bytes)

    def _build_fallback_sample(
        self,
        *,
        process_running: bool,
        elapsed_seconds: float,
        sample_error: str | None = None,
        status_note: str | None = None,
    ) -> dict[str, Any]:
        return {
            "timestamp_utc": _utc_now(),
            "elapsed_seconds": float(round(elapsed_seconds, 6)),
            "pid": int(self.pid),
            "process_running": bool(process_running),
            "cpu_percent": 0.0,
            "rss_mb": 0.0,
            "vms_mb": 0.0,
            "num_threads": 0,
            "read_bytes": 0,
            "write_bytes": 0,
            "child_process_count": 0,
            "sample_error": sample_error,
            "status_note": status_note,
        }

    def _sample_once(self, *, status_note: str | None = None) -> None:
        if self._start_perf is None:
            self._start_perf = perf_counter()
        elapsed_seconds = float(perf_counter() - self._start_perf)
        if not self._sampling_enabled:
            self._append_sample(
                self._build_fallback_sample(
                    process_running=False,
                    elapsed_seconds=elapsed_seconds,
                    sample_error="psutil_not_available",
                    status_note=status_note,
                )
            )
            return

        try:
            if self._process is None:
                self._process = psutil.Process(self.pid)  # type: ignore[operator]
            process_running = bool(self._process.is_running()) and (
                self._process.status() != psutil.STATUS_ZOMBIE  # type: ignore[union-attr]
            )
            cpu_percent = float(self._process.cpu_percent(interval=None))  # type: ignore[union-attr]
            memory = self._process.memory_info()  # type: ignore[union-attr]
            rss_mb = float(memory.rss) / (1024.0 * 1024.0)
            vms_mb = float(memory.vms) / (1024.0 * 1024.0)
            num_threads = int(self._process.num_threads())  # type: ignore[union-attr]
            read_bytes = 0
            write_bytes = 0
            if self.include_io_counters:
                try:
                    io_counters = self._process.io_counters()  # type: ignore[union-attr]
                    read_bytes = int(getattr(io_counters, "read_bytes", 0))
                    write_bytes = int(getattr(io_counters, "write_bytes", 0))
                except Exception as exc:  # pragma: no cover - OS specific
                    self._record_sampling_error(f"io_counters_unavailable: {exc}")
            try:
                child_process_count = int(len(self._process.children(recursive=True)))  # type: ignore[union-attr]
            except Exception as exc:  # pragma: no cover - OS specific
                child_process_count = 0
                self._record_sampling_error(f"children_unavailable: {exc}")
            self._append_sample(
                {
                    "timestamp_utc": _utc_now(),
                    "elapsed_seconds": float(round(elapsed_seconds, 6)),
                    "pid": int(self.pid),
                    "process_running": bool(process_running),
                    "cpu_percent": float(cpu_percent),
                    "rss_mb": float(round(rss_mb, 6)),
                    "vms_mb": float(round(vms_mb, 6)),
                    "num_threads": int(num_threads),
                    "read_bytes": int(read_bytes),
                    "write_bytes": int(write_bytes),
                    "child_process_count": int(child_process_count),
                    "sample_error": None,
                    "status_note": status_note,
                }
            )
        except Exception as exc:
            self._record_sampling_error(str(exc))
            self._append_sample(
                self._build_fallback_sample(
                    process_running=False,
                    elapsed_seconds=elapsed_seconds,
                    sample_error=str(exc),
                    status_note=status_note,
                )
            )

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            self._sample_once()
            if self._stop_event.wait(self.sample_interval_seconds):
                break

    def start(self) -> None:
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.samples_path.touch(exist_ok=True)
        self._start_perf = perf_counter()
        self._sample_once(status_note="sampler_started")
        self._thread = threading.Thread(
            target=self._run_loop,
            name=f"process-sampler-{self.pid}",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(self.sample_interval_seconds * 2.0, 1.0))
            self._thread = None
        self._sample_once(status_note="sampler_stopped")

    def finalize(
        self,
        *,
        wall_clock_elapsed_seconds: float,
        terminated_by_watchdog: bool,
        termination_method: str,
        child_pid: int | None = None,
    ) -> dict[str, Any]:
        mean_cpu = float(self._cpu_sum / self._sample_count) if self._sample_count > 0 else 0.0
        summary = {
            "sampling_enabled": bool(self._sampling_enabled),
            "sample_interval_seconds": float(self.sample_interval_seconds),
            "sample_count": int(self._sample_count),
            "first_sample_at_utc": self._first_sample_at_utc,
            "last_sample_at_utc": self._last_sample_at_utc,
            "wall_clock_elapsed_seconds": float(wall_clock_elapsed_seconds),
            "peak_rss_mb": float(round(self._peak_rss_mb, 6)),
            "peak_vms_mb": float(round(self._peak_vms_mb, 6)),
            "peak_thread_count": int(self._peak_thread_count),
            "mean_cpu_percent": float(round(mean_cpu, 6)),
            "peak_cpu_percent": float(round(self._peak_cpu_percent, 6)),
            "peak_child_process_count": int(self._peak_child_process_count),
            "peak_read_bytes": int(self._peak_read_bytes),
            "peak_write_bytes": int(self._peak_write_bytes),
            "sampling_errors": list(self._sampling_errors),
            "terminated_by_watchdog": bool(terminated_by_watchdog),
            "termination_method": str(termination_method),
            "child_pid": int(child_pid if child_pid is not None else self.pid),
        }
        self.summary_path.write_text(
            f"{json.dumps(_json_safe(summary), indent=2)}\n", encoding="utf-8"
        )
        return summary


__all__ = ["ProcessSampler"]
