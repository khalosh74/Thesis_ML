from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

RUN_STATUS_FILENAME = "run_status.json"


def _utc_timestamp() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def run_status_path(report_dir: Path) -> Path:
    return Path(report_dir) / RUN_STATUS_FILENAME


def read_run_status(report_dir: Path) -> dict[str, Any] | None:
    status_path = run_status_path(report_dir)
    if not status_path.exists():
        return None
    try:
        payload = json.loads(status_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def write_run_status(
    report_dir: Path,
    *,
    run_id: str,
    status: str,
    message: str | None = None,
    error: str | None = None,
    executed_sections: list[str] | None = None,
    reused_sections: list[str] | None = None,
) -> Path:
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "run_id": str(run_id),
        "status": str(status),
        "updated_at_utc": _utc_timestamp(),
    }
    if message:
        payload["message"] = str(message)
    if error:
        payload["error"] = str(error)
    if executed_sections is not None:
        payload["executed_sections"] = [str(section) for section in executed_sections]
    if reused_sections is not None:
        payload["reused_sections"] = [str(section) for section in reused_sections]

    path = run_status_path(report_dir)
    path.write_text(f"{json.dumps(payload, indent=2)}\n", encoding="utf-8")
    return path


def prepare_report_dir(
    report_dir: Path,
    *,
    run_id: str,
    force: bool,
    resume: bool,
) -> str:
    report_dir = Path(report_dir)
    if force and resume:
        raise ValueError("force=True and resume=True are mutually exclusive.")

    if not report_dir.exists():
        if resume:
            raise FileNotFoundError(
                f"Cannot resume run '{run_id}': output directory does not exist ({report_dir})."
            )
        report_dir.mkdir(parents=True, exist_ok=False)
        return "fresh"

    if not report_dir.is_dir():
        raise ValueError(f"Run output path exists but is not a directory: {report_dir}")

    status_payload = read_run_status(report_dir)
    status_value = str(status_payload.get("status", "unknown")) if status_payload else "unknown"
    updated_at = (
        str(status_payload.get("updated_at_utc", "unknown")) if status_payload else "unknown"
    )

    if force:
        shutil.rmtree(report_dir)
        report_dir.mkdir(parents=True, exist_ok=False)
        return "forced_rerun"

    if status_value == "completed":
        raise FileExistsError(
            f"Run '{run_id}' already completed at {updated_at}. "
            "Use force=True to rerun from scratch with the same run_id, "
            "or provide a new run_id."
        )

    if resume:
        return "resume"

    raise RuntimeError(
        f"Run output directory already exists for run '{run_id}' with status='{status_value}'. "
        "Refusing to continue without explicit mode. Use resume=True to continue a partial run "
        "or force=True to rerun from scratch."
    )


__all__ = [
    "RUN_STATUS_FILENAME",
    "prepare_report_dir",
    "read_run_status",
    "run_status_path",
    "write_run_status",
]
