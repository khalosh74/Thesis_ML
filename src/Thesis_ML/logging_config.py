"""Centralized logging configuration for Thesis_ML.

Provide a single `configure_logging()` entrypoint that CLI modules
and higher-level scripts can call to ensure consistent formatting,
level selection and optional file logging. Environment variables:

- `THESIS_ML_LOG_LEVEL` - desired log level (e.g. DEBUG, INFO).
- `THESIS_ML_LOG_FORMAT` - set to `json` to enable JSON output.
- `THESIS_ML_LOG_FILE` - optional path to a rotating log file.
- `THESIS_ML_LOG_FORCE` - if truthy, force reconfiguration even if
  the root logger already has handlers.

This module intentionally does not require external dependencies.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime
from logging.handlers import RotatingFileHandler


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:  # pragma: no cover - trivial
        payload = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "process": record.process,
            "thread": record.thread,
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def _parse_level(value: str | None) -> int:
    if value is None:
        return logging.INFO
    v = str(value).strip()
    if not v:
        return logging.INFO
    # numeric allowed
    try:
        return int(v)
    except Exception:
        pass
    vup = v.upper()
    return getattr(logging, vup, logging.INFO)


def configure_logging(
    *,
    level: str | int | None = None,
    json_format: bool | None = None,
    log_file: str | None = None,
    force: bool = False,
) -> None:
    """Configure root logging for Thesis_ML.

    Safe to call repeatedly; by default it will not reconfigure an
    already-configured root logger unless ``force`` is truthy.
    """
    env_level = os.environ.get("THESIS_ML_LOG_LEVEL")
    env_format = os.environ.get("THESIS_ML_LOG_FORMAT")
    env_file = os.environ.get("THESIS_ML_LOG_FILE")
    env_force = os.environ.get("THESIS_ML_LOG_FORCE")

    if level is None:
        level = env_level
    if json_format is None:
        json_format = bool(str(env_format or "").strip().lower() == "json")
    if log_file is None:
        log_file = env_file
    if not force:
        force = bool(str(env_force or "").strip().lower() in {"1", "true", "yes"})

    level_val = _parse_level(level)

    root = logging.getLogger()
    # If already configured and not forcing, just update level and return.
    if root.handlers and not force:
        root.setLevel(level_val)
        return

    if force:
        for h in list(root.handlers):
            try:
                root.removeHandler(h)
            except Exception:
                pass

    if json_format:
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s: %(message)s",
            "%Y-%m-%dT%H:%M:%S%z",
        )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    root.setLevel(level_val)
    root.addHandler(stream_handler)

    if log_file:
        try:
            file_handler = RotatingFileHandler(log_file, maxBytes=10 * 1024 * 1024, backupCount=5)
            file_handler.setFormatter(formatter)
            root.addHandler(file_handler)
        except Exception:
            # Avoid raising from logging configuration failures.
            root.warning("Failed to configure file logging to %s", log_file)

    # Route warnings from the warnings module into the logging system.
    logging.captureWarnings(True)
