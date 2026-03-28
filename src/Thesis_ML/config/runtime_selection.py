from __future__ import annotations

from pathlib import Path
from typing import Any

from Thesis_ML.config.config_registry import resolve_config_alias
from Thesis_ML.config.paths import PROJECT_ROOT, SOURCE_REPO_ROOT


def _normalize_optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def resolve_runtime_config_path(
    path_value: Any,
    alias_value: Any,
    *,
    default_alias: str,
    fallback_path: str | Path,
) -> Path:
    raw_path = _normalize_optional_text(path_value)
    if raw_path is not None:
        return Path(raw_path).resolve()

    explicit_alias = _normalize_optional_text(alias_value)
    selected_alias = explicit_alias or str(default_alias)

    fallback = Path(fallback_path).resolve() if explicit_alias is None else None
    return resolve_config_alias(
        selected_alias,
        source_repo_root=SOURCE_REPO_ROOT,
        project_root=PROJECT_ROOT,
        fallback=fallback,
    ).resolve()


__all__ = ["resolve_runtime_config_path"]
