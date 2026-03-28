from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_REGISTRY_SCHEMA_VERSION = "config-registry-v1"


def resolve_config_registry_path(
    *,
    source_repo_root: Path | None = None,
    project_root: Path | None = None,
) -> Path | None:
    if source_repo_root is not None:
        source_candidate = Path(source_repo_root) / "configs" / "config_registry.json"
        if source_candidate.exists():
            return source_candidate.resolve()
    if project_root is not None:
        project_candidate = Path(project_root) / "configs" / "config_registry.json"
        if project_candidate.exists():
            return project_candidate.resolve()
    return None


def load_config_registry(
    *,
    source_repo_root: Path | None = None,
    project_root: Path | None = None,
    registry_path: Path | None = None,
) -> dict[str, Any]:
    resolved_registry_path = (
        Path(registry_path).resolve()
        if registry_path is not None
        else resolve_config_registry_path(
            source_repo_root=source_repo_root,
            project_root=project_root,
        )
    )
    if resolved_registry_path is None:
        raise FileNotFoundError("Config registry file not found.")

    payload = json.loads(resolved_registry_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Config registry must be a top-level JSON object.")
    if payload.get("schema_version") != _REGISTRY_SCHEMA_VERSION:
        raise ValueError(
            "Config registry schema_version mismatch: "
            f"expected '{_REGISTRY_SCHEMA_VERSION}', got '{payload.get('schema_version')}'."
        )
    aliases = payload.get("aliases")
    configs = payload.get("configs")
    if not isinstance(aliases, dict):
        raise ValueError("Config registry 'aliases' must be an object.")
    if not isinstance(configs, list):
        raise ValueError("Config registry 'configs' must be an array.")
    return payload


def _resolve_registry_root(
    *,
    source_repo_root: Path | None = None,
    project_root: Path | None = None,
    registry_path: Path | None = None,
) -> tuple[Path, Path]:
    resolved_registry_path = (
        Path(registry_path).resolve()
        if registry_path is not None
        else resolve_config_registry_path(
            source_repo_root=source_repo_root,
            project_root=project_root,
        )
    )
    if resolved_registry_path is None:
        raise FileNotFoundError("Config registry file not found.")
    return resolved_registry_path, resolved_registry_path.parent.parent.resolve()


def resolve_config_id(
    config_id: str,
    *,
    source_repo_root: Path | None = None,
    project_root: Path | None = None,
    registry_path: Path | None = None,
) -> Path:
    payload = load_config_registry(
        source_repo_root=source_repo_root,
        project_root=project_root,
        registry_path=registry_path,
    )
    _, registry_root = _resolve_registry_root(
        source_repo_root=source_repo_root,
        project_root=project_root,
        registry_path=registry_path,
    )
    target_config_id = str(config_id)
    for entry in payload["configs"]:
        if not isinstance(entry, dict):
            continue
        if str(entry.get("config_id")) != target_config_id:
            continue
        raw_path = str(entry.get("path", "")).strip()
        if not raw_path:
            raise ValueError(f"Config registry entry '{target_config_id}' has an empty path.")
        entry_path = Path(raw_path)
        if entry_path.is_absolute():
            return entry_path.resolve()
        return (registry_root / entry_path).resolve()
    raise KeyError(f"Unknown config_id in registry: '{target_config_id}'.")


def resolve_config_alias(
    alias: str,
    *,
    source_repo_root: Path | None = None,
    project_root: Path | None = None,
    registry_path: Path | None = None,
    fallback: str | Path | None = None,
) -> Path:
    resolved_registry_path = (
        Path(registry_path).resolve()
        if registry_path is not None
        else resolve_config_registry_path(
            source_repo_root=source_repo_root,
            project_root=project_root,
        )
    )
    if resolved_registry_path is None:
        if fallback is not None:
            return Path(fallback).resolve()
        raise FileNotFoundError("Config registry file not found.")

    payload = load_config_registry(registry_path=resolved_registry_path)
    aliases = payload["aliases"]
    alias_key = str(alias)
    if alias_key not in aliases:
        raise KeyError(f"Unknown config alias in registry: '{alias_key}'.")
    return resolve_config_id(str(aliases[alias_key]), registry_path=resolved_registry_path)


__all__ = [
    "load_config_registry",
    "resolve_config_alias",
    "resolve_config_id",
    "resolve_config_registry_path",
]
