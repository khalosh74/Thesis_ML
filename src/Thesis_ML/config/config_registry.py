from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_REGISTRY_SCHEMA_VERSION = "config-registry-v1"


def _default_source_repo_root() -> Path | None:
    candidate = Path(__file__).resolve().parents[3]
    registry_candidate = candidate / "configs" / "config_registry.json"
    if registry_candidate.exists():
        return candidate
    return None


def _default_project_root() -> Path:
    return Path.cwd().resolve()


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


def _resolve_registry_payload_and_root(
    *,
    source_repo_root: Path | None = None,
    project_root: Path | None = None,
    registry_path: Path | None = None,
) -> tuple[dict[str, Any] | None, Path | None]:
    resolved_registry_path = (
        Path(registry_path).resolve()
        if registry_path is not None
        else resolve_config_registry_path(
            source_repo_root=(
                source_repo_root if source_repo_root is not None else _default_source_repo_root()
            ),
            project_root=(project_root if project_root is not None else _default_project_root()),
        )
    )
    if resolved_registry_path is None:
        return None, None
    payload = load_config_registry(registry_path=resolved_registry_path)
    registry_root = resolved_registry_path.parent.parent.resolve()
    return payload, registry_root


def _entry_resolved_path(entry: dict[str, Any], *, registry_root: Path) -> Path | None:
    raw_path = str(entry.get("path", "")).strip()
    if not raw_path:
        return None
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate.resolve()
    return (registry_root / candidate).resolve()


def _aliases_for_config_id_from_payload(payload: dict[str, Any], config_id: str) -> list[str]:
    aliases_raw = payload.get("aliases")
    if not isinstance(aliases_raw, dict):
        return []
    target = str(config_id)
    aliases: list[str] = []
    for alias_name, alias_target in aliases_raw.items():
        if str(alias_target) == target:
            aliases.append(str(alias_name))
    return sorted(aliases)


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
    bundles = payload.get("bundles")
    if not isinstance(aliases, dict):
        raise ValueError("Config registry 'aliases' must be an object.")
    if not isinstance(configs, list):
        raise ValueError("Config registry 'configs' must be an array.")
    if bundles is not None and not isinstance(bundles, list):
        raise ValueError("Config registry 'bundles' must be an array when present.")
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


def get_config_entry(
    config_id: str,
    *,
    source_repo_root: Path | None = None,
    project_root: Path | None = None,
    registry_path: Path | None = None,
) -> dict[str, Any] | None:
    payload, _ = _resolve_registry_payload_and_root(
        source_repo_root=source_repo_root,
        project_root=project_root,
        registry_path=registry_path,
    )
    if payload is None:
        return None
    target = str(config_id)
    for entry in payload.get("configs", []):
        if not isinstance(entry, dict):
            continue
        if str(entry.get("config_id", "")) == target:
            return dict(entry)
    return None


def aliases_for_config_id(
    config_id: str,
    *,
    source_repo_root: Path | None = None,
    project_root: Path | None = None,
    registry_path: Path | None = None,
) -> list[str]:
    payload, _ = _resolve_registry_payload_and_root(
        source_repo_root=source_repo_root,
        project_root=project_root,
        registry_path=registry_path,
    )
    if payload is None:
        return []
    return _aliases_for_config_id_from_payload(payload, str(config_id))


def describe_config_path(
    path: str | Path,
    *,
    source_repo_root: Path | None = None,
    project_root: Path | None = None,
    registry_path: Path | None = None,
) -> dict[str, Any]:
    resolved_input_path = Path(path).resolve()
    payload, registry_root = _resolve_registry_payload_and_root(
        source_repo_root=source_repo_root,
        project_root=project_root,
        registry_path=registry_path,
    )

    path_relative: str | None = None
    if registry_root is not None:
        try:
            path_relative = str(resolved_input_path.relative_to(registry_root).as_posix())
        except ValueError:
            path_relative = None

    base_payload: dict[str, Any] = {
        "registered": False,
        "config_id": None,
        "kind": None,
        "family": None,
        "version": None,
        "lifecycle": None,
        "replay_allowed": None,
        "superseded_by": None,
        "path": str(resolved_input_path),
        "path_relative": path_relative,
        "aliases": [],
    }
    if payload is None or registry_root is None:
        return base_payload

    for entry in payload.get("configs", []):
        if not isinstance(entry, dict):
            continue
        entry_resolved_path = _entry_resolved_path(entry, registry_root=registry_root)
        if entry_resolved_path is None or entry_resolved_path != resolved_input_path:
            continue
        config_id = str(entry.get("config_id", "")).strip() or None
        superseded_by_raw = entry.get("superseded_by")
        superseded_by = (
            str(superseded_by_raw).strip() if superseded_by_raw is not None else None
        )
        if superseded_by == "":
            superseded_by = None
        return {
            "registered": True,
            "config_id": config_id,
            "kind": (str(entry.get("kind", "")).strip() or None),
            "family": (str(entry.get("family", "")).strip() or None),
            "version": (str(entry.get("version", "")).strip() or None),
            "lifecycle": (str(entry.get("lifecycle", "")).strip() or None),
            "replay_allowed": (
                bool(entry.get("replay_allowed")) if entry.get("replay_allowed") is not None else None
            ),
            "superseded_by": superseded_by,
            "path": str(resolved_input_path),
            "path_relative": path_relative,
            "aliases": (
                _aliases_for_config_id_from_payload(payload, config_id) if config_id is not None else []
            ),
        }

    return base_payload


def list_config_bundles(
    *,
    source_repo_root: Path | None = None,
    project_root: Path | None = None,
    registry_path: Path | None = None,
) -> list[dict[str, Any]]:
    payload, _ = _resolve_registry_payload_and_root(
        source_repo_root=source_repo_root,
        project_root=project_root,
        registry_path=registry_path,
    )
    if payload is None:
        return []
    bundles = payload.get("bundles")
    if not isinstance(bundles, list):
        return []
    return [dict(entry) for entry in bundles if isinstance(entry, dict)]


def get_bundle_entry(
    bundle_id: str,
    *,
    source_repo_root: Path | None = None,
    project_root: Path | None = None,
    registry_path: Path | None = None,
) -> dict[str, Any] | None:
    target = str(bundle_id)
    for entry in list_config_bundles(
        source_repo_root=source_repo_root,
        project_root=project_root,
        registry_path=registry_path,
    ):
        if str(entry.get("bundle_id", "")) == target:
            return dict(entry)
    return None


def validate_config_bundle(
    *,
    protocol_path: str | Path | None = None,
    comparison_path: str | Path | None = None,
    target_path: str | Path | None = None,
    source_repo_root: Path | None = None,
    project_root: Path | None = None,
    registry_path: Path | None = None,
) -> dict[str, Any]:
    protocol_identity = (
        describe_config_path(
            protocol_path,
            source_repo_root=source_repo_root,
            project_root=project_root,
            registry_path=registry_path,
        )
        if protocol_path is not None
        else None
    )
    comparison_identity = (
        describe_config_path(
            comparison_path,
            source_repo_root=source_repo_root,
            project_root=project_root,
            registry_path=registry_path,
        )
        if comparison_path is not None
        else None
    )
    target_identity = (
        describe_config_path(
            target_path,
            source_repo_root=source_repo_root,
            project_root=project_root,
            registry_path=registry_path,
        )
        if target_path is not None
        else None
    )

    identities = {
        "protocol": protocol_identity,
        "comparison": comparison_identity,
        "target": target_identity,
    }
    provided_components = {
        key: value for key, value in identities.items() if value is not None
    }
    errors: list[str] = []
    if not provided_components:
        return {
            "checked": False,
            "valid": False,
            "matched_bundle_ids": [],
            "matched_bundle_id": None,
            "protocol": protocol_identity,
            "comparison": comparison_identity,
            "target": target_identity,
            "errors": ["no bundle components provided"],
        }

    for component_name, identity in provided_components.items():
        if not bool(identity.get("registered", False)):
            errors.append(f"unregistered {component_name} component")

    if errors:
        return {
            "checked": True,
            "valid": False,
            "matched_bundle_ids": [],
            "matched_bundle_id": None,
            "protocol": protocol_identity,
            "comparison": comparison_identity,
            "target": target_identity,
            "errors": errors,
        }

    bundles = list_config_bundles(
        source_repo_root=source_repo_root,
        project_root=project_root,
        registry_path=registry_path,
    )
    protocol_id = (
        str(protocol_identity.get("config_id", "")) if isinstance(protocol_identity, dict) else ""
    )
    comparison_id = (
        str(comparison_identity.get("config_id", ""))
        if isinstance(comparison_identity, dict)
        else ""
    )
    target_id = (
        str(target_identity.get("config_id", "")) if isinstance(target_identity, dict) else ""
    )

    matched_bundle_ids: list[str] = []
    for bundle in bundles:
        bundle_id = str(bundle.get("bundle_id", "")).strip()
        if not bundle_id:
            continue
        if protocol_identity is not None and str(bundle.get("protocol", "")) != protocol_id:
            continue
        if comparison_identity is not None and str(bundle.get("comparison", "")) != comparison_id:
            continue
        if target_identity is not None and str(bundle.get("target", "")) != target_id:
            continue
        matched_bundle_ids.append(bundle_id)

    matched_bundle_ids = sorted(set(matched_bundle_ids))
    if not matched_bundle_ids:
        errors.append("provided components do not match any registered bundle")
        return {
            "checked": True,
            "valid": False,
            "matched_bundle_ids": [],
            "matched_bundle_id": None,
            "protocol": protocol_identity,
            "comparison": comparison_identity,
            "target": target_identity,
            "errors": errors,
        }

    return {
        "checked": True,
        "valid": True,
        "matched_bundle_ids": matched_bundle_ids,
        "matched_bundle_id": (matched_bundle_ids[0] if len(matched_bundle_ids) == 1 else None),
        "protocol": protocol_identity,
        "comparison": comparison_identity,
        "target": target_identity,
        "errors": [],
    }


__all__ = [
    "aliases_for_config_id",
    "describe_config_path",
    "get_config_entry",
    "get_bundle_entry",
    "list_config_bundles",
    "load_config_registry",
    "resolve_config_alias",
    "resolve_config_id",
    "resolve_config_registry_path",
    "validate_config_bundle",
]
