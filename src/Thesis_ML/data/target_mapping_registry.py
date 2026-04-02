from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from Thesis_ML.config.paths import DEFAULT_TARGET_CONFIGS_DIR

_ALLOWED_COARSE_AFFECT_OUTPUT_LABELS = frozenset({"positive", "neutral", "negative"})


@dataclass(frozen=True)
class TargetMappingEntry:
    mapping: Mapping[str, str]
    version: str
    mapping_hash: str
    path: Path


def canonical_mapping_object_hash(mapping: Mapping[str, str]) -> str:
    canonical = json.dumps(
        {str(key): str(value) for key, value in mapping.items()},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _normalize_mapping(mapping_payload: dict[str, Any], *, mapping_path: Path) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for raw_key, raw_value in mapping_payload.items():
        key = str(raw_key).strip().lower()
        value = str(raw_value).strip().lower()
        if not key:
            raise ValueError(f"Target mapping '{mapping_path}' has an empty source label key.")
        if not value:
            raise ValueError(
                f"Target mapping '{mapping_path}' has an empty output label for key='{key}'."
            )
        if value not in _ALLOWED_COARSE_AFFECT_OUTPUT_LABELS:
            allowed = ", ".join(sorted(_ALLOWED_COARSE_AFFECT_OUTPUT_LABELS))
            raise ValueError(
                "Target mapping contains unsupported output label "
                f"'{value}' for key='{key}'. Allowed labels: {allowed}."
            )
        if key in normalized:
            raise ValueError(
                f"Target mapping '{mapping_path}' defines duplicate source label key='{key}'."
            )
        normalized[key] = value
    if not normalized:
        raise ValueError(f"Target mapping '{mapping_path}' must not be empty.")
    return normalized


def load_target_mapping(
    mapping_version: str, *, target_configs_dir: Path = DEFAULT_TARGET_CONFIGS_DIR
) -> TargetMappingEntry:
    version = str(mapping_version).strip()
    if not version:
        raise ValueError("Target mapping version must be a non-empty string.")

    mapping_path = target_configs_dir / f"{version}.json"
    if not mapping_path.exists():
        raise FileNotFoundError(f"Target mapping file not found: {mapping_path}")

    try:
        payload = json.loads(mapping_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Target mapping JSON parsing failed for '{mapping_path}': {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Target mapping '{mapping_path}' must be a JSON object.")

    normalized = _normalize_mapping(payload, mapping_path=mapping_path)
    mapping_hash = canonical_mapping_object_hash(normalized)
    immutable_mapping = MappingProxyType(dict(sorted(normalized.items())))
    return TargetMappingEntry(
        mapping=immutable_mapping,
        version=version,
        mapping_hash=mapping_hash,
        path=mapping_path.resolve(),
    )


__all__ = [
    "TargetMappingEntry",
    "canonical_mapping_object_hash",
    "load_target_mapping",
]
