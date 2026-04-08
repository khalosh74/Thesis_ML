from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from Thesis_ML.data.target_mapping_registry import canonical_mapping_object_hash


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path | str) -> str:
    resolved = Path(path)
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def stable_json_sha256(payload: Any) -> str:
    normalized = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return sha256_bytes(normalized.encode("utf-8"))


def combined_release_hash(component_hashes: dict[str, str]) -> str:
    keys = sorted(component_hashes)
    normalized = {key: str(component_hashes[key]) for key in keys}
    return stable_json_sha256(normalized)


def canonical_target_mapping_hash(path: Path | str) -> str:
    resolved = Path(path).resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Target mapping JSON must be an object: '{resolved}'.")

    normalized: dict[str, str] = {}
    for raw_key, raw_value in payload.items():
        key = str(raw_key).strip().lower()
        value = str(raw_value).strip().lower()
        if not key:
            raise ValueError(f"Target mapping '{resolved}' contains an empty source-label key.")
        if key in normalized:
            raise ValueError(
                f"Target mapping '{resolved}' defines duplicate source-label key '{key}'."
            )
        normalized[key] = value

    if not normalized:
        raise ValueError(f"Target mapping '{resolved}' must not be empty.")
    return canonical_mapping_object_hash(normalized)


__all__ = [
    "canonical_target_mapping_hash",
    "combined_release_hash",
    "sha256_bytes",
    "sha256_file",
    "stable_json_sha256",
]
