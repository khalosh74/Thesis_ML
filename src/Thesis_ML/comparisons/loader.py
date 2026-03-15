from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from Thesis_ML.comparisons.models import (
    SUPPORTED_COMPARISON_SCHEMA_VERSIONS,
    ComparisonSpec,
)


def load_comparison_spec(comparison_path: Path | str) -> ComparisonSpec:
    resolved_path = Path(comparison_path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"Comparison spec file was not found: {resolved_path}")

    try:
        payload = json.loads(resolved_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Comparison spec JSON parsing failed for '{resolved_path}': {exc}"
        ) from exc

    if not isinstance(payload, dict):
        raise ValueError(f"Comparison spec file '{resolved_path}' must contain a JSON object.")

    schema_version = payload.get("comparison_schema_version")
    if schema_version not in SUPPORTED_COMPARISON_SCHEMA_VERSIONS:
        allowed = ", ".join(sorted(SUPPORTED_COMPARISON_SCHEMA_VERSIONS))
        raise ValueError(
            f"Unsupported comparison_schema_version '{schema_version}'. Allowed values: {allowed}."
        )

    try:
        return ComparisonSpec.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(f"Comparison spec validation failed for '{resolved_path}': {exc}") from exc
