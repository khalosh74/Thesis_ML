from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from Thesis_ML.config.schema_versions import SUPPORTED_THESIS_PROTOCOL_SCHEMA_VERSIONS
from Thesis_ML.protocols.models import ThesisProtocol


def load_protocol(protocol_path: Path | str) -> ThesisProtocol:
    resolved_path = Path(protocol_path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"Protocol file was not found: {resolved_path}")

    try:
        payload = json.loads(resolved_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Protocol JSON parsing failed for '{resolved_path}': {exc}") from exc

    if not isinstance(payload, dict):
        raise ValueError(f"Protocol file '{resolved_path}' must contain a JSON object.")

    schema_version = payload.get("protocol_schema_version")
    if schema_version not in SUPPORTED_THESIS_PROTOCOL_SCHEMA_VERSIONS:
        allowed = ", ".join(sorted(SUPPORTED_THESIS_PROTOCOL_SCHEMA_VERSIONS))
        raise ValueError(
            "Unsupported protocol_schema_version "
            f"'{schema_version}'. Allowed values: {allowed}."
        )

    try:
        return ThesisProtocol.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(f"Protocol validation failed for '{resolved_path}': {exc}") from exc

