from __future__ import annotations

from enum import StrEnum


class FrameworkMode(StrEnum):
    EXPLORATORY = "exploratory"
    LOCKED_COMPARISON = "locked_comparison"
    CONFIRMATORY = "confirmatory"


def coerce_framework_mode(value: FrameworkMode | str) -> FrameworkMode:
    if isinstance(value, FrameworkMode):
        return value
    normalized = str(value).strip()
    try:
        return FrameworkMode(normalized)
    except ValueError as exc:
        allowed = ", ".join(mode.value for mode in FrameworkMode)
        raise ValueError(f"Unsupported framework_mode '{value}'. Allowed values: {allowed}.") from exc


__all__ = ["FrameworkMode", "coerce_framework_mode"]

