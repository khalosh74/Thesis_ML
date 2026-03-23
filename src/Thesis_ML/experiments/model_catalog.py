from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from Thesis_ML.config.framework_mode import FrameworkMode, coerce_framework_mode
from Thesis_ML.config.methodology import MethodologyPolicyName
from Thesis_ML.experiments.model_factory import ALL_MODEL_NAMES


class ModelCostTier(StrEnum):
    OFFICIAL_FAST = "official_fast"
    OFFICIAL_ALLOWED = "official_allowed"
    BENCHMARK_EXPENSIVE = "benchmark_expensive"
    EXPLORATORY_ONLY = "exploratory_only"


@dataclass(frozen=True)
class ModelCostEntry:
    model_name: str
    cost_tier: ModelCostTier
    projected_runtime_seconds_by_mode: dict[str, int]
    grouped_nested_runtime_multiplier: float = 1.0
    timeout_override_seconds: int | None = None
    requires_explicit_comparison_spec: bool = False
    notes: str | None = None


def _validate_runtime_map(
    model_name: str,
    runtime_map: dict[str, int],
) -> dict[str, int]:
    required_modes = {
        FrameworkMode.EXPLORATORY.value,
        FrameworkMode.LOCKED_COMPARISON.value,
        FrameworkMode.CONFIRMATORY.value,
    }
    missing_modes = sorted(required_modes - set(runtime_map.keys()))
    if missing_modes:
        raise ValueError(
            f"Model '{model_name}' runtime map is missing modes: {', '.join(missing_modes)}."
        )
    resolved: dict[str, int] = {}
    for key, value in runtime_map.items():
        seconds = int(value)
        if seconds <= 0:
            raise ValueError(
                f"Model '{model_name}' projected runtime for mode '{key}' must be > 0."
            )
        resolved[str(key)] = seconds
    return resolved


def _build_catalog() -> dict[str, ModelCostEntry]:
    entries = [
        ModelCostEntry(
            model_name="ridge",
            cost_tier=ModelCostTier.OFFICIAL_FAST,
            projected_runtime_seconds_by_mode=_validate_runtime_map(
                "ridge",
                {
                    FrameworkMode.EXPLORATORY.value: 15 * 60,
                    FrameworkMode.LOCKED_COMPARISON.value: 25 * 60,
                    FrameworkMode.CONFIRMATORY.value: 20 * 60,
                },
            ),
            grouped_nested_runtime_multiplier=1.25,
            notes="Linear baseline with stable runtime envelope.",
        ),
        ModelCostEntry(
            model_name="linearsvc",
            cost_tier=ModelCostTier.OFFICIAL_ALLOWED,
            projected_runtime_seconds_by_mode=_validate_runtime_map(
                "linearsvc",
                {
                    FrameworkMode.EXPLORATORY.value: 20 * 60,
                    FrameworkMode.LOCKED_COMPARISON.value: 35 * 60,
                    FrameworkMode.CONFIRMATORY.value: 30 * 60,
                },
            ),
            grouped_nested_runtime_multiplier=1.25,
            notes="Allowed official model with moderate runtime.",
        ),
        ModelCostEntry(
            model_name="logreg",
            cost_tier=ModelCostTier.BENCHMARK_EXPENSIVE,
            projected_runtime_seconds_by_mode=_validate_runtime_map(
                "logreg",
                {
                    FrameworkMode.EXPLORATORY.value: 35 * 60,
                    FrameworkMode.LOCKED_COMPARISON.value: 70 * 60,
                    FrameworkMode.CONFIRMATORY.value: 55 * 60,
                },
            ),
            grouped_nested_runtime_multiplier=1.25,
            timeout_override_seconds=120 * 60,
            requires_explicit_comparison_spec=True,
            notes="Expensive benchmark model that must be explicitly declared in comparison specs.",
        ),
        ModelCostEntry(
            model_name="dummy",
            cost_tier=ModelCostTier.OFFICIAL_FAST,
            projected_runtime_seconds_by_mode=_validate_runtime_map(
                "dummy",
                {
                    FrameworkMode.EXPLORATORY.value: 60,
                    FrameworkMode.LOCKED_COMPARISON.value: 90,
                    FrameworkMode.CONFIRMATORY.value: 90,
                },
            ),
            grouped_nested_runtime_multiplier=1.0,
            notes="Control baseline with negligible runtime.",
        ),
        ModelCostEntry(
            model_name="xgboost",
            cost_tier=ModelCostTier.EXPLORATORY_ONLY,
            projected_runtime_seconds_by_mode=_validate_runtime_map(
                "xgboost",
                {
                    FrameworkMode.EXPLORATORY.value: 45 * 60,
                    FrameworkMode.LOCKED_COMPARISON.value: 90 * 60,
                    FrameworkMode.CONFIRMATORY.value: 90 * 60,
                },
            ),
            grouped_nested_runtime_multiplier=1.25,
            notes=(
                "Exploratory-only gradient boosting family. "
                "Not admitted on locked-comparison or confirmatory official paths."
            ),
        ),
    ]

    catalog = {entry.model_name: entry for entry in entries}
    unsupported = sorted(set(ALL_MODEL_NAMES) - set(catalog.keys()))
    if unsupported:
        raise ValueError(
            "Model cost catalog is missing supported models: " + ", ".join(unsupported)
        )
    return catalog


_MODEL_CATALOG = _build_catalog()


def supported_model_cost_tiers() -> tuple[ModelCostTier, ...]:
    return (
        ModelCostTier.OFFICIAL_FAST,
        ModelCostTier.OFFICIAL_ALLOWED,
        ModelCostTier.BENCHMARK_EXPENSIVE,
        ModelCostTier.EXPLORATORY_ONLY,
    )


def get_model_cost_entry(model_name: str) -> ModelCostEntry:
    key = str(model_name).strip().lower()
    entry = _MODEL_CATALOG.get(key)
    if entry is None:
        allowed = ", ".join(sorted(_MODEL_CATALOG.keys()))
        raise ValueError(f"Unsupported model '{model_name}'. Allowed values: {allowed}.")
    return entry


def _is_grouped_nested_context(
    *,
    methodology_policy: str | MethodologyPolicyName | None,
    tuning_enabled: bool | None,
) -> bool:
    if tuning_enabled is True:
        return True
    if methodology_policy is None:
        return False
    normalized = str(methodology_policy).strip()
    return normalized == MethodologyPolicyName.GROUPED_NESTED_TUNING.value


def projected_runtime_seconds(
    model_name: str,
    framework_mode: FrameworkMode | str,
    methodology_policy: str | MethodologyPolicyName | None = None,
    *,
    tuning_enabled: bool | None = None,
) -> int:
    entry = get_model_cost_entry(model_name)
    mode = coerce_framework_mode(framework_mode).value
    base_seconds = int(entry.projected_runtime_seconds_by_mode[mode])
    if _is_grouped_nested_context(
        methodology_policy=methodology_policy,
        tuning_enabled=tuning_enabled,
    ):
        base_seconds = int(round(base_seconds * float(entry.grouped_nested_runtime_multiplier)))
    return max(base_seconds, 1)


def model_timeout_overrides_seconds() -> dict[str, int]:
    overrides: dict[str, int] = {}
    for model_name, entry in _MODEL_CATALOG.items():
        if entry.timeout_override_seconds is None:
            continue
        overrides[model_name] = int(entry.timeout_override_seconds)
    return overrides


def model_catalog_snapshot() -> dict[str, dict[str, Any]]:
    snapshot: dict[str, dict[str, Any]] = {}
    for model_name, entry in sorted(_MODEL_CATALOG.items()):
        snapshot[model_name] = {
            "model_name": entry.model_name,
            "cost_tier": entry.cost_tier.value,
            "projected_runtime_seconds_by_mode": dict(entry.projected_runtime_seconds_by_mode),
            "grouped_nested_runtime_multiplier": float(entry.grouped_nested_runtime_multiplier),
            "timeout_override_seconds": (
                int(entry.timeout_override_seconds)
                if entry.timeout_override_seconds is not None
                else None
            ),
            "requires_explicit_comparison_spec": bool(entry.requires_explicit_comparison_spec),
            "notes": entry.notes,
        }
    return snapshot


__all__ = [
    "ModelCostEntry",
    "ModelCostTier",
    "get_model_cost_entry",
    "model_catalog_snapshot",
    "model_timeout_overrides_seconds",
    "projected_runtime_seconds",
    "supported_model_cost_tiers",
]
