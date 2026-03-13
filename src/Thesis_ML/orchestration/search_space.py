from __future__ import annotations

import itertools
from typing import Any

from Thesis_ML.orchestration.contracts import SearchMode, SearchSpaceSpec

_SEGMENT_KEYS = {"start_section", "end_section", "base_artifact_id", "reuse_policy"}


def build_search_space_map(search_spaces: list[SearchSpaceSpec]) -> dict[str, SearchSpaceSpec]:
    return {space.search_space_id: space for space in search_spaces if bool(space.enabled)}


def _apply_assignment(base_variant: dict[str, Any], assignment: dict[str, Any]) -> dict[str, Any]:
    updated = dict(base_variant)
    params = dict(updated.get("params", {}))
    for key, value in assignment.items():
        if key in _SEGMENT_KEYS:
            updated[key] = value
        else:
            params[key] = value
    updated["params"] = params
    updated["search_assignment"] = assignment
    return updated


def _expand_deterministic(
    base_variant: dict[str, Any], search_space: SearchSpaceSpec
) -> list[dict[str, Any]]:
    dimensions = list(search_space.dimensions)
    if not dimensions:
        return [dict(base_variant)]
    names = [dim.parameter_name for dim in dimensions]
    value_grid = [list(dim.values) for dim in dimensions]
    expanded: list[dict[str, Any]] = []
    for combo in itertools.product(*value_grid):
        assignment = {name: value for name, value in zip(names, combo, strict=True)}
        expanded.append(_apply_assignment(base_variant, assignment))
    return expanded


def _expand_optuna(
    base_variant: dict[str, Any],
    *,
    search_space: SearchSpaceSpec,
    seed: int,
    optuna_trials: int | None,
) -> list[dict[str, Any]]:
    try:
        import optuna
    except Exception as exc:  # pragma: no cover - environment-dependent
        raise ValueError(
            "Search space requires optimization_mode='optuna' but 'optuna' is not installed."
        ) from exc

    n_trials = optuna_trials or search_space.max_trials or 10
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    expanded: list[dict[str, Any]] = []
    seen: set[tuple[tuple[str, str], ...]] = set()

    for _ in range(int(n_trials)):
        trial = study.ask()
        assignment: dict[str, Any] = {}
        for dimension in search_space.dimensions:
            assignment[dimension.parameter_name] = trial.suggest_categorical(
                dimension.parameter_name,
                list(dimension.values),
            )
        fingerprint = tuple(sorted((key, str(value)) for key, value in assignment.items()))
        if fingerprint in seen:
            study.tell(trial, 0.0)
            continue
        seen.add(fingerprint)
        expanded.append(_apply_assignment(base_variant, assignment))
        study.tell(trial, 0.0)

    if not expanded:
        return [dict(base_variant)]
    return expanded


def expand_variant_search_space(
    base_variant: dict[str, Any],
    *,
    search_space: SearchSpaceSpec,
    seed: int,
    optuna_enabled: bool,
    optuna_trials: int | None = None,
) -> list[dict[str, Any]]:
    mode = SearchMode(search_space.optimization_mode)
    if mode == SearchMode.DETERMINISTIC_GRID:
        return _expand_deterministic(base_variant, search_space)
    if not optuna_enabled:
        raise ValueError(
            "Encountered optimization_mode='optuna' but optuna mode is disabled. "
            "Enable --search-mode optuna to allow this search space."
        )
    return _expand_optuna(
        base_variant,
        search_space=search_space,
        seed=seed,
        optuna_trials=optuna_trials,
    )
