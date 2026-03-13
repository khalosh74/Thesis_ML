from __future__ import annotations

from typing import Any

from Thesis_ML.orchestration.campaign_runner import (
    _expand_experiment_variants,
    _expand_template_variants,
)


def expand_template_variants(
    experiment: dict[str, Any],
    template: dict[str, Any],
    dataset_scope: dict[str, Any],
    search_space_map: dict[str, Any] | None = None,
    search_seed: int = 42,
    optuna_enabled: bool = False,
    optuna_trials: int | None = None,
) -> list[dict[str, Any]]:
    return _expand_template_variants(
        experiment=experiment,
        template=template,
        dataset_scope=dataset_scope,
        search_space_map=search_space_map,
        search_seed=search_seed,
        optuna_enabled=optuna_enabled,
        optuna_trials=optuna_trials,
    )


def expand_experiment_variants(
    experiment: dict[str, Any],
    dataset_scope: dict[str, Any],
    search_space_map: dict[str, Any] | None = None,
    search_seed: int = 42,
    optuna_enabled: bool = False,
    optuna_trials: int | None = None,
    max_runs_per_experiment: int | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    return _expand_experiment_variants(
        experiment=experiment,
        dataset_scope=dataset_scope,
        search_space_map=search_space_map,
        search_seed=search_seed,
        optuna_enabled=optuna_enabled,
        optuna_trials=optuna_trials,
        max_runs_per_experiment=max_runs_per_experiment,
    )


__all__ = ["expand_template_variants", "expand_experiment_variants"]
