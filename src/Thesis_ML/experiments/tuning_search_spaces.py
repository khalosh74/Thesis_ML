from __future__ import annotations

from typing import Any

LINEAR_GROUPED_NESTED_SEARCH_SPACE_ID = "linear-grouped-nested-v1"
LINEAR_GROUPED_NESTED_SEARCH_SPACE_VERSION = "1.0.0"

_SEARCH_SPACES: dict[str, dict[str, Any]] = {
    LINEAR_GROUPED_NESTED_SEARCH_SPACE_ID: {
        "search_space_id": LINEAR_GROUPED_NESTED_SEARCH_SPACE_ID,
        "search_space_version": LINEAR_GROUPED_NESTED_SEARCH_SPACE_VERSION,
        "description": (
            "Conservative grouped nested tuning space for linear baseline models. "
            "Applied on training folds only."
        ),
        "param_grids": {
            "ridge": {"model__alpha": [0.1, 1.0, 10.0]},
            "logreg": {"model__C": [0.1, 1.0, 10.0], "model__penalty": ["l2"]},
            "linearsvc": {"model__C": [0.1, 1.0, 10.0]},
            "xgboost": {
                "model__n_estimators": [200],
                "model__max_depth": [3, 6],
                "model__learning_rate": [0.05, 0.1],
            },
        },
    }
}


def get_search_space(search_space_id: str, model_name: str) -> tuple[str, dict[str, list[Any]]]:
    search_space = _SEARCH_SPACES.get(str(search_space_id))
    if search_space is None:
        known = ", ".join(sorted(_SEARCH_SPACES))
        raise ValueError(
            f"Unknown tuning_search_space_id '{search_space_id}'. Known values: {known}."
        )
    param_grids = search_space.get("param_grids", {})
    model_grid = param_grids.get(str(model_name))
    if not isinstance(model_grid, dict) or not model_grid:
        raise ValueError(
            f"Search space '{search_space_id}' does not define a param grid for model '{model_name}'."
        )
    return str(search_space["search_space_version"]), dict(model_grid)


def known_search_space_ids() -> list[str]:
    return sorted(_SEARCH_SPACES)
