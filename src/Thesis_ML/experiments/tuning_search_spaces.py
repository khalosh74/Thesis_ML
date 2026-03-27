from __future__ import annotations

from typing import Any

# Official grouped-nested tuning surface for the current thesis-facing milestone.
OFFICIAL_LINEAR_GROUPED_NESTED_SEARCH_SPACE_ID = "official-linear-grouped-nested-v2"
OFFICIAL_LINEAR_GROUPED_NESTED_SEARCH_SPACE_VERSION = "2.0.0"

# Legacy grouped-nested space kept loadable for reproducibility/backward compatibility.
LINEAR_GROUPED_NESTED_SEARCH_SPACE_ID = "linear-grouped-nested-v1"
LINEAR_GROUPED_NESTED_SEARCH_SPACE_VERSION = "1.0.0"

# Exploratory-only extension surface for non-linear model family.
EXPLORATORY_XGBOOST_GROUPED_NESTED_SEARCH_SPACE_ID = "exploratory-xgboost-grouped-nested-v1"
EXPLORATORY_XGBOOST_GROUPED_NESTED_SEARCH_SPACE_VERSION = "1.0.0"

OFFICIAL_TUNING_SEARCH_SPACE_IDS: tuple[str, ...] = (
    OFFICIAL_LINEAR_GROUPED_NESTED_SEARCH_SPACE_ID,
)

_SEARCH_SPACES: dict[str, dict[str, Any]] = {
    OFFICIAL_LINEAR_GROUPED_NESTED_SEARCH_SPACE_ID: {
        "search_space_id": OFFICIAL_LINEAR_GROUPED_NESTED_SEARCH_SPACE_ID,
        "search_space_version": OFFICIAL_LINEAR_GROUPED_NESTED_SEARCH_SPACE_VERSION,
        "description": (
            "Official grouped nested tuning space for thesis-facing linear models. "
            "Applied on training folds only."
        ),
        "official_admitted": True,
        "param_grids": {
            "ridge": {"model__alpha": [0.1, 1.0, 10.0]},
            "logreg": {"model__C": [0.1, 1.0, 10.0], "model__penalty": ["l2"]},
            "linearsvc": {"model__C": [0.1, 1.0, 10.0]},
        },
    },
    LINEAR_GROUPED_NESTED_SEARCH_SPACE_ID: {
        "search_space_id": LINEAR_GROUPED_NESTED_SEARCH_SPACE_ID,
        "search_space_version": LINEAR_GROUPED_NESTED_SEARCH_SPACE_VERSION,
        "description": (
            "Legacy grouped nested tuning space retained for backward compatibility. "
            "This surface is not admitted as the official thesis-facing tuning contract."
        ),
        "official_admitted": False,
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
    },
    EXPLORATORY_XGBOOST_GROUPED_NESTED_SEARCH_SPACE_ID: {
        "search_space_id": EXPLORATORY_XGBOOST_GROUPED_NESTED_SEARCH_SPACE_ID,
        "search_space_version": EXPLORATORY_XGBOOST_GROUPED_NESTED_SEARCH_SPACE_VERSION,
        "description": (
            "Exploratory grouped nested tuning space for xgboost and future non-linear extensions."
        ),
        "official_admitted": False,
        "param_grids": {
            "xgboost": {
                "model__n_estimators": [200],
                "model__max_depth": [3, 6],
                "model__learning_rate": [0.05, 0.1],
            },
        },
    },
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


def search_space_is_official(
    search_space_id: str,
    *,
    search_space_version: str | None = None,
) -> bool:
    search_space = _SEARCH_SPACES.get(str(search_space_id))
    if not isinstance(search_space, dict):
        return False
    if not bool(search_space.get("official_admitted", False)):
        return False
    if search_space_version is None:
        return True
    return str(search_space.get("search_space_version")) == str(search_space_version)


def known_search_space_ids() -> list[str]:
    return sorted(_SEARCH_SPACES)
