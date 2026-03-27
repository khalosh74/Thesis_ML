from __future__ import annotations

from typing import Any

from Thesis_ML.experiments.model_registry import (
    EXPLORATORY_XGBOOST_GROUPED_NESTED_SEARCH_SPACE_ID,
    EXPLORATORY_XGBOOST_GROUPED_NESTED_SEARCH_SPACE_VERSION,
    LINEAR_GROUPED_NESTED_SEARCH_SPACE_ID,
    LINEAR_GROUPED_NESTED_SEARCH_SPACE_VERSION,
    OFFICIAL_LINEAR_GROUPED_NESTED_SEARCH_SPACE_ID,
    OFFICIAL_LINEAR_GROUPED_NESTED_SEARCH_SPACE_VERSION,
    get_model_spec,
)

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


def search_space_allowed_for_model(
    *,
    model_name: str,
    search_space_id: str,
) -> bool:
    spec = get_model_spec(model_name)
    return str(search_space_id) in set(spec.tuning_policy.allowed_search_space_ids)


def resolve_tuning_search_space_for_model(
    *,
    model_name: str,
    search_space_id: str | None,
    search_space_version: str | None,
) -> tuple[str, str, dict[str, list[Any]]]:
    spec = get_model_spec(model_name)
    if not bool(spec.tuning_policy.supports_tuning):
        raise ValueError(f"Model '{spec.logical_name}' does not support grouped nested tuning.")

    resolved_search_space_id = (
        str(search_space_id)
        if search_space_id is not None and str(search_space_id).strip()
        else spec.tuning_policy.default_search_space_id
    )
    if not resolved_search_space_id:
        raise ValueError(
            f"grouped_nested_tuning requires tuning_search_space_id for model '{spec.logical_name}'."
        )

    allowed_search_spaces = set(spec.tuning_policy.allowed_search_space_ids)
    if resolved_search_space_id not in allowed_search_spaces:
        allowed = ", ".join(sorted(allowed_search_spaces))
        raise ValueError(
            f"Model '{spec.logical_name}' does not allow tuning_search_space_id "
            f"'{resolved_search_space_id}'. Allowed values: {allowed}."
        )

    resolved_search_space_version, param_grid = get_search_space(
        resolved_search_space_id,
        spec.logical_name,
    )
    if (
        search_space_version is not None
        and str(search_space_version).strip()
        and str(search_space_version) != str(resolved_search_space_version)
    ):
        raise ValueError(
            "Declared tuning_search_space_version does not match search-space registry version."
        )

    return resolved_search_space_id, resolved_search_space_version, dict(param_grid)


def known_search_space_ids() -> list[str]:
    return sorted(_SEARCH_SPACES)


__all__ = [
    "EXPLORATORY_XGBOOST_GROUPED_NESTED_SEARCH_SPACE_ID",
    "EXPLORATORY_XGBOOST_GROUPED_NESTED_SEARCH_SPACE_VERSION",
    "LINEAR_GROUPED_NESTED_SEARCH_SPACE_ID",
    "LINEAR_GROUPED_NESTED_SEARCH_SPACE_VERSION",
    "OFFICIAL_LINEAR_GROUPED_NESTED_SEARCH_SPACE_ID",
    "OFFICIAL_LINEAR_GROUPED_NESTED_SEARCH_SPACE_VERSION",
    "OFFICIAL_TUNING_SEARCH_SPACE_IDS",
    "get_search_space",
    "known_search_space_ids",
    "resolve_tuning_search_space_for_model",
    "search_space_allowed_for_model",
    "search_space_is_official",
]
