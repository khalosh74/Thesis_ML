from __future__ import annotations

from sklearn.model_selection import ParameterGrid

from Thesis_ML.experiments.tuning_search_spaces import (
    LINEAR_GROUPED_NESTED_SEARCH_SPACE_ID,
    get_search_space,
)


def test_xgboost_search_space_entry_is_explicit_and_nonempty() -> None:
    version, grid = get_search_space(LINEAR_GROUPED_NESTED_SEARCH_SPACE_ID, "xgboost")
    assert version == "1.0.0"
    assert grid
    assert set(grid) == {
        "model__n_estimators",
        "model__max_depth",
        "model__learning_rate",
    }
    assert all(len(values) > 0 for values in grid.values())


def test_xgboost_search_space_grid_is_small_and_deterministic() -> None:
    _, grid = get_search_space(LINEAR_GROUPED_NESTED_SEARCH_SPACE_ID, "xgboost")
    candidates = list(ParameterGrid(grid))
    assert len(candidates) == 4
