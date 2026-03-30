from __future__ import annotations

import numpy as np
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut, ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifier

from Thesis_ML.config.metric_policy import metric_scorer
from Thesis_ML.experiments.ridge_tuning import (
    SPECIALIZED_RIDGE_TUNING_EXECUTOR_ID,
    is_specialized_ridge_grouped_nested_supported,
    run_specialized_ridge_grouped_nested_tuning,
)
from Thesis_ML.experiments.stage_registry import TUNING_GENERIC_EXECUTOR_ID, run_tuning_executor
from Thesis_ML.verification.backend_parity import compare_tuning_parity


def _ridge_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("model", RidgeClassifier()),
        ]
    )


def _grouped_binary_dataset() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_matrix = np.asarray(
        [
            [0.0, 0.2, 0.1],
            [0.1, 0.1, 0.0],
            [1.0, 1.1, 1.0],
            [1.2, 1.0, 1.1],
            [2.0, 2.1, 2.0],
            [2.1, 2.2, 2.1],
            [3.0, 3.1, 3.0],
            [3.2, 3.0, 3.1],
            [4.0, 4.1, 4.0],
            [4.1, 4.2, 4.1],
            [5.0, 5.1, 5.0],
            [5.2, 5.0, 5.1],
        ],
        dtype=np.float64,
    )
    y = np.asarray(
        [
            "neg",
            "pos",
            "neg",
            "pos",
            "neg",
            "pos",
            "neg",
            "pos",
            "neg",
            "pos",
            "neg",
            "pos",
        ]
    )
    groups = np.asarray(
        [
            "g1",
            "g1",
            "g2",
            "g2",
            "g3",
            "g3",
            "g4",
            "g4",
            "g5",
            "g5",
            "g6",
            "g6",
        ]
    )
    return x_matrix, y, groups


def test_specialized_ridge_support_detection() -> None:
    supported, reason = is_specialized_ridge_grouped_nested_supported(
        model_name="ridge",
        pipeline_template=_ridge_pipeline(),
        param_grid={"model__alpha": [0.1, 1.0, 10.0]},
    )
    assert supported is True
    assert reason is None


def test_specialized_ridge_rejects_unsupported_param_grid() -> None:
    supported, reason = is_specialized_ridge_grouped_nested_supported(
        model_name="ridge",
        pipeline_template=_ridge_pipeline(),
        param_grid={"model__alpha": [0.1, 1.0], "model__fit_intercept": [True]},
    )
    assert supported is False
    assert isinstance(reason, str)


def test_specialized_ridge_matches_gridsearchcv_parity() -> None:
    x_matrix, y, groups = _grouped_binary_dataset()
    param_grid = {"model__alpha": [0.1, 1.0, 10.0]}

    specialized = run_specialized_ridge_grouped_nested_tuning(
        pipeline_template=_ridge_pipeline(),
        x_train=x_matrix,
        y_train=y,
        inner_groups=groups,
        param_grid=param_grid,
        primary_metric_name="balanced_accuracy",
    )

    search = GridSearchCV(
        estimator=_ridge_pipeline(),
        param_grid=param_grid,
        scoring=metric_scorer("balanced_accuracy"),
        cv=LeaveOneGroupOut(),
        refit=True,
        n_jobs=1,
    )
    search.fit(x_matrix, y, groups=groups)

    assert specialized.executor_id == SPECIALIZED_RIDGE_TUNING_EXECUTOR_ID
    assert specialized.best_params == search.best_params_
    assert specialized.best_score == float(search.best_score_)
    np.testing.assert_allclose(
        specialized.cv_results["mean_test_score"],
        np.asarray(search.cv_results_["mean_test_score"], dtype=np.float64),
        atol=1e-12,
        rtol=1e-12,
    )
    np.testing.assert_array_equal(
        specialized.best_estimator.predict(x_matrix),
        search.best_estimator_.predict(x_matrix),
    )

    parity = compare_tuning_parity(
        reference_payload={
            "best_params": search.best_params_,
            "best_score": float(search.best_score_),
            "cv_results": {
                "mean_test_score": np.asarray(
                    search.cv_results_["mean_test_score"], dtype=np.float64
                ).tolist()
            },
        },
        candidate_payload={
            "best_params": specialized.best_params,
            "best_score": float(specialized.best_score),
            "cv_results": {"mean_test_score": specialized.cv_results["mean_test_score"]},
        },
        category="exact",
    )
    assert parity.passed is True


def test_specialized_ridge_profile_caps_apply_with_extrapolation() -> None:
    x_matrix, y, groups = _grouped_binary_dataset()
    param_grid = {"model__alpha": [0.01, 0.1, 1.0, 10.0]}

    result = run_specialized_ridge_grouped_nested_tuning(
        pipeline_template=_ridge_pipeline(),
        x_train=x_matrix,
        y_train=y,
        inner_groups=groups,
        param_grid=param_grid,
        primary_metric_name="balanced_accuracy",
        profile_inner_folds=2,
        profile_tuning_candidates=2,
    )

    assert result.configured_inner_fold_count == 6
    assert result.profiled_inner_fold_count == 2
    assert result.configured_candidate_count == 4
    assert result.profiled_candidate_count == 2
    assert result.tuning_extrapolation_applied is True
    assert result.estimated_full_inner_tuning_seconds >= result.measured_inner_tuning_seconds
    assert result.estimated_full_tuned_search_seconds >= result.tuned_search_elapsed_seconds


def test_ridge_workflow_level_specialized_vs_generic_parity() -> None:
    x_matrix, y, groups = _grouped_binary_dataset()
    pipeline = _ridge_pipeline()
    inner_splits = list(LeaveOneGroupOut().split(x_matrix, y, groups))
    param_grid = {"model__alpha": [0.1, 1.0, 10.0]}
    candidate_params = list(ParameterGrid(param_grid))

    generic_payload = run_tuning_executor(
        executor_id=TUNING_GENERIC_EXECUTOR_ID,
        fallback_executor_id=None,
        pipeline_template=pipeline,
        x_outer_train=x_matrix,
        y_outer_train=y,
        inner_groups=groups,
        inner_splits=inner_splits,
        param_grid=param_grid,
        candidate_params=candidate_params,
        configured_candidate_count=len(candidate_params),
        configured_inner_fold_count=len(inner_splits),
        profiled_candidate_count=len(candidate_params),
        profiled_inner_fold_count=len(inner_splits),
        primary_metric_name="balanced_accuracy",
    )
    specialized_payload = run_tuning_executor(
        executor_id=SPECIALIZED_RIDGE_TUNING_EXECUTOR_ID,
        fallback_executor_id=TUNING_GENERIC_EXECUTOR_ID,
        pipeline_template=pipeline,
        x_outer_train=x_matrix,
        y_outer_train=y,
        inner_groups=groups,
        inner_splits=inner_splits,
        param_grid=param_grid,
        candidate_params=candidate_params,
        configured_candidate_count=len(candidate_params),
        configured_inner_fold_count=len(inner_splits),
        profiled_candidate_count=len(candidate_params),
        profiled_inner_fold_count=len(inner_splits),
        primary_metric_name="balanced_accuracy",
    )

    assert specialized_payload["tuning_executor"] == SPECIALIZED_RIDGE_TUNING_EXECUTOR_ID
    assert specialized_payload["specialized_ridge_tuning_used"] is True
    assert specialized_payload["tuned_search_candidate_count"] == generic_payload[
        "tuned_search_candidate_count"
    ]
    assert specialized_payload["tuned_search_profiled_inner_fold_count"] == generic_payload[
        "tuned_search_profiled_inner_fold_count"
    ]
    assert specialized_payload["tuned_search_configured_inner_fold_count"] == generic_payload[
        "tuned_search_configured_inner_fold_count"
    ]

    parity = compare_tuning_parity(
        reference_payload={
            "best_params_json": generic_payload["best_params_json"],
            "best_score": generic_payload["best_score"],
        },
        candidate_payload={
            "best_params_json": specialized_payload["best_params_json"],
            "best_score": specialized_payload["best_score"],
        },
        category="exact",
    )
    assert parity.passed is True
    np.testing.assert_array_equal(
        np.asarray(generic_payload["estimator"].predict(x_matrix)),
        np.asarray(specialized_payload["estimator"].predict(x_matrix)),
    )
