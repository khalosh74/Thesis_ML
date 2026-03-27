from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from Thesis_ML.config.metric_policy import metric_scorer
from Thesis_ML.experiments.logreg_tuning import (
    SPECIALIZED_LOGREG_TUNING_EXECUTOR_ID,
    is_specialized_logreg_grouped_nested_supported,
    run_specialized_logreg_grouped_nested_tuning,
)
from Thesis_ML.experiments.progress import ProgressEvent
from Thesis_ML.features.preprocessing import (
    SAMPLE_CENTER_STANDARD_SCALER_RECIPE_ID,
    build_feature_preprocessing_recipe,
)


def _logreg_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            (
                "model",
                LogisticRegression(
                    solver="saga",
                    penalty="l2",
                    max_iter=5000,
                    random_state=19,
                ),
            ),
        ]
    )


def _logreg_pipeline_with_nonbaseline_preprocessor() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "scaler",
                build_feature_preprocessing_recipe(SAMPLE_CENTER_STANDARD_SCALER_RECIPE_ID),
            ),
            (
                "model",
                LogisticRegression(
                    solver="saga",
                    penalty="l2",
                    max_iter=5000,
                    random_state=19,
                ),
            ),
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


def test_specialized_logreg_support_detection() -> None:
    supported, reason = is_specialized_logreg_grouped_nested_supported(
        model_name="logreg",
        pipeline_template=_logreg_pipeline(),
        param_grid={"model__C": [0.1, 1.0, 10.0], "model__penalty": ["l2"]},
    )
    assert supported is True
    assert reason is None


def test_specialized_logreg_rejects_unsupported_param_grid() -> None:
    supported, reason = is_specialized_logreg_grouped_nested_supported(
        model_name="logreg",
        pipeline_template=_logreg_pipeline(),
        param_grid={"model__C": [0.1, 1.0], "model__penalty": ["l1"]},
    )
    assert supported is False
    assert isinstance(reason, str)


def test_specialized_logreg_rejects_nonbaseline_preprocessor() -> None:
    supported, reason = is_specialized_logreg_grouped_nested_supported(
        model_name="logreg",
        pipeline_template=_logreg_pipeline_with_nonbaseline_preprocessor(),
        param_grid={"model__C": [0.1, 1.0], "model__penalty": ["l2"]},
    )
    assert supported is False
    assert reason == "preprocessor_not_plain_standard_scaler"


def test_specialized_logreg_matches_gridsearchcv_parity() -> None:
    x_matrix, y, groups = _grouped_binary_dataset()
    pipeline_template = _logreg_pipeline()
    param_grid = {"model__C": [0.1, 1.0, 10.0], "model__penalty": ["l2"]}

    specialized = run_specialized_logreg_grouped_nested_tuning(
        pipeline_template=pipeline_template,
        x_train=x_matrix,
        y_train=y,
        inner_groups=groups,
        param_grid=param_grid,
        primary_metric_name="balanced_accuracy",
    )

    search = GridSearchCV(
        estimator=_logreg_pipeline(),
        param_grid=param_grid,
        scoring=metric_scorer("balanced_accuracy"),
        cv=LeaveOneGroupOut(),
        refit=True,
        n_jobs=1,
    )
    search.fit(x_matrix, y, groups=groups)

    assert specialized.executor_id == SPECIALIZED_LOGREG_TUNING_EXECUTOR_ID
    assert specialized.best_params == search.best_params_
    assert specialized.best_score == float(search.best_score_)
    np.testing.assert_allclose(
        specialized.cv_results["mean_test_score"],
        np.asarray(search.cv_results_["mean_test_score"], dtype=np.float64),
        atol=1e-12,
        rtol=1e-12,
    )
    specialized_pred = specialized.best_estimator.predict(x_matrix)
    search_pred = search.best_estimator_.predict(x_matrix)
    np.testing.assert_array_equal(specialized_pred, search_pred)


def test_specialized_logreg_profile_caps_and_progress_telemetry() -> None:
    x_matrix, y, groups = _grouped_binary_dataset()
    param_grid = {"model__C": [0.01, 0.1, 1.0, 10.0], "model__penalty": ["l2"]}
    events: list[ProgressEvent] = []

    result = run_specialized_logreg_grouped_nested_tuning(
        pipeline_template=_logreg_pipeline(),
        x_train=x_matrix,
        y_train=y,
        inner_groups=groups,
        param_grid=param_grid,
        primary_metric_name="balanced_accuracy",
        profile_inner_folds=2,
        profile_tuning_candidates=2,
        progress_callback=events.append,
        progress_metadata={"run_id": "logreg-specialized-test"},
    )

    assert result.configured_inner_fold_count == 6
    assert result.profiled_inner_fold_count == 2
    assert result.configured_candidate_count == 4
    assert result.profiled_candidate_count == 2
    assert result.tuning_extrapolation_applied is True
    assert result.estimated_full_inner_tuning_seconds >= result.measured_inner_tuning_seconds
    assert result.estimated_full_tuned_search_seconds >= result.tuned_search_elapsed_seconds
    assert result.progress_total_units == 4
    assert result.progress_event_count == len(events)
    assert len(events) >= 6
    assert any("candidate" in str(event.message) for event in events)
    assert events[-1].completed_units == float(result.progress_total_units)
