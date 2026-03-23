from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut, ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from Thesis_ML.config.framework_mode import FrameworkMode
from Thesis_ML.experiments.compute_policy import ResolvedComputePolicy
from Thesis_ML.experiments.stage_execution import StageKey
from Thesis_ML.experiments.stage_registry import (
    MODEL_FIT_TORCH_LOGREG_EXECUTOR_ID,
    SPECIALIZED_LINEARSVC_TUNING_EXECUTOR_ID,
    SPECIALIZED_LOGREG_TUNING_EXECUTOR_ID,
    TUNING_GENERIC_EXECUTOR_ID,
    TUNING_SKIPPED_CONTROL_EXECUTOR_ID,
    StageExecutorSelectionContext,
    get_stage_executor,
    iter_stage_executors,
    run_tuning_executor,
    stage_executor_support_status,
)


def _torch_compute_policy() -> ResolvedComputePolicy:
    return ResolvedComputePolicy(
        hardware_mode_requested="gpu_only",
        hardware_mode_effective="gpu_only",
        requested_backend_family="torch_gpu",
        effective_backend_family="torch_gpu",
        gpu_device_id=0,
        gpu_device_name="synthetic_gpu",
        gpu_device_total_memory_mb=4096,
        deterministic_compute=False,
        allow_backend_fallback=False,
        backend_stack_id="torch_gpu_reference_v1",
        backend_fallback_used=False,
        backend_fallback_reason=None,
    )


def _linearsvc_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("model", LinearSVC(dual=True, max_iter=5000, random_state=19)),
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


def test_stage_registry_lists_known_tuning_executors() -> None:
    tuning_executors = {spec.executor_id for spec in iter_stage_executors(StageKey.TUNING)}
    assert SPECIALIZED_LINEARSVC_TUNING_EXECUTOR_ID in tuning_executors
    assert SPECIALIZED_LOGREG_TUNING_EXECUTOR_ID in tuning_executors
    assert TUNING_GENERIC_EXECUTOR_ID in tuning_executors
    assert TUNING_SKIPPED_CONTROL_EXECUTOR_ID in tuning_executors


def test_stage_registry_blocks_non_admitted_executor_on_official_path() -> None:
    context = StageExecutorSelectionContext(
        stage=StageKey.MODEL_FIT,
        model_name="logreg",
        framework_mode=FrameworkMode.CONFIRMATORY,
        compute_policy=_torch_compute_policy(),
    )
    torch_logreg = get_stage_executor(MODEL_FIT_TORCH_LOGREG_EXECUTOR_ID)
    supported, reason = stage_executor_support_status(torch_logreg, context)
    assert supported is False
    assert reason == "executor_not_admitted_for_official_path"


def test_stage_registry_tuning_executor_falls_back_from_specialized_to_generic() -> None:
    x_matrix, y, groups = _grouped_binary_dataset()
    pipeline = _linearsvc_pipeline()
    inner_splits = list(LeaveOneGroupOut().split(x_matrix, y, groups))
    param_grid = {
        "model__C": [0.1, 1.0],
        "model__tol": [1e-4],
    }
    candidate_params = list(ParameterGrid(param_grid))

    payload = run_tuning_executor(
        executor_id=SPECIALIZED_LINEARSVC_TUNING_EXECUTOR_ID,
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

    assert payload["tuning_executor"] == TUNING_GENERIC_EXECUTOR_ID
    assert payload["specialized_linearsvc_tuning_used"] is False
    assert isinstance(payload["tuning_executor_fallback_reason"], str)


def test_stage_registry_logreg_specialized_falls_back_to_generic_when_unsupported() -> None:
    x_matrix, y, groups = _grouped_binary_dataset()
    pipeline = Pipeline(
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
    inner_splits = list(LeaveOneGroupOut().split(x_matrix, y, groups))
    param_grid = {"model__C": [0.1, 1.0], "model__penalty": ["l1"]}
    candidate_params = list(ParameterGrid(param_grid))

    payload = run_tuning_executor(
        executor_id=SPECIALIZED_LOGREG_TUNING_EXECUTOR_ID,
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

    assert payload["tuning_executor"] == TUNING_GENERIC_EXECUTOR_ID
    assert payload["specialized_logreg_tuning_used"] is False
    assert isinstance(payload["tuning_executor_fallback_reason"], str)
