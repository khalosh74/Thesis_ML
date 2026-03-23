from __future__ import annotations

from Thesis_ML.config.framework_mode import FrameworkMode
from Thesis_ML.experiments.compute_policy import resolve_compute_policy
from Thesis_ML.experiments.stage_execution import (
    StageAssignment,
    StageBackendFamily,
    StageExecutionResult,
    StageKey,
    build_stage_execution_result,
    stage_execution_payload,
)


def test_stage_key_enum_covers_phase1_operational_stages() -> None:
    assert {stage.value for stage in StageKey} == {
        "dataset_selection",
        "feature_cache_build",
        "feature_matrix_load",
        "spatial_validation",
        "preprocess",
        "model_fit",
        "tuning",
        "permutation",
        "evaluation",
        "reporting",
    }


def test_stage_execution_models_roundtrip_and_status_derivation() -> None:
    compute_policy = resolve_compute_policy(
        framework_mode=FrameworkMode.EXPLORATORY,
        hardware_mode="cpu_only",
    )
    stage_result = build_stage_execution_result(
        compute_policy=compute_policy,
        planned_sections=[
            "dataset_selection",
            "feature_cache_build",
            "feature_matrix_load",
            "spatial_validation",
            "model_fit",
            "interpretability",
            "evaluation",
        ],
        executed_sections=[
            "dataset_selection",
            "feature_cache_build",
            "feature_matrix_load",
            "spatial_validation",
            "model_fit",
            "interpretability",
            "evaluation",
        ],
        reused_sections=[],
        tuning_enabled=False,
        n_permutations=0,
        section_timings_seconds={
            "dataset_selection": 0.02,
            "feature_cache_build": 0.03,
            "feature_matrix_load": 0.04,
            "spatial_validation": 0.01,
            "model_fit": 0.2,
            "evaluation": 0.05,
        },
        stage_timings_seconds={"metrics_stamping": 0.01, "config_write": 0.02},
        reporting_status="planned",
    )
    payload = stage_result.model_dump(mode="json")
    roundtrip = StageExecutionResult.model_validate(payload)

    assert roundtrip.policy.hardware_mode_requested == "cpu_only"
    assert roundtrip.policy.effective_backend_family == "sklearn_cpu"

    telemetry_by_stage = {row.stage: row for row in roundtrip.telemetry}
    assert telemetry_by_stage["model_fit"].status == "executed"
    assert telemetry_by_stage["preprocess"].status == "executed"
    assert telemetry_by_stage["tuning"].status == "skipped"
    assert telemetry_by_stage["permutation"].status == "skipped"
    assert telemetry_by_stage["reporting"].status == "planned"
    assert telemetry_by_stage["dataset_selection"].duration_seconds == 0.02
    assert telemetry_by_stage["reporting"].duration_seconds == 0.03


def test_stage_execution_bridge_honors_actual_estimator_backend_for_compute_stages() -> None:
    compute_policy = resolve_compute_policy(
        framework_mode=FrameworkMode.EXPLORATORY,
        hardware_mode="cpu_only",
    )
    stage_result = build_stage_execution_result(
        compute_policy=compute_policy,
        planned_sections=["model_fit", "interpretability", "evaluation"],
        executed_sections=["model_fit", "interpretability", "evaluation"],
        reused_sections=[],
        tuning_enabled=True,
        n_permutations=10,
        actual_estimator_backend_family="torch_gpu",
        reporting_status="planned",
    )

    assignments_by_stage = {row.stage: row for row in stage_result.assignments}
    assert assignments_by_stage["dataset_selection"].backend_family == "sklearn_cpu"
    assert assignments_by_stage["model_fit"].backend_family == "torch_gpu"
    assert assignments_by_stage["tuning"].backend_family == "torch_gpu"
    assert assignments_by_stage["permutation"].backend_family == "torch_gpu"
    assert assignments_by_stage["evaluation"].backend_family == "torch_gpu"


def test_stage_execution_payload_is_backward_safe_optional_metadata() -> None:
    assert stage_execution_payload(None) is None

    payload = stage_execution_payload(
        {
            "policy": {
                "source": "run_level_compute_policy_bridge_v1",
                "hardware_mode_requested": "cpu_only",
                "hardware_mode_effective": "cpu_only",
                "requested_backend_family": "sklearn_cpu",
                "effective_backend_family": "sklearn_cpu",
                "assigned_compute_lane": "cpu",
                "deterministic_compute": False,
            },
            "assignments": [
                {
                    "stage": "dataset_selection",
                    "backend_family": "sklearn_cpu",
                    "compute_lane": "cpu",
                    "source": "run_level_default_assignment_v1",
                    "reason": "phase1_stage_execution_default_assignment",
                }
            ],
            "telemetry": [
                {
                    "stage": "dataset_selection",
                    "status": "executed",
                    "duration_seconds": 0.001,
                    "details": {},
                }
            ],
        }
    )
    assert payload is not None
    assert payload["policy"]["effective_backend_family"] == "sklearn_cpu"


def test_stage_execution_includes_planner_assignment_metadata_in_telemetry() -> None:
    compute_policy = resolve_compute_policy(
        framework_mode=FrameworkMode.EXPLORATORY,
        hardware_mode="cpu_only",
    )
    stage_result = build_stage_execution_result(
        compute_policy=compute_policy,
        planned_sections=["model_fit", "evaluation"],
        executed_sections=["model_fit", "evaluation"],
        reused_sections=[],
        tuning_enabled=True,
        n_permutations=4,
        reporting_status="planned",
        planned_assignments=[
            StageAssignment(
                stage=StageKey.MODEL_FIT,
                backend_family=StageBackendFamily.SKLEARN_CPU,
                compute_lane="cpu",
                source="stage_planner_v1",
                reason="phase2_test_assignment",
                executor_id="model_fit_cpu_reference_v1",
                equivalence_class="exact_reference_equivalent",
                official_admitted=True,
                fallback_used=False,
                fallback_reason=None,
            ),
        ],
    )

    telemetry_by_stage = {row.stage: row for row in stage_result.telemetry}
    model_fit_details = telemetry_by_stage["model_fit"].details
    assert model_fit_details["executor_id"] == "model_fit_cpu_reference_v1"
    assert model_fit_details["equivalence_class"] == "exact_reference_equivalent"
