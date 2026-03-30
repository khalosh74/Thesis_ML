from __future__ import annotations

from Thesis_ML.config.framework_mode import FrameworkMode
from Thesis_ML.experiments.compute_policy import resolve_compute_policy
from Thesis_ML.experiments.stage_execution import StageAssignment, StageBackendFamily, StageKey, build_stage_execution_result
from Thesis_ML.verification.stage_execution_verifier import verify_stage_execution_evidence


def test_stage_execution_verifier_detects_mismatches_and_unexpected_fallbacks() -> None:
    stage_execution = {
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
                "stage": "model_fit",
                "backend_family": "sklearn_cpu",
                "compute_lane": "cpu",
                "source": "stage_planner_v1",
                "reason": "test",
                "executor_id": "model_fit_cpu_reference_v1",
                "fallback_used": False,
            }
        ],
        "telemetry": [
            {
                "stage": "model_fit",
                "status": "executed",
                "planned_backend_family": "sklearn_cpu",
                "planned_compute_lane": "cpu",
                "planned_executor_id": "model_fit_cpu_reference_v1",
                "observed_backend_family": "torch_gpu",
                "observed_compute_lane": "gpu",
                "observed_executor_id": "torch_ridge_gpu_v2",
                "backend_match": False,
                "lane_match": False,
                "executor_match": False,
                "planning_match": False,
                "fallback_expected": False,
                "fallback_used": True,
                "fallback_reason": "runtime_fallback",
                "observed_evidence_present": False,
                "missing_observed_evidence": True,
                "evidence_quality": "low",
                "resource_coverage": "none",
                "primary_artifacts": [],
                "details": {},
            }
        ],
    }

    result = verify_stage_execution_evidence(
        stage_execution=stage_execution,
        run_status={"status": "completed"},
        process_profile_summary={"sample_count": 0},
    )
    codes = {str(item.get("code")) for item in result["findings"]}
    assert "backend_mismatch" in codes
    assert "lane_mismatch" in codes
    assert "executor_mismatch" in codes
    assert "unexpected_fallback" in codes
    assert "missing_evidence" in codes
    assert "low_confidence_evidence" in codes
    assert result["passed"] is False


def test_stage_execution_verifier_passes_on_planned_observed_agreement() -> None:
    compute_policy = resolve_compute_policy(
        framework_mode=FrameworkMode.EXPLORATORY,
        hardware_mode="cpu_only",
    )
    stage_result = build_stage_execution_result(
        compute_policy=compute_policy,
        planned_sections=["model_fit", "evaluation"],
        executed_sections=["model_fit", "evaluation"],
        reused_sections=[],
        tuning_enabled=False,
        n_permutations=0,
        planned_assignments=[
            StageAssignment(
                stage=StageKey.MODEL_FIT,
                backend_family=StageBackendFamily.SKLEARN_CPU,
                compute_lane="cpu",
                source="stage_planner_v1",
                reason="test_model_fit",
                executor_id="model_fit_cpu_reference_v1",
                official_admitted=True,
            ),
            StageAssignment(
                stage=StageKey.EVALUATION,
                backend_family=StageBackendFamily.SKLEARN_CPU,
                compute_lane="cpu",
                source="stage_planner_v1",
                reason="test_evaluation",
                executor_id="evaluation_cpu_reference_v1",
                official_admitted=True,
            ),
        ],
        observed_stage_evidence={
            "preprocess": {
                "observed_status": "executed",
                "status_source": "derived_from_model_fit",
                "observed_backend_family": "sklearn_cpu",
                "observed_compute_lane": "cpu",
                "observed_executor_id": "preprocess_cpu_reference_v1",
                "started_at_utc": "2026-01-01T00:00:00+00:00",
                "ended_at_utc": "2026-01-01T00:00:05+00:00",
                "derived_from_stage": "model_fit",
            },
            "model_fit": {
                "observed_status": "executed",
                "status_source": "stage_events",
                "observed_backend_family": "sklearn_cpu",
                "observed_compute_lane": "cpu",
                "observed_executor_id": "model_fit_cpu_reference_v1",
                "started_at_utc": "2026-01-01T00:00:00+00:00",
                "ended_at_utc": "2026-01-01T00:00:05+00:00",
                "primary_artifacts": ["fit_timing_summary.json"],
            },
            "evaluation": {
                "observed_status": "executed",
                "status_source": "stage_events",
                "observed_backend_family": "sklearn_cpu",
                "observed_compute_lane": "cpu",
                "observed_executor_id": "evaluation_cpu_reference_v1",
                "started_at_utc": "2026-01-01T00:00:06+00:00",
                "ended_at_utc": "2026-01-01T00:00:08+00:00",
                "primary_artifacts": ["metrics.json"],
            },
        },
        stage_resource_attribution={
            "model_fit": {
                "resource_coverage": "high",
                "evidence_quality": "high",
            },
            "evaluation": {
                "resource_coverage": "high",
                "evidence_quality": "high",
            },
        },
    )

    result = verify_stage_execution_evidence(
        stage_execution=stage_result,
        run_status={"status": "completed"},
        process_profile_summary={"sample_count": 4},
    )
    assert result["passed"] is True
    assert result["findings"] == []


def test_stage_execution_verifier_flags_missing_required_gpu_lease_evidence() -> None:
    stage_execution = {
        "policy": {
            "source": "run_level_compute_policy_bridge_v1",
            "hardware_mode_requested": "gpu_only",
            "hardware_mode_effective": "gpu_only",
            "requested_backend_family": "torch_gpu",
            "effective_backend_family": "torch_gpu",
            "assigned_compute_lane": "gpu",
            "deterministic_compute": False,
        },
        "assignments": [
            {
                "stage": "permutation",
                "backend_family": "torch_gpu",
                "compute_lane": "gpu",
                "source": "stage_planner_v1",
                "reason": "test",
                "executor_id": "permutation_ridge_gpu_preferred_v1",
                "fallback_used": False,
            }
        ],
        "telemetry": [
            {
                "stage": "permutation",
                "status": "executed",
                "planned_backend_family": "torch_gpu",
                "planned_compute_lane": "gpu",
                "planned_executor_id": "permutation_ridge_gpu_preferred_v1",
                "observed_backend_family": "torch_gpu",
                "observed_compute_lane": "gpu",
                "observed_executor_id": "permutation_ridge_gpu_preferred_v1",
                "observed_evidence_present": True,
                "missing_observed_evidence": False,
                "lease_required": True,
                "lease_class": "gpu",
                "lease_acquired": False,
                "lease_released": False,
                "evidence_quality": "medium",
                "resource_coverage": "partial",
                "details": {},
            }
        ],
    }

    result = verify_stage_execution_evidence(stage_execution=stage_execution)
    codes = {str(item.get("code")) for item in result["findings"]}
    assert "required_gpu_lease_missing" in codes
    assert "observed_gpu_without_lease" in codes
