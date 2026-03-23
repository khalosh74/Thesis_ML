from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from Thesis_ML.experiments.compute_policy import (
    ResolvedComputePolicy,
    extract_compute_policy_payload,
)

StageStatus = Literal["planned", "executed", "reused", "skipped", "not_planned"]
ComputeLane = Literal["cpu", "gpu"]
StageExecutionPolicySource = Literal["run_level_compute_policy_bridge_v1"]
StageAssignmentSource = Literal["run_level_default_assignment_v1", "stage_planner_v1"]
StageExecutorEquivalence = Literal["exact_reference_equivalent", "validated_variant"]


class StageKey(StrEnum):
    DATASET_SELECTION = "dataset_selection"
    FEATURE_CACHE_BUILD = "feature_cache_build"
    FEATURE_MATRIX_LOAD = "feature_matrix_load"
    SPATIAL_VALIDATION = "spatial_validation"
    PREPROCESS = "preprocess"
    MODEL_FIT = "model_fit"
    TUNING = "tuning"
    PERMUTATION = "permutation"
    EVALUATION = "evaluation"
    REPORTING = "reporting"


class StageBackendFamily(StrEnum):
    SKLEARN_CPU = "sklearn_cpu"
    TORCH_GPU = "torch_gpu"
    AUTO_MIXED = "auto_mixed"


class _StageModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        use_enum_values=True,
    )


class StageExecutionPolicy(_StageModel):
    source: StageExecutionPolicySource = "run_level_compute_policy_bridge_v1"
    hardware_mode_requested: str = Field(min_length=1)
    hardware_mode_effective: str = Field(min_length=1)
    requested_backend_family: StageBackendFamily
    effective_backend_family: StageBackendFamily
    assigned_compute_lane: ComputeLane | None = None
    deterministic_compute: bool = False


class StageAssignment(_StageModel):
    stage: StageKey
    backend_family: StageBackendFamily
    compute_lane: ComputeLane | None = None
    source: StageAssignmentSource = "run_level_default_assignment_v1"
    reason: str = Field(min_length=1)
    executor_id: str | None = None
    equivalence_class: StageExecutorEquivalence | None = None
    official_admitted: bool | None = None
    fallback_used: bool = False
    fallback_reason: str | None = None


class StageExecutionTelemetry(_StageModel):
    stage: StageKey
    status: StageStatus
    duration_seconds: float | None = None
    details: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_duration(self) -> StageExecutionTelemetry:
        if self.duration_seconds is not None and float(self.duration_seconds) < 0.0:
            raise ValueError("duration_seconds must be >= 0 when provided.")
        return self


class StageExecutionResult(_StageModel):
    policy: StageExecutionPolicy
    assignments: list[StageAssignment] = Field(default_factory=list)
    telemetry: list[StageExecutionTelemetry] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_unique_stage_rows(self) -> StageExecutionResult:
        assignment_stages = [assignment.stage for assignment in self.assignments]
        telemetry_stages = [telemetry.stage for telemetry in self.telemetry]
        if len(set(assignment_stages)) != len(assignment_stages):
            raise ValueError("assignments must not contain duplicate stage entries.")
        if len(set(telemetry_stages)) != len(telemetry_stages):
            raise ValueError("telemetry must not contain duplicate stage entries.")
        return self


_SECTION_TO_STAGE: dict[str, StageKey] = {
    "dataset_selection": StageKey.DATASET_SELECTION,
    "feature_cache_build": StageKey.FEATURE_CACHE_BUILD,
    "feature_matrix_load": StageKey.FEATURE_MATRIX_LOAD,
    "spatial_validation": StageKey.SPATIAL_VALIDATION,
    "model_fit": StageKey.MODEL_FIT,
    "evaluation": StageKey.EVALUATION,
}
_MODEL_FIT_DEPENDENT_STAGES: tuple[StageKey, ...] = (StageKey.PREPROCESS, StageKey.TUNING)
_EVALUATION_DEPENDENT_STAGES: tuple[StageKey, ...] = (
    StageKey.PERMUTATION,
    StageKey.EVALUATION,
)
_COMPUTE_HEAVY_STAGES: tuple[StageKey, ...] = (
    StageKey.PREPROCESS,
    StageKey.MODEL_FIT,
    StageKey.TUNING,
    StageKey.PERMUTATION,
    StageKey.EVALUATION,
)


def _coerce_backend_family(value: str) -> StageBackendFamily:
    normalized = str(value).strip()
    try:
        return StageBackendFamily(normalized)
    except ValueError:
        return StageBackendFamily.SKLEARN_CPU


def _default_compute_lane_for_backend(backend_family: StageBackendFamily) -> ComputeLane:
    return "gpu" if backend_family == StageBackendFamily.TORCH_GPU else "cpu"


def _policy_from_compute_policy(
    compute_policy: ResolvedComputePolicy | Mapping[str, Any],
) -> StageExecutionPolicy:
    if isinstance(compute_policy, ResolvedComputePolicy):
        payload = compute_policy.to_payload()
    else:
        extracted = extract_compute_policy_payload(dict(compute_policy))
        if extracted is None:
            raise ValueError("compute_policy payload is missing required compute policy fields.")
        payload = extracted
    return StageExecutionPolicy(
        hardware_mode_requested=str(payload["hardware_mode_requested"]),
        hardware_mode_effective=str(payload["hardware_mode_effective"]),
        requested_backend_family=_coerce_backend_family(str(payload["requested_backend_family"])),
        effective_backend_family=_coerce_backend_family(str(payload["effective_backend_family"])),
        assigned_compute_lane=payload.get("assigned_compute_lane"),
        deterministic_compute=bool(payload.get("deterministic_compute", False)),
    )


def _normalize_stage_duration_map(
    section_timings_seconds: Mapping[str, float] | None,
    stage_timings_seconds: Mapping[str, float] | None,
) -> dict[StageKey, float]:
    stage_durations: dict[StageKey, float] = {}
    if section_timings_seconds is not None:
        for section_name, duration_seconds in section_timings_seconds.items():
            mapped_stage = _SECTION_TO_STAGE.get(str(section_name))
            if mapped_stage is None:
                continue
            stage_durations[mapped_stage] = float(duration_seconds)
    if stage_timings_seconds is not None:
        reporting_duration = 0.0
        for key in ("metrics_stamping", "config_write", "artifact_registry_update"):
            value = stage_timings_seconds.get(key)
            if isinstance(value, (int, float)):
                reporting_duration += float(value)
        if reporting_duration > 0.0:
            stage_durations[StageKey.REPORTING] = reporting_duration
    return stage_durations


def _planned_status_for_stage(
    *,
    stage: StageKey,
    section_stage_status: dict[StageKey, StageStatus],
    tuning_enabled: bool,
    n_permutations: int,
    reporting_status: StageStatus,
) -> StageStatus:
    if stage in section_stage_status:
        return section_stage_status[stage]

    model_fit_status = section_stage_status.get(StageKey.MODEL_FIT, "not_planned")
    evaluation_status = section_stage_status.get(StageKey.EVALUATION, "not_planned")

    if stage == StageKey.PREPROCESS:
        return model_fit_status
    if stage == StageKey.TUNING:
        if model_fit_status == "not_planned":
            return "not_planned"
        if not bool(tuning_enabled):
            return "skipped"
        return model_fit_status
    if stage == StageKey.PERMUTATION:
        if evaluation_status == "not_planned":
            return "not_planned"
        if int(n_permutations) <= 0:
            return "skipped"
        return evaluation_status
    if stage == StageKey.REPORTING:
        return reporting_status
    return "not_planned"


def build_stage_execution_result(
    *,
    compute_policy: ResolvedComputePolicy | Mapping[str, Any],
    planned_sections: Sequence[str],
    executed_sections: Sequence[str],
    reused_sections: Sequence[str],
    tuning_enabled: bool,
    n_permutations: int,
    section_timings_seconds: Mapping[str, float] | None = None,
    stage_timings_seconds: Mapping[str, float] | None = None,
    reporting_status: StageStatus = "planned",
    actual_estimator_backend_family: str | None = None,
    planned_assignments: Sequence[StageAssignment | Mapping[str, Any]] | None = None,
) -> StageExecutionResult:
    policy = _policy_from_compute_policy(compute_policy)
    effective_compute_backend = (
        _coerce_backend_family(actual_estimator_backend_family)
        if isinstance(actual_estimator_backend_family, str)
        and str(actual_estimator_backend_family).strip()
        else policy.effective_backend_family
    )
    compute_lane = policy.assigned_compute_lane or _default_compute_lane_for_backend(
        effective_compute_backend
    )

    planned_set = {str(name) for name in planned_sections}
    executed_set = {str(name) for name in executed_sections}
    reused_set = {str(name) for name in reused_sections}

    section_stage_status: dict[StageKey, StageStatus] = {}
    for section_name, stage_key in _SECTION_TO_STAGE.items():
        if section_name in reused_set:
            section_stage_status[stage_key] = "reused"
            continue
        if section_name in executed_set:
            section_stage_status[stage_key] = "executed"
            continue
        if section_name in planned_set:
            section_stage_status[stage_key] = "planned"
            continue
        section_stage_status[stage_key] = "not_planned"

    assignment_map: dict[StageKey, StageAssignment] = {}
    if planned_assignments is not None:
        for raw_assignment in planned_assignments:
            if isinstance(raw_assignment, StageAssignment):
                assignment = raw_assignment
            else:
                assignment = StageAssignment.model_validate(dict(raw_assignment))
            assignment_map[StageKey(str(assignment.stage))] = assignment
    for stage in StageKey:
        if stage in assignment_map:
            continue
        if stage in _COMPUTE_HEAVY_STAGES:
            backend_family = effective_compute_backend
            stage_lane: ComputeLane | None = compute_lane
        else:
            backend_family = StageBackendFamily.SKLEARN_CPU
            stage_lane = "cpu"
        assignment_map[stage] = StageAssignment(
            stage=stage,
            backend_family=backend_family,
            compute_lane=stage_lane,
            reason="phase1_stage_execution_default_assignment",
        )
    assignments = [assignment_map[stage] for stage in StageKey]

    stage_durations = _normalize_stage_duration_map(
        section_timings_seconds=section_timings_seconds,
        stage_timings_seconds=stage_timings_seconds,
    )

    telemetry_rows: list[StageExecutionTelemetry] = []
    for stage in StageKey:
        assignment = assignment_map[stage]
        status = _planned_status_for_stage(
            stage=stage,
            section_stage_status=section_stage_status,
            tuning_enabled=bool(tuning_enabled),
            n_permutations=int(n_permutations),
            reporting_status=reporting_status,
        )
        details: dict[str, Any] = {}
        if assignment.executor_id is not None:
            details["executor_id"] = str(assignment.executor_id)
        if assignment.equivalence_class is not None:
            details["equivalence_class"] = str(assignment.equivalence_class)
        if assignment.fallback_reason is not None:
            details["fallback_reason"] = str(assignment.fallback_reason)
        if stage in _MODEL_FIT_DEPENDENT_STAGES:
            details["derived_from"] = StageKey.MODEL_FIT.value
        elif stage in _EVALUATION_DEPENDENT_STAGES:
            details["derived_from"] = StageKey.EVALUATION.value
        duration_seconds = stage_durations.get(stage)
        telemetry_rows.append(
            StageExecutionTelemetry(
                stage=stage,
                status=status,
                duration_seconds=(float(duration_seconds) if duration_seconds is not None else None),
                details=details,
            )
        )

    return StageExecutionResult(
        policy=policy,
        assignments=assignments,
        telemetry=telemetry_rows,
    )


def stage_execution_payload(
    value: StageExecutionResult | Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, StageExecutionResult):
        return value.model_dump(mode="json")
    try:
        return StageExecutionResult.model_validate(dict(value)).model_dump(mode="json")
    except Exception:
        return None


__all__ = [
    "StageAssignment",
    "StageAssignmentSource",
    "StageBackendFamily",
    "StageExecutionPolicy",
    "StageExecutionPolicySource",
    "StageExecutionResult",
    "StageExecutionTelemetry",
    "StageExecutorEquivalence",
    "StageKey",
    "StageStatus",
    "build_stage_execution_result",
    "stage_execution_payload",
]
