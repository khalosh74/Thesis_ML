from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from Thesis_ML.experiments.compute_policy import (
    ResolvedComputePolicy,
    extract_compute_policy_payload,
)

StageStatus = Literal["planned", "executed", "reused", "skipped", "not_planned", "missing"]
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
    XGBOOST_CPU = "xgboost_cpu"
    XGBOOST_GPU = "xgboost_gpu"


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
    planned_backend_family: StageBackendFamily | None = None
    planned_compute_lane: ComputeLane | None = None
    planned_executor_id: str | None = None
    official_admitted: bool | None = None
    assignment_source: str | None = None
    observed_backend_family: StageBackendFamily | None = None
    observed_compute_lane: ComputeLane | None = None
    observed_executor_id: str | None = None
    fallback_used: bool | None = None
    fallback_reason: str | None = None
    execution_mode: str | None = None
    started_at_utc: str | None = None
    ended_at_utc: str | None = None
    duration_source: str | None = None
    resource_coverage: str | None = None
    evidence_quality: str | None = None
    mean_cpu_percent: float | None = None
    peak_cpu_percent: float | None = None
    peak_rss_mb: float | None = None
    peak_vms_mb: float | None = None
    peak_thread_count: int | None = None
    read_bytes_delta: int | None = None
    write_bytes_delta: int | None = None
    peak_gpu_memory_mb: float | None = None
    peak_gpu_utilization_percent: float | None = None
    mean_gpu_utilization_percent: float | None = None
    lease_required: bool | None = None
    lease_class: str | None = None
    lease_owner_identity: str | None = None
    lease_acquired: bool | None = None
    lease_wait_seconds: float | None = None
    lease_queue_depth_at_acquire: int | None = None
    lease_acquired_at_utc: str | None = None
    lease_released_at_utc: str | None = None
    lease_held_seconds: float | None = None
    lease_released: bool | None = None
    primary_artifacts: list[str] = Field(default_factory=list)
    status_source: str | None = None
    derived_from_stage: StageKey | None = None
    planning_match: bool | None = None
    backend_match: bool | None = None
    lane_match: bool | None = None
    executor_match: bool | None = None
    fallback_expected: bool | None = None
    observed_evidence_present: bool = False
    missing_observed_evidence: bool = False
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
    return (
        "gpu"
        if backend_family in {StageBackendFamily.TORCH_GPU, StageBackendFamily.XGBOOST_GPU}
        else "cpu"
    )


def _assignment_requires_gpu_lease(assignment: StageAssignment) -> bool:
    backend_family_value = str(assignment.backend_family).strip().lower()
    compute_lane_value = (
        str(assignment.compute_lane).strip().lower()
        if assignment.compute_lane is not None
        else ""
    )
    executor_id_value = (
        str(assignment.executor_id).strip().lower() if assignment.executor_id is not None else ""
    )
    return bool(
        backend_family_value in {StageBackendFamily.TORCH_GPU.value, StageBackendFamily.XGBOOST_GPU.value}
        or compute_lane_value == "gpu"
        or (executor_id_value and ("gpu" in executor_id_value or executor_id_value.startswith("torch_")))
    )


def _parse_utc_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _coerce_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _duration_from_interval(
    *,
    started_at_utc: str | None,
    ended_at_utc: str | None,
) -> float | None:
    start = _parse_utc_timestamp(started_at_utc)
    end = _parse_utc_timestamp(ended_at_utc)
    if start is None or end is None:
        return None
    return max(0.0, float((end - start).total_seconds()))


def _normalize_observed_stage_evidence(
    value: Mapping[str, Mapping[str, Any]] | None,
) -> dict[StageKey, dict[str, Any]]:
    if value is None:
        return {}
    normalized: dict[StageKey, dict[str, Any]] = {}
    for raw_key, raw_payload in dict(value).items():
        if not isinstance(raw_payload, Mapping):
            continue
        try:
            stage_key = StageKey(str(raw_key))
        except ValueError:
            continue
        normalized[stage_key] = {str(key): item for key, item in dict(raw_payload).items()}
    return normalized


def _normalize_stage_resource_map(
    value: Mapping[str, Mapping[str, Any]] | None,
) -> dict[StageKey, dict[str, Any]]:
    if value is None:
        return {}
    normalized: dict[StageKey, dict[str, Any]] = {}
    for raw_key, raw_payload in dict(value).items():
        if not isinstance(raw_payload, Mapping):
            continue
        try:
            stage_key = StageKey(str(raw_key))
        except ValueError:
            continue
        normalized[stage_key] = {str(key): item for key, item in dict(raw_payload).items()}
    return normalized


def _coerce_stage_status(value: Any) -> StageStatus | None:
    if not isinstance(value, str):
        return None
    candidate = value.strip().lower()
    if candidate in {"planned", "executed", "reused", "skipped", "not_planned", "missing"}:
        return candidate  # type: ignore[return-value]
    if candidate == "started":
        return "planned"
    return None


def _evidence_quality_for_stage(
    *,
    status: StageStatus,
    observed_present: bool,
    duration_seconds: float | None,
    resource_coverage: str | None,
) -> str:
    if status == "not_planned":
        return "high"
    if not observed_present:
        if status in {"executed", "reused"}:
            return "low"
        return "medium"
    if resource_coverage == "high":
        return "high"
    if resource_coverage == "partial":
        return "medium"
    if duration_seconds is not None:
        return "medium"
    return "low"


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
) -> tuple[dict[StageKey, float], dict[StageKey, dict[str, Any]]]:
    stage_durations: dict[StageKey, float] = {}
    duration_metadata: dict[StageKey, dict[str, Any]] = {}
    if section_timings_seconds is not None:
        for section_name, duration_seconds in section_timings_seconds.items():
            mapped_stage = _SECTION_TO_STAGE.get(str(section_name))
            if mapped_stage is None:
                continue
            stage_durations[mapped_stage] = float(duration_seconds)
            duration_metadata[mapped_stage] = {
                "duration_source": "section_timing",
                "derived_from": str(section_name),
            }
    if stage_timings_seconds is not None:
        for stage_name, duration_seconds in stage_timings_seconds.items():
            try:
                mapped_stage = StageKey(str(stage_name))
            except ValueError:
                mapped_stage = None
            if mapped_stage is not None and isinstance(duration_seconds, (int, float)):
                stage_durations[mapped_stage] = float(duration_seconds)
                duration_metadata[mapped_stage] = {
                    "duration_source": "stage_timing_map",
                    "derived_from": str(stage_name),
                }
        reporting_duration = 0.0
        reporting_sources: list[str] = []
        for key in ("metrics_stamping", "config_write", "artifact_registry_update"):
            value = stage_timings_seconds.get(key)
            if isinstance(value, (int, float)):
                reporting_duration += float(value)
                reporting_sources.append(str(key))
        if reporting_duration > 0.0:
            stage_durations[StageKey.REPORTING] = reporting_duration
            duration_metadata[StageKey.REPORTING] = {
                "duration_source": "run_stage_timings_rollup",
                "derived_from": list(reporting_sources),
            }
    return stage_durations, duration_metadata


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
    stage_timing_metadata: Mapping[str, Mapping[str, Any]] | None = None,
    reporting_status: StageStatus = "planned",
    actual_estimator_backend_family: str | None = None,
    planned_assignments: Sequence[StageAssignment | Mapping[str, Any]] | None = None,
    observed_stage_evidence: Mapping[str, Mapping[str, Any]] | None = None,
    stage_resource_attribution: Mapping[str, Mapping[str, Any]] | None = None,
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

    stage_durations, stage_duration_metadata = _normalize_stage_duration_map(
        section_timings_seconds=section_timings_seconds,
        stage_timings_seconds=stage_timings_seconds,
    )
    if stage_timing_metadata is not None:
        for raw_stage, raw_metadata in stage_timing_metadata.items():
            try:
                stage_key = StageKey(str(raw_stage))
            except ValueError:
                continue
            if not isinstance(raw_metadata, Mapping):
                continue
            stage_duration_metadata[stage_key] = {
                str(key): value for key, value in dict(raw_metadata).items()
            }

    observed_stage_map = _normalize_observed_stage_evidence(observed_stage_evidence)
    stage_resource_map = _normalize_stage_resource_map(stage_resource_attribution)
    observed_evidence_enabled = observed_stage_evidence is not None

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
        observed_payload = dict(observed_stage_map.get(stage, {}))
        resource_payload = dict(stage_resource_map.get(stage, {}))
        if not resource_payload and isinstance(observed_payload.get("resource_summary"), Mapping):
            resource_payload = {
                str(key): value
                for key, value in dict(observed_payload["resource_summary"]).items()
            }
        observed_status = _coerce_stage_status(
            observed_payload.get("observed_status") or observed_payload.get("status")
        )
        status_source = "planned_section_status"
        if observed_status is not None:
            status = observed_status
            status_source = str(observed_payload.get("status_source") or "observed_stage_evidence")

        started_at_utc = (
            str(observed_payload.get("started_at_utc"))
            if isinstance(observed_payload.get("started_at_utc"), str)
            else None
        )
        ended_at_utc = (
            str(observed_payload.get("ended_at_utc"))
            if isinstance(observed_payload.get("ended_at_utc"), str)
            else None
        )

        details: dict[str, Any] = {}
        if assignment.executor_id is not None:
            details["executor_id"] = str(assignment.executor_id)
        if assignment.equivalence_class is not None:
            details["equivalence_class"] = str(assignment.equivalence_class)
        if assignment.fallback_reason is not None:
            details["fallback_reason"] = str(assignment.fallback_reason)
        details["planned_backend_family"] = str(assignment.backend_family)
        details["planned_compute_lane"] = (
            str(assignment.compute_lane) if assignment.compute_lane is not None else None
        )
        details["planned_executor_id"] = (
            str(assignment.executor_id) if assignment.executor_id is not None else None
        )
        details["official_admitted"] = assignment.official_admitted
        details["assignment_source"] = str(assignment.source)
        if stage in _MODEL_FIT_DEPENDENT_STAGES:
            details["derived_from"] = StageKey.MODEL_FIT.value
        elif stage in _EVALUATION_DEPENDENT_STAGES:
            details["derived_from"] = StageKey.EVALUATION.value

        duration_seconds = stage_durations.get(stage)
        metadata_payload = stage_duration_metadata.get(stage)
        if isinstance(metadata_payload, Mapping):
            details.update({str(key): value for key, value in metadata_payload.items()})
        duration_source: str | None = (
            str(details.get("duration_source"))
            if isinstance(details.get("duration_source"), str)
            else None
        )
        observed_duration = _coerce_float(observed_payload.get("duration_seconds"))
        if duration_seconds is None and observed_duration is not None:
            duration_seconds = observed_duration
            if duration_source is None:
                duration_source = "observed_stage_evidence"
        if duration_seconds is None:
            interval_duration = _duration_from_interval(
                started_at_utc=started_at_utc,
                ended_at_utc=ended_at_utc,
            )
            if interval_duration is not None:
                duration_seconds = interval_duration
                if duration_source is None:
                    duration_source = "observed_stage_interval"
        if duration_source is None and isinstance(observed_payload.get("duration_source"), str):
            duration_source = str(observed_payload.get("duration_source"))
        if duration_source is None:
            if stage in _MODEL_FIT_DEPENDENT_STAGES or stage in _EVALUATION_DEPENDENT_STAGES:
                duration_source = "unavailable_derived_stage"
                details.setdefault("fallback_reason", "no_direct_duration_measurement")
            else:
                duration_source = "unavailable"
        details["duration_source"] = duration_source

        observed_backend_candidate = observed_payload.get("observed_backend_family")
        if not isinstance(observed_backend_candidate, str) and stage == StageKey.MODEL_FIT:
            observed_backend_candidate = actual_estimator_backend_family
        observed_backend_family = (
            _coerce_backend_family(observed_backend_candidate)
            if isinstance(observed_backend_candidate, str) and observed_backend_candidate.strip()
            else None
        )
        observed_compute_lane = (
            str(observed_payload.get("observed_compute_lane"))
            if isinstance(observed_payload.get("observed_compute_lane"), str)
            and str(observed_payload.get("observed_compute_lane")).strip()
            else None
        )
        observed_executor_id = (
            str(observed_payload.get("observed_executor_id"))
            if isinstance(observed_payload.get("observed_executor_id"), str)
            and str(observed_payload.get("observed_executor_id")).strip()
            else None
        )
        fallback_used = (
            bool(observed_payload.get("fallback_used"))
            if isinstance(observed_payload.get("fallback_used"), bool)
            else bool(assignment.fallback_used)
        )
        fallback_reason = (
            str(observed_payload.get("fallback_reason"))
            if isinstance(observed_payload.get("fallback_reason"), str)
            and str(observed_payload.get("fallback_reason")).strip()
            else (str(assignment.fallback_reason) if assignment.fallback_reason else None)
        )
        execution_mode = (
            str(observed_payload.get("execution_mode"))
            if isinstance(observed_payload.get("execution_mode"), str)
            and str(observed_payload.get("execution_mode")).strip()
            else None
        )
        planned_lease_required = _assignment_requires_gpu_lease(assignment)
        lease_required_raw = observed_payload.get("lease_required")
        if isinstance(lease_required_raw, bool):
            lease_required = bool(lease_required_raw)
        elif isinstance(lease_required_raw, int):
            lease_required = bool(int(lease_required_raw))
        else:
            lease_required = bool(planned_lease_required)
        lease_class = (
            str(observed_payload.get("lease_class"))
            if isinstance(observed_payload.get("lease_class"), str)
            and str(observed_payload.get("lease_class")).strip()
            else ("gpu" if lease_required else "cpu")
        )
        lease_owner_identity = (
            str(observed_payload.get("lease_owner_identity"))
            if isinstance(observed_payload.get("lease_owner_identity"), str)
            and str(observed_payload.get("lease_owner_identity")).strip()
            else None
        )
        lease_acquired = (
            bool(observed_payload.get("lease_acquired"))
            if isinstance(observed_payload.get("lease_acquired"), (bool, int))
            else None
        )
        lease_wait_seconds = _coerce_float(observed_payload.get("lease_wait_seconds"))
        lease_queue_depth_at_acquire = (
            int(observed_payload.get("lease_queue_depth_at_acquire"))
            if isinstance(observed_payload.get("lease_queue_depth_at_acquire"), int)
            else None
        )
        lease_acquired_at_utc = (
            str(observed_payload.get("lease_acquired_at_utc"))
            if isinstance(observed_payload.get("lease_acquired_at_utc"), str)
            and str(observed_payload.get("lease_acquired_at_utc")).strip()
            else None
        )
        lease_released_at_utc = (
            str(observed_payload.get("lease_released_at_utc"))
            if isinstance(observed_payload.get("lease_released_at_utc"), str)
            and str(observed_payload.get("lease_released_at_utc")).strip()
            else None
        )
        lease_held_seconds = _coerce_float(observed_payload.get("lease_held_seconds"))
        lease_released = (
            bool(observed_payload.get("lease_released"))
            if isinstance(observed_payload.get("lease_released"), (bool, int))
            else None
        )
        primary_artifacts = (
            [str(item) for item in observed_payload.get("primary_artifacts", []) if str(item).strip()]
            if isinstance(observed_payload.get("primary_artifacts"), list)
            else []
        )

        resource_coverage = (
            str(resource_payload.get("resource_coverage"))
            if isinstance(resource_payload.get("resource_coverage"), str)
            and str(resource_payload.get("resource_coverage")).strip()
            else (
                str(observed_payload.get("resource_coverage"))
                if isinstance(observed_payload.get("resource_coverage"), str)
                and str(observed_payload.get("resource_coverage")).strip()
                else None
            )
        )
        observed_present = bool(
            observed_payload
            and (
                started_at_utc is not None
                or ended_at_utc is not None
                or observed_status is not None
                or observed_backend_family is not None
                or observed_executor_id is not None
            )
        )
        missing_observed_evidence = bool(
            observed_evidence_enabled and status in {"executed", "reused"} and not observed_present
        )
        if missing_observed_evidence:
            status = "missing"
            status_source = "missing_observed_evidence"

        backend_match = (
            bool(observed_backend_family == assignment.backend_family)
            if observed_backend_family is not None
            else None
        )
        lane_match = (
            bool(observed_compute_lane == assignment.compute_lane)
            if observed_compute_lane is not None and assignment.compute_lane is not None
            else None
        )
        executor_match = (
            bool(observed_executor_id == assignment.executor_id)
            if observed_executor_id is not None and assignment.executor_id is not None
            else None
        )
        match_components = [
            match for match in (backend_match, lane_match, executor_match) if match is not None
        ]
        planning_match = bool(all(match_components)) if match_components else None
        fallback_expected = bool(assignment.fallback_used)
        derived_from_stage: StageKey | None = None
        derived_from_stage_raw = observed_payload.get("derived_from_stage")
        if isinstance(derived_from_stage_raw, str):
            try:
                derived_from_stage = StageKey(str(derived_from_stage_raw))
            except ValueError:
                derived_from_stage = None
        elif stage in _MODEL_FIT_DEPENDENT_STAGES:
            derived_from_stage = StageKey.MODEL_FIT
        elif stage in _EVALUATION_DEPENDENT_STAGES:
            derived_from_stage = StageKey.EVALUATION

        evidence_quality = (
            str(observed_payload.get("evidence_quality"))
            if isinstance(observed_payload.get("evidence_quality"), str)
            and str(observed_payload.get("evidence_quality")).strip()
            else (
                str(resource_payload.get("evidence_quality"))
                if isinstance(resource_payload.get("evidence_quality"), str)
                and str(resource_payload.get("evidence_quality")).strip()
                else _evidence_quality_for_stage(
                    status=status,
                    observed_present=observed_present,
                    duration_seconds=duration_seconds,
                    resource_coverage=resource_coverage,
                )
            )
        )

        if resource_payload:
            details["resource_summary"] = dict(resource_payload)
        details["status_source"] = status_source
        details["observed_evidence_present"] = observed_present
        details["missing_observed_evidence"] = missing_observed_evidence
        details["resource_coverage"] = resource_coverage
        details["planning_match"] = planning_match
        details["backend_match"] = backend_match
        details["lane_match"] = lane_match
        details["executor_match"] = executor_match
        details["fallback_expected"] = fallback_expected
        details["lease_required"] = lease_required
        details["lease_class"] = lease_class
        details["lease_owner_identity"] = lease_owner_identity
        details["lease_acquired"] = lease_acquired
        details["lease_wait_seconds"] = lease_wait_seconds
        details["lease_queue_depth_at_acquire"] = lease_queue_depth_at_acquire
        details["lease_acquired_at_utc"] = lease_acquired_at_utc
        details["lease_released_at_utc"] = lease_released_at_utc
        details["lease_held_seconds"] = lease_held_seconds
        details["lease_released"] = lease_released

        telemetry_rows.append(
            StageExecutionTelemetry(
                stage=stage,
                status=status,
                duration_seconds=(float(duration_seconds) if duration_seconds is not None else None),
                planned_backend_family=assignment.backend_family,
                planned_compute_lane=assignment.compute_lane,
                planned_executor_id=assignment.executor_id,
                official_admitted=assignment.official_admitted,
                assignment_source=str(assignment.source),
                observed_backend_family=observed_backend_family,
                observed_compute_lane=observed_compute_lane,
                observed_executor_id=observed_executor_id,
                fallback_used=fallback_used,
                fallback_reason=fallback_reason,
                execution_mode=execution_mode,
                started_at_utc=started_at_utc,
                ended_at_utc=ended_at_utc,
                duration_source=duration_source,
                resource_coverage=resource_coverage,
                evidence_quality=evidence_quality,
                mean_cpu_percent=_coerce_float(resource_payload.get("mean_cpu_percent")),
                peak_cpu_percent=_coerce_float(resource_payload.get("peak_cpu_percent")),
                peak_rss_mb=_coerce_float(resource_payload.get("peak_rss_mb")),
                peak_vms_mb=_coerce_float(resource_payload.get("peak_vms_mb")),
                peak_thread_count=(
                    int(resource_payload.get("peak_thread_count"))
                    if isinstance(resource_payload.get("peak_thread_count"), int)
                    else None
                ),
                read_bytes_delta=(
                    int(resource_payload.get("read_bytes_delta"))
                    if isinstance(resource_payload.get("read_bytes_delta"), int)
                    else None
                ),
                write_bytes_delta=(
                    int(resource_payload.get("write_bytes_delta"))
                    if isinstance(resource_payload.get("write_bytes_delta"), int)
                    else None
                ),
                peak_gpu_memory_mb=_coerce_float(resource_payload.get("peak_gpu_memory_mb")),
                peak_gpu_utilization_percent=_coerce_float(
                    resource_payload.get("peak_gpu_utilization_percent")
                ),
                mean_gpu_utilization_percent=_coerce_float(
                    resource_payload.get("mean_gpu_utilization_percent")
                ),
                lease_required=lease_required,
                lease_class=lease_class,
                lease_owner_identity=lease_owner_identity,
                lease_acquired=lease_acquired,
                lease_wait_seconds=lease_wait_seconds,
                lease_queue_depth_at_acquire=lease_queue_depth_at_acquire,
                lease_acquired_at_utc=lease_acquired_at_utc,
                lease_released_at_utc=lease_released_at_utc,
                lease_held_seconds=lease_held_seconds,
                lease_released=lease_released,
                primary_artifacts=primary_artifacts,
                status_source=status_source,
                derived_from_stage=derived_from_stage,
                planning_match=planning_match,
                backend_match=backend_match,
                lane_match=lane_match,
                executor_match=executor_match,
                fallback_expected=fallback_expected,
                observed_evidence_present=observed_present,
                missing_observed_evidence=missing_observed_evidence,
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
