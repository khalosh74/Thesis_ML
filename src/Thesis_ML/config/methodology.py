from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from Thesis_ML.config.metric_policy import SUPPORTED_CLASSIFICATION_METRICS

ALLOWED_SUBGROUP_DIMENSIONS = frozenset({"label", "task", "modality", "session", "subject"})


class _MethodologyModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class MethodologyPolicyName(StrEnum):
    FIXED_BASELINES_ONLY = "fixed_baselines_only"
    GROUPED_NESTED_TUNING = "grouped_nested_tuning"


class ClassWeightPolicy(StrEnum):
    NONE = "none"
    BALANCED = "balanced"


class MethodologyPolicy(_MethodologyModel):
    policy_name: MethodologyPolicyName
    class_weight_policy: ClassWeightPolicy = ClassWeightPolicy.NONE
    tuning_enabled: bool = False
    inner_cv_scheme: Literal["grouped_leave_one_group_out"] | None = None
    inner_group_field: str | None = None
    tuning_search_space_id: str | None = None
    tuning_search_space_version: str | None = None
    notes: str | None = None

    @model_validator(mode="after")
    def _validate_methodology(self) -> MethodologyPolicy:
        if self.policy_name == MethodologyPolicyName.FIXED_BASELINES_ONLY:
            if self.tuning_enabled:
                raise ValueError(
                    "methodology_policy.policy_name='fixed_baselines_only' forbids tuning_enabled=true."
                )
            disallowed = {
                "inner_cv_scheme": self.inner_cv_scheme,
                "inner_group_field": self.inner_group_field,
                "tuning_search_space_id": self.tuning_search_space_id,
                "tuning_search_space_version": self.tuning_search_space_version,
            }
            present = [key for key, value in disallowed.items() if value is not None]
            if present:
                raise ValueError(
                    "fixed_baselines_only forbids tuning fields: " + ", ".join(sorted(present))
                )
            return self

        if self.policy_name == MethodologyPolicyName.GROUPED_NESTED_TUNING:
            if not self.tuning_enabled:
                raise ValueError(
                    "methodology_policy.policy_name='grouped_nested_tuning' requires tuning_enabled=true."
                )
            required = {
                "inner_cv_scheme": self.inner_cv_scheme,
                "inner_group_field": self.inner_group_field,
                "tuning_search_space_id": self.tuning_search_space_id,
                "tuning_search_space_version": self.tuning_search_space_version,
            }
            missing = [key for key, value in required.items() if value is None or value == ""]
            if missing:
                raise ValueError(
                    "grouped_nested_tuning is missing required tuning fields: "
                    + ", ".join(sorted(missing))
                )
            return self

        raise ValueError(f"Unsupported methodology policy: {self.policy_name}")


class MetricPolicy(_MethodologyModel):
    primary_metric: str = "balanced_accuracy"
    secondary_metrics: list[str] = Field(default_factory=lambda: ["macro_f1", "accuracy"])

    @model_validator(mode="after")
    def _validate_metrics(self) -> MetricPolicy:
        if self.primary_metric not in SUPPORTED_CLASSIFICATION_METRICS:
            allowed = ", ".join(sorted(SUPPORTED_CLASSIFICATION_METRICS))
            raise ValueError(
                f"Unsupported metric_policy.primary_metric '{self.primary_metric}'. "
                f"Allowed values: {allowed}."
            )
        for metric_name in self.secondary_metrics:
            if metric_name not in SUPPORTED_CLASSIFICATION_METRICS:
                allowed = ", ".join(sorted(SUPPORTED_CLASSIFICATION_METRICS))
                raise ValueError(
                    f"Unsupported secondary metric '{metric_name}'. Allowed values: {allowed}."
                )
        if len(set(self.secondary_metrics)) != len(self.secondary_metrics):
            raise ValueError("metric_policy.secondary_metrics must be unique.")
        return self


class SubgroupReportingPolicy(_MethodologyModel):
    enabled: bool = True
    subgroup_dimensions: list[str] = Field(
        default_factory=lambda: ["label", "task", "modality", "session", "subject"]
    )
    min_samples_per_group: int = 1

    @model_validator(mode="after")
    def _validate_subgroups(self) -> SubgroupReportingPolicy:
        if int(self.min_samples_per_group) <= 0:
            raise ValueError("subgroup_reporting_policy.min_samples_per_group must be > 0.")
        if len(set(self.subgroup_dimensions)) != len(self.subgroup_dimensions):
            raise ValueError("subgroup_reporting_policy.subgroup_dimensions must be unique.")
        invalid = [
            value for value in self.subgroup_dimensions if value not in ALLOWED_SUBGROUP_DIMENSIONS
        ]
        if invalid:
            allowed = ", ".join(sorted(ALLOWED_SUBGROUP_DIMENSIONS))
            raise ValueError(
                "Unsupported subgroup dimensions: "
                + ", ".join(sorted(set(invalid)))
                + f". Allowed values: {allowed}."
            )
        return self


class ComparisonDecisionPolicy(_MethodologyModel):
    primary_metric: str = "balanced_accuracy"
    require_all_runs_completed: bool = True
    invalid_on_missing_metrics: bool = True
    require_permutation_control_pass: bool = False
    permutation_p_value_threshold: float = 0.05
    tie_tolerance: float = 1e-9
    status_on_tie: Literal["inconclusive"] = "inconclusive"
    allow_mixed_methodology_policies: bool = False
    block_on_subgroup_failures: bool = False

    @model_validator(mode="after")
    def _validate_decision_policy(self) -> ComparisonDecisionPolicy:
        if self.primary_metric not in SUPPORTED_CLASSIFICATION_METRICS:
            allowed = ", ".join(sorted(SUPPORTED_CLASSIFICATION_METRICS))
            raise ValueError(
                f"Unsupported comparison decision primary_metric '{self.primary_metric}'. "
                f"Allowed values: {allowed}."
            )
        threshold = float(self.permutation_p_value_threshold)
        if threshold <= 0.0 or threshold > 1.0:
            raise ValueError(
                "comparison decision permutation_p_value_threshold must be in (0.0, 1.0]."
            )
        if float(self.tie_tolerance) < 0.0:
            raise ValueError("comparison decision tie_tolerance must be >= 0.0.")
        return self

