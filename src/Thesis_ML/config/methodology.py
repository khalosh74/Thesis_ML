from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from Thesis_ML.config.metric_policy import (
    SUPPORTED_CLASSIFICATION_METRICS,
    validate_metric_name,
)

ALLOWED_SUBGROUP_DIMENSIONS = frozenset({"label", "task", "modality", "session", "subject"})
ALLOWED_DATA_BALANCE_AXES = frozenset({"overall", "subject", "session", "task", "modality"})


class _MethodologyModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class MethodologyPolicyName(StrEnum):
    FIXED_BASELINES_ONLY = "fixed_baselines_only"
    GROUPED_NESTED_TUNING = "grouped_nested_tuning"


class ClassWeightPolicy(StrEnum):
    NONE = "none"
    BALANCED = "balanced"


class FeatureQualityPolicy(_MethodologyModel):
    warn_on_any_nonfinite_repair: bool = True
    warn_on_any_all_zero_vector: bool = True
    warn_on_any_constant_vector: bool = True
    fail_on_any_all_zero_vector: bool = True
    fail_on_any_constant_vector: bool = False


class MethodologyPolicy(_MethodologyModel):
    policy_name: MethodologyPolicyName
    class_weight_policy: ClassWeightPolicy = ClassWeightPolicy.NONE
    tuning_enabled: bool = False
    inner_cv_scheme: Literal["grouped_leave_one_group_out"] | None = None
    inner_group_field: str | None = None
    tuning_search_space_id: str | None = None
    tuning_search_space_version: str | None = None
    feature_quality: FeatureQualityPolicy = Field(default_factory=FeatureQualityPolicy)
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
        self.primary_metric = validate_metric_name(self.primary_metric)
        normalized_secondary: list[str] = []
        for metric_name in self.secondary_metrics:
            normalized_secondary.append(validate_metric_name(metric_name))
        self.secondary_metrics = normalized_secondary
        if len(set(self.secondary_metrics)) != len(self.secondary_metrics):
            raise ValueError("metric_policy.secondary_metrics must be unique.")
        if self.primary_metric in set(self.secondary_metrics):
            raise ValueError("metric_policy.secondary_metrics must not include primary_metric.")
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


class ConfidenceIntervalMethod(StrEnum):
    GROUPED_BOOTSTRAP_PERCENTILE = "grouped_bootstrap_percentile"


class PairedComparisonMethod(StrEnum):
    PAIRED_SIGN_FLIP_PERMUTATION = "paired_sign_flip_permutation"


class EvidenceRunRole(StrEnum):
    PRIMARY = "primary"
    UNTUNED_BASELINE = "untuned_baseline"


class RepeatEvaluationPolicy(_MethodologyModel):
    repeat_count: int = 1
    seed_stride: int = 1000

    @model_validator(mode="after")
    def _validate_repeat_policy(self) -> RepeatEvaluationPolicy:
        if int(self.repeat_count) <= 0:
            raise ValueError("evidence_policy.repeat_evaluation.repeat_count must be > 0.")
        if int(self.seed_stride) <= 0:
            raise ValueError("evidence_policy.repeat_evaluation.seed_stride must be > 0.")
        return self


class ConfidenceIntervalPolicy(_MethodologyModel):
    method: ConfidenceIntervalMethod = ConfidenceIntervalMethod.GROUPED_BOOTSTRAP_PERCENTILE
    confidence_level: float = 0.95
    n_bootstrap: int = 1000
    seed: int = 2026

    @model_validator(mode="after")
    def _validate_confidence_intervals(self) -> ConfidenceIntervalPolicy:
        level = float(self.confidence_level)
        if level <= 0.0 or level >= 1.0:
            raise ValueError(
                "evidence_policy.confidence_intervals.confidence_level must be in (0.0, 1.0)."
            )
        if int(self.n_bootstrap) <= 0:
            raise ValueError(
                "evidence_policy.confidence_intervals.n_bootstrap must be > 0."
            )
        if int(self.seed) < 0:
            raise ValueError("evidence_policy.confidence_intervals.seed must be >= 0.")
        return self


class PairedComparisonPolicy(_MethodologyModel):
    method: PairedComparisonMethod = PairedComparisonMethod.PAIRED_SIGN_FLIP_PERMUTATION
    n_permutations: int = 5000
    alpha: float = 0.05
    require_significant_win: bool = False

    @model_validator(mode="after")
    def _validate_paired_comparison(self) -> PairedComparisonPolicy:
        if int(self.n_permutations) <= 0:
            raise ValueError("evidence_policy.paired_comparisons.n_permutations must be > 0.")
        alpha = float(self.alpha)
        if alpha <= 0.0 or alpha > 1.0:
            raise ValueError("evidence_policy.paired_comparisons.alpha must be in (0.0, 1.0].")
        return self


class PermutationEvidencePolicy(_MethodologyModel):
    alpha: float = 0.05
    minimum_permutations: int = 100
    require_pass_for_validity: bool = False

    @model_validator(mode="after")
    def _validate_permutation_evidence(self) -> PermutationEvidencePolicy:
        alpha = float(self.alpha)
        if alpha <= 0.0 or alpha > 1.0:
            raise ValueError("evidence_policy.permutation.alpha must be in (0.0, 1.0].")
        if int(self.minimum_permutations) < 0:
            raise ValueError(
                "evidence_policy.permutation.minimum_permutations must be >= 0."
            )
        return self


class CalibrationPolicy(_MethodologyModel):
    enabled: bool = True
    n_bins: int = 10
    require_probabilities_for_validity: bool = False

    @model_validator(mode="after")
    def _validate_calibration(self) -> CalibrationPolicy:
        if int(self.n_bins) <= 1:
            raise ValueError("evidence_policy.calibration.n_bins must be > 1.")
        return self


class RequiredEvidencePackagePolicy(_MethodologyModel):
    require_dummy_baseline: bool = True
    require_permutation_control: bool = True
    require_untuned_baseline_if_tuning: bool = True


class EvidencePolicy(_MethodologyModel):
    repeat_evaluation: RepeatEvaluationPolicy = Field(default_factory=RepeatEvaluationPolicy)
    confidence_intervals: ConfidenceIntervalPolicy = Field(default_factory=ConfidenceIntervalPolicy)
    paired_comparisons: PairedComparisonPolicy = Field(default_factory=PairedComparisonPolicy)
    permutation: PermutationEvidencePolicy = Field(default_factory=PermutationEvidencePolicy)
    calibration: CalibrationPolicy = Field(default_factory=CalibrationPolicy)
    required_package: RequiredEvidencePackagePolicy = Field(
        default_factory=RequiredEvidencePackagePolicy
    )


class ClassBalanceDataPolicy(_MethodologyModel):
    enabled: bool = True
    axes: list[str] = Field(
        default_factory=lambda: ["overall", "subject", "session", "task", "modality"]
    )
    min_class_fraction_warning: float | None = 0.05
    min_class_fraction_blocking: float | None = None

    @model_validator(mode="after")
    def _validate_balance_policy(self) -> ClassBalanceDataPolicy:
        if len(set(self.axes)) != len(self.axes):
            raise ValueError("data_policy.class_balance.axes must be unique.")
        invalid = [value for value in self.axes if value not in ALLOWED_DATA_BALANCE_AXES]
        if invalid:
            allowed = ", ".join(sorted(ALLOWED_DATA_BALANCE_AXES))
            raise ValueError(
                "Unsupported data_policy.class_balance.axes values: "
                + ", ".join(sorted(set(invalid)))
                + f". Allowed values: {allowed}."
            )
        if self.min_class_fraction_warning is not None:
            value = float(self.min_class_fraction_warning)
            if value < 0.0 or value > 1.0:
                raise ValueError(
                    "data_policy.class_balance.min_class_fraction_warning must be in [0.0, 1.0]."
                )
        if self.min_class_fraction_blocking is not None:
            value = float(self.min_class_fraction_blocking)
            if value < 0.0 or value > 1.0:
                raise ValueError(
                    "data_policy.class_balance.min_class_fraction_blocking must be in [0.0, 1.0]."
                )
        if (
            self.min_class_fraction_warning is not None
            and self.min_class_fraction_blocking is not None
            and float(self.min_class_fraction_blocking) > float(self.min_class_fraction_warning)
        ):
            raise ValueError(
                "data_policy.class_balance.min_class_fraction_blocking must be <= "
                "min_class_fraction_warning."
            )
        return self


class MissingnessDataPolicy(_MethodologyModel):
    enabled: bool = True
    max_missing_fraction_warning: float | None = 0.1
    max_missing_fraction_blocking: float | None = None

    @model_validator(mode="after")
    def _validate_missingness_policy(self) -> MissingnessDataPolicy:
        if self.max_missing_fraction_warning is not None:
            value = float(self.max_missing_fraction_warning)
            if value < 0.0 or value > 1.0:
                raise ValueError(
                    "data_policy.missingness.max_missing_fraction_warning must be in [0.0, 1.0]."
                )
        if self.max_missing_fraction_blocking is not None:
            value = float(self.max_missing_fraction_blocking)
            if value < 0.0 or value > 1.0:
                raise ValueError(
                    "data_policy.missingness.max_missing_fraction_blocking must be in [0.0, 1.0]."
                )
        if (
            self.max_missing_fraction_warning is not None
            and self.max_missing_fraction_blocking is not None
            and float(self.max_missing_fraction_blocking)
            < float(self.max_missing_fraction_warning)
        ):
            raise ValueError(
                "data_policy.missingness.max_missing_fraction_blocking must be >= "
                "max_missing_fraction_warning."
            )
        return self


class LeakageDataPolicy(_MethodologyModel):
    enabled: bool = True
    fail_on_duplicate_sample_id: bool = True
    warn_on_duplicate_beta_path: bool = True
    fail_on_duplicate_beta_path: bool = True
    warn_on_duplicate_beta_content_hash: bool = True
    fail_on_duplicate_beta_content_hash: bool = True
    fail_on_subject_overlap_for_transfer: bool = True
    fail_on_cv_group_overlap: bool = True


class ExternalDatasetCompatibilitySpec(_MethodologyModel):
    dataset_id: str = Field(min_length=1)
    index_csv: str = Field(min_length=1)
    target_column: str | None = None
    required_columns: list[str] = Field(default_factory=list)
    required: bool = False
    notes: str | None = None

    @model_validator(mode="after")
    def _validate_external_dataset(self) -> ExternalDatasetCompatibilitySpec:
        if len(set(self.required_columns)) != len(self.required_columns):
            raise ValueError(
                "data_policy.external_validation.datasets[].required_columns must be unique."
            )
        return self


class ExternalValidationDataPolicy(_MethodologyModel):
    enabled: bool = False
    mode: Literal["compatibility_only"] = "compatibility_only"
    require_compatible: bool = False
    require_for_official_runs: bool = False
    datasets: list[ExternalDatasetCompatibilitySpec] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_external_validation_policy(self) -> ExternalValidationDataPolicy:
        if self.require_for_official_runs and not self.enabled:
            raise ValueError(
                "data_policy.external_validation.require_for_official_runs=true requires enabled=true."
            )
        if self.enabled and self.require_for_official_runs and not self.datasets:
            raise ValueError(
                "data_policy.external_validation enabled+required requires at least one dataset."
            )
        dataset_ids = [dataset.dataset_id for dataset in self.datasets]
        if len(set(dataset_ids)) != len(dataset_ids):
            raise ValueError(
                "data_policy.external_validation.datasets contains duplicate dataset_id values."
            )
        return self


class DataPolicy(_MethodologyModel):
    class_balance: ClassBalanceDataPolicy = Field(default_factory=ClassBalanceDataPolicy)
    missingness: MissingnessDataPolicy = Field(default_factory=MissingnessDataPolicy)
    leakage: LeakageDataPolicy = Field(default_factory=LeakageDataPolicy)
    external_validation: ExternalValidationDataPolicy = Field(
        default_factory=ExternalValidationDataPolicy
    )
    required_index_columns: list[str] = Field(default_factory=list)
    intended_use: str = (
        "Official confirmatory/comparison evaluation under locked protocol/comparison contracts."
    )
    not_intended_use: list[str] = Field(
        default_factory=lambda: [
            "Exploratory hypothesis generation from official artifacts.",
            "Causal, clinical, or localization claims unsupported by the protocol.",
        ]
    )
    known_limitations: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_data_policy(self) -> DataPolicy:
        if len(set(self.required_index_columns)) != len(self.required_index_columns):
            raise ValueError("data_policy.required_index_columns must be unique.")
        if not self.intended_use.strip():
            raise ValueError("data_policy.intended_use must be non-empty.")
        return self
