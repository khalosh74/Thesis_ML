from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from Thesis_ML.config.framework_mode import FrameworkMode
from Thesis_ML.config.methodology import (
    ClassWeightPolicy,
    ComparisonDecisionPolicy,
    MethodologyPolicy,
    MethodologyPolicyName,
    MetricPolicy,
    SubgroupReportingPolicy,
)
from Thesis_ML.config.metric_policy import SUPPORTED_CLASSIFICATION_METRICS, validate_metric_name
from Thesis_ML.experiments.model_factory import ALL_MODEL_NAMES
from Thesis_ML.protocols.models import SUPPORTED_CV_MODES

COMPARISON_SCHEMA_VERSION = "comparison-spec-v1"
SUPPORTED_COMPARISON_SCHEMA_VERSIONS = frozenset({COMPARISON_SCHEMA_VERSION})
REQUIRED_COMPARISON_ARTIFACTS = (
    "comparison.json",
    "compiled_comparison_manifest.json",
    "comparison_summary.json",
    "comparison_decision.json",
    "execution_status.json",
    "report_index.csv",
)
REQUIRED_COMPARISON_RUN_ARTIFACTS = (
    "config.json",
    "metrics.json",
    "fold_metrics.csv",
    "fold_splits.csv",
    "predictions.csv",
    "subgroup_metrics.json",
    "subgroup_metrics.csv",
    "tuning_summary.json",
    "best_params_per_fold.csv",
    "spatial_compatibility_report.json",
    "interpretability_summary.json",
)
SUPPORTED_PRIMARY_METRICS = SUPPORTED_CLASSIFICATION_METRICS


class _ComparisonModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class ComparisonStatus(StrEnum):
    DRAFT = "draft"
    LOCKED = "locked"
    EXECUTED = "executed"
    RETIRED = "retired"


class ComparisonDecisionStatus(StrEnum):
    WINNER_SELECTED = "winner_selected"
    INCONCLUSIVE = "inconclusive"
    INVALID_COMPARISON = "invalid_comparison"


class SubjectSource(StrEnum):
    ALL_FROM_INDEX = "all_from_index"
    EXPLICIT = "explicit"


class TransferPairSource(StrEnum):
    ALL_ORDERED_PAIRS_FROM_INDEX = "all_ordered_pairs_from_index"
    EXPLICIT = "explicit"


class ComparisonSeedPolicy(_ComparisonModel):
    global_seed: int = 42

    @model_validator(mode="after")
    def _validate_seed(self) -> ComparisonSeedPolicy:
        if int(self.global_seed) < 0:
            raise ValueError("scientific_contract.seed_policy.global_seed must be >= 0.")
        return self


class ComparisonControlPolicy(_ComparisonModel):
    permutation_enabled: bool = False
    permutation_metric: str | None = None
    n_permutations: int = 0
    dummy_baseline_enabled: bool = False

    @model_validator(mode="after")
    def _validate_controls(self) -> ComparisonControlPolicy:
        if self.permutation_metric is not None:
            validate_metric_name(self.permutation_metric)
        if self.permutation_enabled and int(self.n_permutations) <= 0:
            raise ValueError("control_policy.n_permutations must be > 0 when permutation_enabled is true.")
        if not self.permutation_enabled and int(self.n_permutations) < 0:
            raise ValueError("control_policy.n_permutations must be >= 0.")
        return self


class ComparisonInterpretabilityPolicy(_ComparisonModel):
    enabled: bool = False
    allowed_models: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_models(self) -> ComparisonInterpretabilityPolicy:
        if self.enabled:
            if not self.allowed_models:
                raise ValueError(
                    "interpretability_policy.allowed_models must be non-empty when interpretability is enabled."
                )
            supported = set(ALL_MODEL_NAMES)
            for model_name in self.allowed_models:
                if model_name not in supported:
                    allowed = ", ".join(sorted(supported))
                    raise ValueError(
                        f"Unsupported interpretability model '{model_name}'. Allowed values: {allowed}."
                    )
        return self


class ComparisonSubjectPolicy(_ComparisonModel):
    source: SubjectSource = SubjectSource.ALL_FROM_INDEX
    subjects: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_subjects(self) -> ComparisonSubjectPolicy:
        if self.source == SubjectSource.EXPLICIT and not self.subjects:
            raise ValueError("subject_policy.source='explicit' requires non-empty subjects.")
        return self


class ComparisonTransferPair(_ComparisonModel):
    train_subject: str = Field(min_length=1)
    test_subject: str = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_pair(self) -> ComparisonTransferPair:
        if self.train_subject == self.test_subject:
            raise ValueError("train_subject and test_subject must differ.")
        return self


class ComparisonTransferPolicy(_ComparisonModel):
    source: TransferPairSource = TransferPairSource.ALL_ORDERED_PAIRS_FROM_INDEX
    pairs: list[ComparisonTransferPair] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_pairs(self) -> ComparisonTransferPolicy:
        if self.source == TransferPairSource.EXPLICIT and not self.pairs:
            raise ValueError("transfer_policy.source='explicit' requires non-empty pairs.")
        return self


class ComparisonScientificContract(_ComparisonModel):
    target: str = Field(min_length=1)
    split_mode: Literal["within_subject_loso_session", "frozen_cross_person_transfer"]
    grouping_policy: str = Field(min_length=1)
    seed_policy: ComparisonSeedPolicy = Field(default_factory=ComparisonSeedPolicy)
    subject_policy: ComparisonSubjectPolicy = Field(default_factory=ComparisonSubjectPolicy)
    transfer_policy: ComparisonTransferPolicy = Field(default_factory=ComparisonTransferPolicy)
    filter_task: str | None = None
    filter_modality: str | None = None

    @model_validator(mode="after")
    def _validate_contract(self) -> ComparisonScientificContract:
        if self.split_mode not in SUPPORTED_CV_MODES:
            allowed = ", ".join(sorted(SUPPORTED_CV_MODES))
            raise ValueError(
                f"Unsupported split_mode '{self.split_mode}'. Allowed values: {allowed}."
            )
        if self.split_mode == "within_subject_loso_session":
            if self.transfer_policy.pairs:
                raise ValueError(
                    "split_mode='within_subject_loso_session' cannot define transfer pairs."
                )
        if self.split_mode == "frozen_cross_person_transfer":
            if self.subject_policy.subjects:
                raise ValueError(
                    "split_mode='frozen_cross_person_transfer' cannot define subject_policy.subjects."
                )
        return self


class ComparisonVariant(_ComparisonModel):
    variant_id: str = Field(min_length=1)
    model: str = Field(min_length=1)
    claim_ids: list[str] = Field(min_length=1)
    notes: str | None = None

    @model_validator(mode="after")
    def _validate_variant(self) -> ComparisonVariant:
        if self.model not in set(ALL_MODEL_NAMES):
            allowed = ", ".join(sorted(ALL_MODEL_NAMES))
            raise ValueError(
                f"Unsupported variant model '{self.model}'. Allowed values: {allowed}."
            )
        if len(set(self.claim_ids)) != len(self.claim_ids):
            raise ValueError(f"Variant '{self.variant_id}' contains duplicate claim_ids.")
        return self


class ComparisonArtifactContract(_ComparisonModel):
    required_comparison_artifacts: list[str] = Field(
        default_factory=lambda: list(REQUIRED_COMPARISON_ARTIFACTS)
    )
    required_run_artifacts: list[str] = Field(
        default_factory=lambda: list(REQUIRED_COMPARISON_RUN_ARTIFACTS)
    )
    required_run_metadata_fields: list[str] = Field(
        default_factory=lambda: [
            "framework_mode",
            "canonical_run",
            "methodology_policy_name",
            "comparison_id",
            "comparison_version",
            "comparison_variant_id",
        ]
    )

    @model_validator(mode="after")
    def _validate_contract(self) -> ComparisonArtifactContract:
        missing = [
            value
            for value in REQUIRED_COMPARISON_ARTIFACTS
            if value not in self.required_comparison_artifacts
        ]
        if missing:
            raise ValueError(
                "required_comparison_artifacts is missing entries: " + ", ".join(missing)
            )
        return self


class ComparisonSpec(_ComparisonModel):
    comparison_schema_version: str = COMPARISON_SCHEMA_VERSION
    framework_mode: Literal["locked_comparison"] = FrameworkMode.LOCKED_COMPARISON.value
    comparison_id: str = Field(min_length=1)
    comparison_version: str = Field(min_length=1)
    status: ComparisonStatus
    description: str = Field(min_length=1)
    comparison_dimension: str = Field(min_length=1)
    scientific_contract: ComparisonScientificContract
    methodology_policy: MethodologyPolicy
    metric_policy: MetricPolicy = Field(default_factory=MetricPolicy)
    control_policy: ComparisonControlPolicy = Field(default_factory=ComparisonControlPolicy)
    subgroup_reporting_policy: SubgroupReportingPolicy = Field(
        default_factory=SubgroupReportingPolicy
    )
    decision_policy: ComparisonDecisionPolicy = Field(default_factory=ComparisonDecisionPolicy)
    interpretability_policy: ComparisonInterpretabilityPolicy = Field(
        default_factory=ComparisonInterpretabilityPolicy
    )
    allowed_variants: list[ComparisonVariant] = Field(min_length=1)
    artifact_contract: ComparisonArtifactContract = Field(
        default_factory=ComparisonArtifactContract
    )

    @model_validator(mode="after")
    def _validate_spec(self) -> ComparisonSpec:
        if self.comparison_schema_version not in SUPPORTED_COMPARISON_SCHEMA_VERSIONS:
            allowed = ", ".join(sorted(SUPPORTED_COMPARISON_SCHEMA_VERSIONS))
            raise ValueError(
                f"Unsupported comparison_schema_version '{self.comparison_schema_version}'. "
                f"Allowed values: {allowed}."
            )
        if self.framework_mode != FrameworkMode.LOCKED_COMPARISON.value:
            raise ValueError("ComparisonSpec.framework_mode must be 'locked_comparison'.")
        variant_ids = [variant.variant_id for variant in self.allowed_variants]
        if len(set(variant_ids)) != len(variant_ids):
            raise ValueError(
                "ComparisonSpec.allowed_variants contains duplicate variant_id values."
            )

        validate_metric_name(self.metric_policy.primary_metric)
        for metric_name in self.metric_policy.secondary_metrics:
            validate_metric_name(metric_name)
        if self.decision_policy.primary_metric != self.metric_policy.primary_metric:
            raise ValueError(
                "decision_policy.primary_metric must match metric_policy.primary_metric "
                "for locked comparisons."
            )

        resolved_permutation_metric = (
            self.control_policy.permutation_metric or self.metric_policy.primary_metric
        )
        if resolved_permutation_metric != self.metric_policy.primary_metric:
            raise ValueError(
                "control_policy.permutation_metric must match metric_policy.primary_metric for locked comparisons."
            )

        if self.control_policy.dummy_baseline_enabled:
            if not any(variant.model == "dummy" for variant in self.allowed_variants):
                raise ValueError(
                    "control_policy.dummy_baseline_enabled=true requires at least one 'dummy' variant."
                )

        if (
            self.methodology_policy.policy_name == MethodologyPolicyName.FIXED_BASELINES_ONLY
            and self.comparison_dimension == "model_family"
            and len({variant.model for variant in self.allowed_variants}) < 2
        ):
            raise ValueError(
                "model_family comparison with fixed_baselines_only requires at least two distinct models."
            )

        return self


class CompiledComparisonRunControls(_ComparisonModel):
    permutation_enabled: bool = False
    permutation_metric: str = "balanced_accuracy"
    n_permutations: int = 0
    dummy_baseline_enabled: bool = False


class CompiledComparisonRunSpec(_ComparisonModel):
    run_id: str = Field(min_length=1)
    framework_mode: Literal["locked_comparison"] = FrameworkMode.LOCKED_COMPARISON.value
    canonical_run: bool = False
    comparison_id: str = Field(min_length=1)
    comparison_version: str = Field(min_length=1)
    variant_id: str = Field(min_length=1)
    claim_ids: list[str] = Field(min_length=1)
    target: str = Field(min_length=1)
    model: str = Field(min_length=1)
    cv_mode: Literal["within_subject_loso_session", "frozen_cross_person_transfer"]
    subject: str | None = None
    train_subject: str | None = None
    test_subject: str | None = None
    filter_task: str | None = None
    filter_modality: str | None = None
    seed: int
    primary_metric: str = "balanced_accuracy"
    controls: CompiledComparisonRunControls = Field(default_factory=CompiledComparisonRunControls)
    interpretability_enabled: bool = False
    methodology_policy_name: MethodologyPolicyName = MethodologyPolicyName.FIXED_BASELINES_ONLY
    class_weight_policy: ClassWeightPolicy = ClassWeightPolicy.NONE
    tuning_enabled: bool = False
    tuning_search_space_id: str | None = None
    tuning_search_space_version: str | None = None
    tuning_inner_cv_scheme: Literal["grouped_leave_one_group_out"] | None = None
    tuning_inner_group_field: str | None = None
    subgroup_reporting_enabled: bool = True
    subgroup_dimensions: list[str] = Field(
        default_factory=lambda: ["label", "task", "modality", "session", "subject"]
    )
    subgroup_min_samples_per_group: int = 1
    artifact_requirements: list[str] = Field(
        default_factory=lambda: list(REQUIRED_COMPARISON_RUN_ARTIFACTS)
    )

    @model_validator(mode="after")
    def _validate_run(self) -> CompiledComparisonRunSpec:
        if self.framework_mode != FrameworkMode.LOCKED_COMPARISON.value:
            raise ValueError(
                f"CompiledComparisonRunSpec '{self.run_id}' must use framework_mode='locked_comparison'."
            )
        if self.canonical_run:
            raise ValueError(
                f"CompiledComparisonRunSpec '{self.run_id}' must set canonical_run=false."
            )
        if self.cv_mode == "within_subject_loso_session" and self.subject is None:
            raise ValueError(f"CompiledComparisonRunSpec '{self.run_id}' requires subject.")
        if self.cv_mode == "frozen_cross_person_transfer":
            if self.train_subject is None or self.test_subject is None:
                raise ValueError(
                    f"CompiledComparisonRunSpec '{self.run_id}' requires train_subject and test_subject."
                )
        validate_metric_name(self.primary_metric)
        if self.controls.permutation_enabled:
            if self.controls.permutation_metric != self.primary_metric:
                raise ValueError(
                    f"CompiledComparisonRunSpec '{self.run_id}' requires controls.permutation_metric "
                    "to match primary_metric for locked comparisons."
                )
        MethodologyPolicy(
            policy_name=self.methodology_policy_name,
            class_weight_policy=self.class_weight_policy,
            tuning_enabled=self.tuning_enabled,
            inner_cv_scheme=self.tuning_inner_cv_scheme,
            inner_group_field=self.tuning_inner_group_field,
            tuning_search_space_id=self.tuning_search_space_id,
            tuning_search_space_version=self.tuning_search_space_version,
        )
        SubgroupReportingPolicy(
            enabled=self.subgroup_reporting_enabled,
            subgroup_dimensions=self.subgroup_dimensions,
            min_samples_per_group=self.subgroup_min_samples_per_group,
        )
        return self


class CompiledComparisonManifest(_ComparisonModel):
    compiled_schema_version: str = "comparison-compiled-v1"
    framework_mode: Literal["locked_comparison"] = FrameworkMode.LOCKED_COMPARISON.value
    comparison_id: str = Field(min_length=1)
    comparison_version: str = Field(min_length=1)
    status: ComparisonStatus
    comparison_dimension: str = Field(min_length=1)
    methodology_policy: MethodologyPolicy
    metric_policy: MetricPolicy
    subgroup_reporting_policy: SubgroupReportingPolicy
    decision_policy: ComparisonDecisionPolicy
    variant_ids: list[str] = Field(min_length=1)
    runs: list[CompiledComparisonRunSpec] = Field(min_length=1)
    claim_to_run_map: dict[str, list[str]]
    required_comparison_artifacts: list[str] = Field(
        default_factory=lambda: list(REQUIRED_COMPARISON_ARTIFACTS)
    )
    required_run_artifacts: list[str] = Field(
        default_factory=lambda: list(REQUIRED_COMPARISON_RUN_ARTIFACTS)
    )
    required_run_metadata_fields: list[str] = Field(
        default_factory=lambda: [
            "framework_mode",
            "canonical_run",
            "methodology_policy_name",
            "comparison_id",
            "comparison_version",
            "comparison_variant_id",
        ]
    )

    @model_validator(mode="after")
    def _validate_manifest(self) -> CompiledComparisonManifest:
        run_ids = [run.run_id for run in self.runs]
        if len(set(run_ids)) != len(run_ids):
            raise ValueError("CompiledComparisonManifest.runs contains duplicate run_id values.")
        if not self.required_run_metadata_fields:
            raise ValueError(
                "CompiledComparisonManifest.required_run_metadata_fields must not be empty."
            )
        return self


class ComparisonRunResult(_ComparisonModel):
    run_id: str = Field(min_length=1)
    framework_mode: Literal["locked_comparison"] = FrameworkMode.LOCKED_COMPARISON.value
    comparison_id: str = Field(min_length=1)
    comparison_version: str = Field(min_length=1)
    variant_id: str = Field(min_length=1)
    status: Literal["planned", "completed", "failed"]
    report_dir: str | None = None
    config_path: str | None = None
    metrics_path: str | None = None
    error: str | None = None
    metrics: dict[str, float | int | str | bool | None | dict[str, Any]] | None = None
