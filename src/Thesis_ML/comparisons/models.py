from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from Thesis_ML.config.framework_mode import FrameworkMode
from Thesis_ML.config.methodology import (
    ClassWeightPolicy,
    ComparisonDecisionPolicy,
    DataPolicy,
    EvidencePolicy,
    EvidenceRunRole,
    MethodologyPolicy,
    MethodologyPolicyName,
    MetricPolicy,
    SubgroupReportingPolicy,
)
from Thesis_ML.config.metric_policy import SUPPORTED_CLASSIFICATION_METRICS, validate_metric_name
from Thesis_ML.experiments.model_catalog import (
    ModelCostTier,
    get_model_cost_entry,
    projected_runtime_seconds,
)
from Thesis_ML.experiments.model_admission import (
    admitted_models_for_framework,
    model_allowed_in_locked_comparison,
)
from Thesis_ML.experiments.model_registry import get_model_spec, registered_model_names
from Thesis_ML.experiments.run_states import (
    RUN_STATUS_COMPLETED_LEGACY,
    RUN_STATUS_FAILED,
    RUN_STATUS_SKIPPED_DUE_TO_POLICY,
    RUN_STATUS_SUCCESS,
    RUN_STATUS_TIMED_OUT,
    normalize_run_status,
)
from Thesis_ML.protocols.models import SUPPORTED_CV_MODES

COMPARISON_SCHEMA_VERSION = "comparison-spec-v1"
SUPPORTED_COMPARISON_SCHEMA_VERSIONS = frozenset({COMPARISON_SCHEMA_VERSION})
REQUIRED_COMPARISON_ARTIFACTS = (
    "comparison.json",
    "compiled_comparison_manifest.json",
    "comparison_summary.json",
    "comparison_decision.json",
    "execution_status.json",
    "repeated_run_metrics.csv",
    "repeated_run_summary.json",
    "confidence_intervals.json",
    "metric_intervals.csv",
    "paired_model_comparisons.json",
    "paired_model_comparisons.csv",
    "report_index.csv",
)
REQUIRED_COMPARISON_RUN_ARTIFACTS = (
    "config.json",
    "metrics.json",
    "dataset_card.json",
    "dataset_card.md",
    "dataset_summary.json",
    "dataset_summary.csv",
    "data_quality_report.json",
    "class_balance_report.csv",
    "missingness_report.csv",
    "leakage_audit.json",
    "external_dataset_card.json",
    "external_dataset_summary.json",
    "external_validation_compatibility.json",
    "fold_metrics.csv",
    "fold_splits.csv",
    "predictions.csv",
    "subgroup_metrics.json",
    "subgroup_metrics.csv",
    "tuning_summary.json",
    "best_params_per_fold.csv",
    "spatial_compatibility_report.json",
    "interpretability_summary.json",
    "calibration_summary.json",
    "calibration_table.csv",
)
SUPPORTED_PRIMARY_METRICS = SUPPORTED_CLASSIFICATION_METRICS
ALL_MODEL_NAMES = registered_model_names()
_OFFICIAL_LOCKED_COMPARISON_MODEL_NAMES = admitted_models_for_framework(
    FrameworkMode.LOCKED_COMPARISON
)


def _reject_exploratory_only_official_model(*, model_name: str, field_name: str) -> None:
    normalized_model = str(model_name).strip().lower()
    if model_allowed_in_locked_comparison(normalized_model):
        return
    allowed = ", ".join(sorted(_OFFICIAL_LOCKED_COMPARISON_MODEL_NAMES))
    raise ValueError(
        f"{field_name} model '{model_name}' is exploratory-only and not admitted for "
        f"locked-comparison official execution. Allowed official models: {allowed}."
    )


class _ComparisonModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    _source_config_path: str | None = PrivateAttr(default=None)
    _source_config_identity: dict[str, Any] | None = PrivateAttr(default=None)


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
            raise ValueError(
                "control_policy.n_permutations must be > 0 when permutation_enabled is true."
            )
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
                _reject_exploratory_only_official_model(
                    model_name=model_name,
                    field_name="interpretability_policy.allowed_models",
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
        _reject_exploratory_only_official_model(
            model_name=self.model,
            field_name=f"variant '{self.variant_id}'",
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
            "class_weight_policy",
            "tuning_enabled",
            "model_cost_tier",
            "projected_runtime_seconds",
            "evidence_run_role",
            "repeat_id",
            "repeat_count",
            "base_run_id",
            "data_policy_effective",
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
        missing_run = [
            value
            for value in (
                "config.json",
                "metrics.json",
                "dataset_card.json",
                "dataset_summary.json",
                "data_quality_report.json",
                "leakage_audit.json",
            )
            if value not in self.required_run_artifacts
        ]
        if missing_run:
            raise ValueError(
                "required_run_artifacts is missing required entries: "
                + ", ".join(sorted(missing_run))
            )
        for key in (
            "framework_mode",
            "canonical_run",
            "model_cost_tier",
            "projected_runtime_seconds",
            "data_policy_effective",
            "comparison_id",
            "comparison_version",
            "comparison_variant_id",
        ):
            if key not in self.required_run_metadata_fields:
                raise ValueError("required_run_metadata_fields is missing required key: " + key)
        return self


class ComparisonCostPolicy(_ComparisonModel):
    explicit_benchmark_expensive_models: list[str] = Field(default_factory=list)
    max_projected_runtime_seconds_per_run: int = 120 * 60

    @model_validator(mode="after")
    def _validate_cost_policy(self) -> ComparisonCostPolicy:
        if len(set(self.explicit_benchmark_expensive_models)) != len(
            self.explicit_benchmark_expensive_models
        ):
            raise ValueError("cost_policy.explicit_benchmark_expensive_models must be unique.")
        if int(self.max_projected_runtime_seconds_per_run) <= 0:
            raise ValueError("cost_policy.max_projected_runtime_seconds_per_run must be > 0.")
        supported_models = set(ALL_MODEL_NAMES)
        for model_name in self.explicit_benchmark_expensive_models:
            if model_name not in supported_models:
                allowed = ", ".join(sorted(supported_models))
                raise ValueError(
                    "cost_policy.explicit_benchmark_expensive_models references unsupported "
                    f"model '{model_name}'. Allowed values: {allowed}."
                )
            _reject_exploratory_only_official_model(
                model_name=model_name,
                field_name="cost_policy.explicit_benchmark_expensive_models",
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
    cost_policy: ComparisonCostPolicy = Field(default_factory=ComparisonCostPolicy)
    subgroup_reporting_policy: SubgroupReportingPolicy = Field(
        default_factory=SubgroupReportingPolicy
    )
    data_policy: DataPolicy = Field(default_factory=DataPolicy)
    decision_policy: ComparisonDecisionPolicy = Field(default_factory=ComparisonDecisionPolicy)
    evidence_policy: EvidencePolicy
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
        if (
            self.evidence_policy.required_package.require_dummy_baseline
            and not self.control_policy.dummy_baseline_enabled
        ):
            raise ValueError(
                "evidence_policy.required_package.require_dummy_baseline=true "
                "requires control_policy.dummy_baseline_enabled=true."
            )
        if (
            self.evidence_policy.required_package.require_permutation_control
            and not self.control_policy.permutation_enabled
        ):
            raise ValueError(
                "evidence_policy.required_package.require_permutation_control=true "
                "requires control_policy.permutation_enabled=true."
            )
        if self.control_policy.permutation_enabled and int(
            self.control_policy.n_permutations
        ) < int(self.evidence_policy.permutation.minimum_permutations):
            raise ValueError(
                "control_policy.n_permutations must be >= "
                "evidence_policy.permutation.minimum_permutations."
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

        explicitly_allowed_expensive = set(self.cost_policy.explicit_benchmark_expensive_models)
        max_runtime = int(self.cost_policy.max_projected_runtime_seconds_per_run)
        for variant in self.allowed_variants:
            model_spec = get_model_spec(variant.model)
            if (
                str(self.methodology_policy.class_weight_policy.value)
                not in set(model_spec.supported_class_weight_policies)
            ):
                allowed_class_weight = ", ".join(model_spec.supported_class_weight_policies)
                raise ValueError(
                    f"comparison methodology class_weight_policy='{self.methodology_policy.class_weight_policy.value}' "
                    f"is not supported by model '{variant.model}'. Allowed values: {allowed_class_weight}."
                )
            catalog_entry = get_model_cost_entry(variant.model)
            expensive_model_requires_explicit_allow = bool(
                catalog_entry.cost_tier == ModelCostTier.BENCHMARK_EXPENSIVE
                or catalog_entry.requires_explicit_comparison_spec
            )
            if (
                expensive_model_requires_explicit_allow
                and variant.model not in explicitly_allowed_expensive
            ):
                raise ValueError(
                    "comparison cost_policy requires explicit_benchmark_expensive_models to "
                    f"include model '{variant.model}' (variant_id='{variant.variant_id}')."
                )
            projected_runtime = projected_runtime_seconds(
                model_name=variant.model,
                framework_mode=FrameworkMode.LOCKED_COMPARISON,
                methodology_policy=self.methodology_policy.policy_name,
                tuning_enabled=bool(self.methodology_policy.tuning_enabled),
            )
            if int(projected_runtime) > max_runtime:
                raise ValueError(
                    "comparison cost_policy rejected variant "
                    f"'{variant.variant_id}' (model '{variant.model}'): projected_runtime_seconds="
                    f"{projected_runtime} exceeds max_projected_runtime_seconds_per_run="
                    f"{max_runtime}."
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
    model_cost_tier: ModelCostTier = ModelCostTier.OFFICIAL_FAST
    projected_runtime_seconds: int = 1
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
    repeat_id: int = 1
    repeat_count: int = 1
    base_run_id: str = Field(min_length=1)
    evidence_run_role: EvidenceRunRole = EvidenceRunRole.PRIMARY
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
        if self.model not in set(ALL_MODEL_NAMES):
            allowed = ", ".join(sorted(ALL_MODEL_NAMES))
            raise ValueError(
                f"CompiledComparisonRunSpec model '{self.model}' is unsupported. "
                f"Allowed values: {allowed}."
            )
        _reject_exploratory_only_official_model(
            model_name=self.model,
            field_name="CompiledComparisonRunSpec",
        )
        model_spec = get_model_spec(self.model)
        if str(self.class_weight_policy.value) not in set(model_spec.supported_class_weight_policies):
            allowed = ", ".join(model_spec.supported_class_weight_policies)
            raise ValueError(
                f"CompiledComparisonRunSpec '{self.run_id}' class_weight_policy="
                f"'{self.class_weight_policy.value}' is not supported by model "
                f"'{self.model}'. Allowed values: {allowed}."
            )
        catalog_entry = get_model_cost_entry(self.model)
        if self.model_cost_tier != catalog_entry.cost_tier:
            raise ValueError(
                f"CompiledComparisonRunSpec '{self.run_id}' model_cost_tier="
                f"'{self.model_cost_tier.value}' does not match catalog tier "
                f"'{catalog_entry.cost_tier.value}' for model '{self.model}'."
            )
        expected_projected_runtime = projected_runtime_seconds(
            model_name=self.model,
            framework_mode=FrameworkMode.LOCKED_COMPARISON,
            methodology_policy=self.methodology_policy_name,
            tuning_enabled=bool(self.tuning_enabled),
        )
        if int(self.projected_runtime_seconds) <= 0:
            raise ValueError("CompiledComparisonRunSpec.projected_runtime_seconds must be > 0.")
        if int(self.projected_runtime_seconds) != int(expected_projected_runtime):
            raise ValueError(
                f"CompiledComparisonRunSpec '{self.run_id}' projected_runtime_seconds="
                f"{self.projected_runtime_seconds} does not match expected "
                f"{expected_projected_runtime} for model '{self.model}'."
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
        if self.evidence_run_role == EvidenceRunRole.UNTUNED_BASELINE:
            if self.methodology_policy_name != MethodologyPolicyName.GROUPED_NESTED_TUNING:
                raise ValueError(
                    "evidence_run_role='untuned_baseline' requires grouped_nested_tuning."
                )
            if self.tuning_enabled:
                raise ValueError(
                    "evidence_run_role='untuned_baseline' requires tuning_enabled=false."
                )
            if any(
                value is not None
                for value in (
                    self.tuning_search_space_id,
                    self.tuning_search_space_version,
                    self.tuning_inner_cv_scheme,
                    self.tuning_inner_group_field,
                )
            ):
                raise ValueError(
                    "evidence_run_role='untuned_baseline' forbids tuning search-space and inner-CV metadata."
                )
        else:
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
        if int(self.repeat_count) <= 0:
            raise ValueError("CompiledComparisonRunSpec.repeat_count must be > 0.")
        if int(self.repeat_id) <= 0:
            raise ValueError("CompiledComparisonRunSpec.repeat_id must be > 0.")
        if int(self.repeat_id) > int(self.repeat_count):
            raise ValueError("CompiledComparisonRunSpec.repeat_id must be <= repeat_count.")
        if not str(self.base_run_id).strip():
            raise ValueError("CompiledComparisonRunSpec.base_run_id must be non-empty.")
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
    data_policy: DataPolicy
    decision_policy: ComparisonDecisionPolicy
    evidence_policy: EvidencePolicy
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
            "class_weight_policy",
            "tuning_enabled",
            "model_cost_tier",
            "projected_runtime_seconds",
            "evidence_run_role",
            "repeat_id",
            "repeat_count",
            "base_run_id",
            "data_policy_effective",
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
        for key in (
            "framework_mode",
            "canonical_run",
            "model_cost_tier",
            "projected_runtime_seconds",
            "data_policy_effective",
            "comparison_id",
            "comparison_version",
            "comparison_variant_id",
        ):
            if key not in self.required_run_metadata_fields:
                raise ValueError(
                    "CompiledComparisonManifest.required_run_metadata_fields is missing required key: "
                    + key
                )
        return self


class ComparisonRunResult(_ComparisonModel):
    run_id: str = Field(min_length=1)
    framework_mode: Literal["locked_comparison"] = FrameworkMode.LOCKED_COMPARISON.value
    comparison_id: str = Field(min_length=1)
    comparison_version: str = Field(min_length=1)
    variant_id: str = Field(min_length=1)
    status: Literal[
        "planned",
        "success",
        "failed",
        "timed_out",
        "skipped_due_to_policy",
        "completed",
    ]
    report_dir: str | None = None
    config_path: str | None = None
    metrics_path: str | None = None
    error: str | None = None
    error_code: str | None = None
    error_type: str | None = None
    failure_stage: str | None = None
    error_details: dict[str, Any] | None = None
    timeout_seconds: float | None = None
    elapsed_seconds: float | None = None
    timeout_diagnostics_path: str | None = None
    policy_reason: str | None = None
    metrics: dict[str, float | int | str | bool | None | dict[str, Any]] | None = None
    compute_policy: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _validate_result(self) -> ComparisonRunResult:
        if self.framework_mode != FrameworkMode.LOCKED_COMPARISON.value:
            raise ValueError("ComparisonRunResult.framework_mode must be 'locked_comparison'.")
        normalized_status = normalize_run_status(self.status)
        if normalized_status == RUN_STATUS_SUCCESS and self.status == RUN_STATUS_COMPLETED_LEGACY:
            self.status = RUN_STATUS_SUCCESS

        if normalized_status in {RUN_STATUS_FAILED, RUN_STATUS_TIMED_OUT}:
            if self.error is None:
                raise ValueError(
                    "ComparisonRunResult.error is required when status is failed or timed_out."
                )
            if self.error_code is None:
                raise ValueError(
                    "ComparisonRunResult.error_code is required when status is failed or timed_out."
                )
            if self.error_type is None:
                raise ValueError(
                    "ComparisonRunResult.error_type is required when status is failed or timed_out."
                )
            if self.failure_stage is None:
                raise ValueError(
                    "ComparisonRunResult.failure_stage is required when status is failed or timed_out."
                )
        if normalized_status == RUN_STATUS_TIMED_OUT:
            if self.timeout_seconds is None:
                raise ValueError(
                    "ComparisonRunResult.timeout_seconds is required when status='timed_out'."
                )
            if self.elapsed_seconds is None:
                raise ValueError(
                    "ComparisonRunResult.elapsed_seconds is required when status='timed_out'."
                )
            if self.timeout_diagnostics_path is None:
                raise ValueError(
                    "ComparisonRunResult.timeout_diagnostics_path is required when status='timed_out'."
                )
        if normalized_status not in {RUN_STATUS_FAILED, RUN_STATUS_TIMED_OUT}:
            for field_name in (
                "error",
                "error_code",
                "error_type",
                "failure_stage",
                "error_details",
            ):
                if getattr(self, field_name) is not None:
                    raise ValueError(
                        f"ComparisonRunResult.{field_name} must be null unless status is failed or timed_out."
                    )
        if normalized_status != RUN_STATUS_TIMED_OUT:
            for field_name in ("timeout_seconds", "elapsed_seconds", "timeout_diagnostics_path"):
                if getattr(self, field_name) is not None:
                    raise ValueError(
                        f"ComparisonRunResult.{field_name} must be null unless status='timed_out'."
                    )
        if normalized_status != RUN_STATUS_SKIPPED_DUE_TO_POLICY and self.policy_reason is not None:
            raise ValueError(
                "ComparisonRunResult.policy_reason must be null unless status='skipped_due_to_policy'."
            )
        if (
            normalized_status == RUN_STATUS_SKIPPED_DUE_TO_POLICY
            and not str(self.policy_reason or "").strip()
        ):
            raise ValueError(
                "ComparisonRunResult.policy_reason is required when status='skipped_due_to_policy'."
            )
        return self
