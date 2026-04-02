from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from Thesis_ML.config.framework_mode import FrameworkMode
from Thesis_ML.config.methodology import (
    ClassWeightPolicy,
    DataPolicy,
    EvidencePolicy,
    EvidenceRunRole,
    MethodologyPolicy,
    MethodologyPolicyName,
    MetricPolicy,
    SubgroupReportingPolicy,
)
from Thesis_ML.config.metric_policy import (
    SUPPORTED_CLASSIFICATION_METRICS,
    validate_metric_name,
)
from Thesis_ML.config.schema_versions import (
    SUPPORTED_THESIS_PROTOCOL_SCHEMA_VERSIONS,
    THESIS_PROTOCOL_SCHEMA_VERSION,
)
from Thesis_ML.experiments.model_admission import (
    admitted_models_for_framework,
    model_allowed_in_confirmatory,
)
from Thesis_ML.experiments.model_catalog import (
    ModelCostTier,
    get_model_cost_entry,
    projected_runtime_seconds,
    supported_model_cost_tiers,
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
from Thesis_ML.features.preprocessing import (
    BASELINE_STANDARD_SCALER_RECIPE_ID,
    FEATURE_RECIPE_IDS,
    resolve_feature_recipe_id,
)

SUPPORTED_CV_MODES = frozenset({"within_subject_loso_session", "frozen_cross_person_transfer"})
SUPPORTED_PRIMARY_METRICS = SUPPORTED_CLASSIFICATION_METRICS
REQUIRED_PROTOCOL_ARTIFACTS = (
    "protocol.json",
    "compiled_protocol_manifest.json",
    "claim_to_run_map.json",
    "suite_summary.json",
    "execution_status.json",
    "repeated_run_metrics.csv",
    "repeated_run_summary.json",
    "confidence_intervals.json",
    "metric_intervals.csv",
    "report_index.csv",
    "claim_outcomes.json",
)
REQUIRED_RUN_ARTIFACTS_BASELINE = (
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
    "cv_split_manifest.json",
    "cv_split_manifest.csv",
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
    "feature_qc_summary.json",
    "feature_qc_selected_samples.csv",
    "interpretability_summary.json",
    "calibration_summary.json",
    "calibration_table.csv",
)
ALL_MODEL_NAMES = registered_model_names()
_OFFICIAL_CONFIRMATORY_MODEL_NAMES = admitted_models_for_framework(FrameworkMode.CONFIRMATORY)


def _reject_exploratory_only_official_model(*, model_name: str, field_name: str) -> None:
    normalized_model = str(model_name).strip().lower()
    if model_allowed_in_confirmatory(normalized_model):
        return
    allowed = ", ".join(sorted(_OFFICIAL_CONFIRMATORY_MODEL_NAMES))
    raise ValueError(
        f"{field_name} model '{model_name}' is exploratory-only and not admitted for "
        f"confirmatory protocol execution. Allowed official models: {allowed}."
    )


class _ProtocolModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    _source_config_path: str | None = PrivateAttr(default=None)
    _source_config_identity: dict[str, Any] | None = PrivateAttr(default=None)


class ProtocolStatus(StrEnum):
    DRAFT = "draft"
    LOCKED = "locked"
    RELEASED = "released"


class SuiteType(StrEnum):
    PRIMARY = "primary"
    SECONDARY = "secondary"
    CONTROL = "control"
    SENSITIVITY = "sensitivity"


class SubjectSource(StrEnum):
    ALL_FROM_INDEX = "all_from_index"
    EXPLICIT = "explicit"


class TransferPairSource(StrEnum):
    ALL_ORDERED_PAIRS_FROM_INDEX = "all_ordered_pairs_from_index"
    EXPLICIT = "explicit"


class ModelSelectionStrategy(StrEnum):
    FIXED_BASELINES = "fixed_baselines"
    NESTED_TUNED = "nested_tuned"


class SensitivityRole(StrEnum):
    OFFICIAL_SECONDARY_ANALYSES = "official_secondary_analyses"
    EXPLORATORY_ONLY = "exploratory_only"


class ClaimRole(StrEnum):
    PRIMARY = "primary"
    SECONDARY = "secondary"
    SUPPORTING = "supporting"


class ClaimCategory(StrEnum):
    WITHIN_PERSON_DECODING = "within_person_decoding"
    CROSS_PERSON_TRANSFER = "cross_person_transfer"
    INTERPRETABILITY_ROBUSTNESS = "interpretability_robustness"
    CONTROL_EVIDENCE = "control_evidence"


class ClaimDecisionRule(StrEnum):
    DESCRIPTIVE_ONLY = "descriptive_only"
    ABOVE_BASELINE_AND_PERMUTATION = "above_baseline_and_permutation"
    SUPPORTING_EVIDENCE_ONLY = "supporting_evidence_only"


class EstimandScope(StrEnum):
    WITHIN_SUBJECT_LOSO_SESSION = "within_subject_loso_session"
    FROZEN_CROSS_PERSON_TRANSFER = "frozen_cross_person_transfer"


class TransferPair(_ProtocolModel):
    train_subject: str = Field(min_length=1)
    test_subject: str = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_subject_pair(self) -> TransferPair:
        if self.train_subject == self.test_subject:
            raise ValueError("TransferPair requires train_subject and test_subject to differ.")
        return self


class SeedPolicy(_ProtocolModel):
    global_seed: int = 42
    per_suite_overrides_allowed: bool = False

    @model_validator(mode="after")
    def _validate_seed(self) -> SeedPolicy:
        if int(self.global_seed) < 0:
            raise ValueError("seed_policy.global_seed must be >= 0.")
        return self


class ScientificContract(_ProtocolModel):
    sample_unit: str = Field(min_length=1)
    target: str = Field(min_length=1)
    label_policy: str = Field(min_length=1)
    primary_metric: str = "balanced_accuracy"
    primary_metric_aggregation: Literal["mean_fold_scores", "pooled_held_out_predictions"] = (
        "mean_fold_scores"
    )
    secondary_metrics: list[str] = Field(default_factory=lambda: ["macro_f1", "accuracy"])
    seed_policy: SeedPolicy = Field(default_factory=SeedPolicy)

    @model_validator(mode="after")
    def _validate_metrics(self) -> ScientificContract:
        validate_metric_name(self.primary_metric)
        for metric in self.secondary_metrics:
            validate_metric_name(metric)
        if len(set(self.secondary_metrics)) != len(self.secondary_metrics):
            raise ValueError("scientific_contract.secondary_metrics must be unique.")
        return self


class SplitPolicy(_ProtocolModel):
    primary_mode: Literal["within_subject_loso_session"] = "within_subject_loso_session"
    secondary_mode: Literal["frozen_cross_person_transfer"] = "frozen_cross_person_transfer"
    grouping_field: str = "session"
    transfer_constraints: str | None = None


class ModelPolicy(_ProtocolModel):
    selection_strategy: ModelSelectionStrategy = ModelSelectionStrategy.FIXED_BASELINES
    models: list[str] = Field(default_factory=lambda: ["ridge"])
    tuning_enabled: bool = False
    nested_grouped_cv: bool = False
    class_weight_policy: ClassWeightPolicy = ClassWeightPolicy.NONE

    @model_validator(mode="after")
    def _validate_models(self) -> ModelPolicy:
        if not self.models:
            raise ValueError("model_policy.models must contain at least one model.")
        supported_models = set(ALL_MODEL_NAMES)
        for model_name in self.models:
            if model_name not in supported_models:
                allowed = ", ".join(sorted(supported_models))
                raise ValueError(
                    f"Unsupported model_policy model '{model_name}'. Allowed values: {allowed}."
                )
            _reject_exploratory_only_official_model(
                model_name=model_name,
                field_name="model_policy.models",
            )
            model_spec = get_model_spec(model_name)
            if str(self.class_weight_policy.value) not in set(
                model_spec.supported_class_weight_policies
            ):
                allowed = ", ".join(model_spec.supported_class_weight_policies)
                raise ValueError(
                    f"model_policy.class_weight_policy='{self.class_weight_policy.value}' is not "
                    f"supported by model '{model_name}'. Allowed values: {allowed}."
                )
        if (
            self.selection_strategy == ModelSelectionStrategy.FIXED_BASELINES
            and self.tuning_enabled
        ):
            raise ValueError(
                "model_policy.selection_strategy='fixed_baselines' forbids tuning_enabled=true."
            )
        if self.nested_grouped_cv and not self.tuning_enabled:
            raise ValueError("model_policy.nested_grouped_cv=true requires tuning_enabled=true.")
        return self


class ModelCostPolicy(_ProtocolModel):
    allowed_tiers: list[ModelCostTier] = Field(
        default_factory=lambda: [
            ModelCostTier.OFFICIAL_FAST,
            ModelCostTier.OFFICIAL_ALLOWED,
        ]
    )
    max_projected_runtime_seconds_per_run: int = 90 * 60

    @model_validator(mode="after")
    def _validate_cost_policy(self) -> ModelCostPolicy:
        if not self.allowed_tiers:
            raise ValueError("model_cost_policy.allowed_tiers must contain at least one tier.")
        if len(set(self.allowed_tiers)) != len(self.allowed_tiers):
            raise ValueError("model_cost_policy.allowed_tiers must be unique.")
        valid_tiers = set(supported_model_cost_tiers())
        invalid = sorted({tier.value for tier in self.allowed_tiers if tier not in valid_tiers})
        if invalid:
            allowed = ", ".join(sorted(tier.value for tier in valid_tiers))
            raise ValueError(
                "model_cost_policy.allowed_tiers contains unsupported values: "
                + ", ".join(invalid)
                + f". Allowed values: {allowed}."
            )
        if int(self.max_projected_runtime_seconds_per_run) <= 0:
            raise ValueError("model_cost_policy.max_projected_runtime_seconds_per_run must be > 0.")
        return self


class DummyBaselinePolicy(_ProtocolModel):
    enabled: bool = False
    suites: list[str] = Field(default_factory=list)


class PermutationPolicy(_ProtocolModel):
    enabled: bool = False
    metric: str | None = None
    n_permutations: int = 0
    suites: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_permutation_policy(self) -> PermutationPolicy:
        if self.metric is not None:
            validate_metric_name(self.metric)
        if self.enabled and int(self.n_permutations) <= 0:
            raise ValueError(
                "control_policy.permutation.n_permutations must be > 0 when permutations are enabled."
            )
        if not self.enabled and int(self.n_permutations) < 0:
            raise ValueError("control_policy.permutation.n_permutations must be >= 0.")
        return self


class ControlPolicy(_ProtocolModel):
    dummy_baseline: DummyBaselinePolicy = Field(default_factory=DummyBaselinePolicy)
    permutation: PermutationPolicy = Field(default_factory=PermutationPolicy)


class InterpretabilityPolicy(_ProtocolModel):
    enabled: bool = False
    suites: list[str] = Field(default_factory=list)
    modes: list[str] = Field(default_factory=list)
    models: list[str] = Field(default_factory=list)
    supporting_evidence_only: bool = True

    @model_validator(mode="after")
    def _validate_interpretability_policy(self) -> InterpretabilityPolicy:
        if self.enabled:
            if not self.suites:
                raise ValueError(
                    "interpretability_policy.suites must be non-empty when interpretability is enabled."
                )
            if not self.modes:
                raise ValueError(
                    "interpretability_policy.modes must be non-empty when interpretability is enabled."
                )
            if not self.models:
                raise ValueError(
                    "interpretability_policy.models must be non-empty when interpretability is enabled."
                )
            supported_models = set(ALL_MODEL_NAMES)
            for model_name in self.models:
                if model_name not in supported_models:
                    allowed = ", ".join(sorted(supported_models))
                    raise ValueError(
                        f"Unsupported interpretability model '{model_name}'. Allowed values: {allowed}."
                    )
                _reject_exploratory_only_official_model(
                    model_name=model_name,
                    field_name="interpretability_policy.models",
                )
            for mode in self.modes:
                if mode not in SUPPORTED_CV_MODES:
                    allowed = ", ".join(sorted(SUPPORTED_CV_MODES))
                    raise ValueError(
                        f"Unsupported interpretability mode '{mode}'. Allowed values: {allowed}."
                    )
        return self


class FeatureEngineeringPolicy(_ProtocolModel):
    feature_recipe_id: str = BASELINE_STANDARD_SCALER_RECIPE_ID
    emit_feature_qc_artifacts: bool = True

    @model_validator(mode="after")
    def _validate_feature_engineering_policy(self) -> FeatureEngineeringPolicy:
        self.feature_recipe_id = resolve_feature_recipe_id(self.feature_recipe_id)
        if self.feature_recipe_id not in set(FEATURE_RECIPE_IDS):
            allowed = ", ".join(sorted(FEATURE_RECIPE_IDS))
            raise ValueError(
                "feature_engineering_policy.feature_recipe_id is unsupported. "
                f"Allowed values: {allowed}."
            )
        if not bool(self.emit_feature_qc_artifacts):
            raise ValueError(
                "feature_engineering_policy.emit_feature_qc_artifacts must be true "
                "for official protocol runs."
            )
        return self


class SensitivityPolicy(_ProtocolModel):
    role: SensitivityRole = SensitivityRole.EXPLORATORY_ONLY
    suites: list[str] = Field(default_factory=list)


class ArtifactContract(_ProtocolModel):
    required_run_artifacts: list[str] = Field(
        default_factory=lambda: list(REQUIRED_RUN_ARTIFACTS_BASELINE)
    )
    required_protocol_artifacts: list[str] = Field(
        default_factory=lambda: list(REQUIRED_PROTOCOL_ARTIFACTS)
    )
    required_run_metadata_fields: list[str] = Field(
        default_factory=lambda: [
            "framework_mode",
            "canonical_run",
            "methodology_policy_name",
            "class_weight_policy",
            "tuning_enabled",
            "feature_recipe_id",
            "model_cost_tier",
            "projected_runtime_seconds",
            "evidence_run_role",
            "repeat_id",
            "repeat_count",
            "base_run_id",
            "primary_metric_name",
            "primary_metric_aggregation",
            "data_policy_effective",
            "protocol_id",
            "protocol_version",
            "protocol_schema_version",
            "suite_id",
            "claim_ids",
        ]
    )

    @model_validator(mode="after")
    def _validate_required_artifacts(self) -> ArtifactContract:
        if not self.required_run_artifacts:
            raise ValueError("artifact_contract.required_run_artifacts must not be empty.")
        if not self.required_protocol_artifacts:
            raise ValueError("artifact_contract.required_protocol_artifacts must not be empty.")
        missing_protocol = [
            name
            for name in REQUIRED_PROTOCOL_ARTIFACTS
            if name not in self.required_protocol_artifacts
        ]
        if missing_protocol:
            raise ValueError(
                "artifact_contract.required_protocol_artifacts is missing required entries: "
                + ", ".join(missing_protocol)
            )
        missing_run = [
            name
            for name in (
                "config.json",
                "metrics.json",
                "dataset_card.json",
                "dataset_summary.json",
                "data_quality_report.json",
                "leakage_audit.json",
                "cv_split_manifest.json",
                "cv_split_manifest.csv",
                "feature_qc_summary.json",
                "feature_qc_selected_samples.csv",
            )
            if name not in self.required_run_artifacts
        ]
        if missing_run:
            raise ValueError(
                "artifact_contract.required_run_artifacts is missing required entries: "
                + ", ".join(missing_run)
            )
        if not self.required_run_metadata_fields:
            raise ValueError("artifact_contract.required_run_metadata_fields must not be empty.")
        for key in (
            "framework_mode",
            "canonical_run",
            "methodology_policy_name",
            "class_weight_policy",
            "tuning_enabled",
            "feature_recipe_id",
            "model_cost_tier",
            "projected_runtime_seconds",
            "evidence_run_role",
            "repeat_id",
            "repeat_count",
            "base_run_id",
            "primary_metric_name",
            "primary_metric_aggregation",
            "data_policy_effective",
            "protocol_id",
            "protocol_version",
            "suite_id",
        ):
            if key not in self.required_run_metadata_fields:
                raise ValueError(
                    "artifact_contract.required_run_metadata_fields is missing required key: " + key
                )
        return self


class ClaimSpec(_ProtocolModel):
    claim_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    description: str = Field(min_length=1)
    role: ClaimRole
    category: ClaimCategory
    estimand_scope: EstimandScope
    decision_metric: str
    decision_rule: ClaimDecisionRule
    suite_ids: list[str] = Field(min_length=1)
    baseline_required: bool = False
    permutation_required: bool = False
    interpretation_limits: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_claim(self) -> ClaimSpec:
        self.decision_metric = validate_metric_name(self.decision_metric)

        if len(set(self.suite_ids)) != len(self.suite_ids):
            raise ValueError("claim suite_ids must be unique")

        if self.role == ClaimRole.PRIMARY:
            if self.category != ClaimCategory.WITHIN_PERSON_DECODING:
                raise ValueError("primary claim must use category='within_person_decoding'")
            if self.estimand_scope != EstimandScope.WITHIN_SUBJECT_LOSO_SESSION:
                raise ValueError(
                    "primary claim must use estimand_scope='within_subject_loso_session'"
                )

        if self.category == ClaimCategory.CROSS_PERSON_TRANSFER:
            if self.role == ClaimRole.PRIMARY:
                raise ValueError("cross_person_transfer claim cannot be primary")
            if self.estimand_scope != EstimandScope.FROZEN_CROSS_PERSON_TRANSFER:
                raise ValueError(
                    "cross_person_transfer claim must use estimand_scope='frozen_cross_person_transfer'"
                )

        if self.category == ClaimCategory.INTERPRETABILITY_ROBUSTNESS:
            if self.role != ClaimRole.SUPPORTING:
                raise ValueError("interpretability_robustness claim must be supporting")
            if self.decision_rule != ClaimDecisionRule.SUPPORTING_EVIDENCE_ONLY:
                raise ValueError(
                    "interpretability_robustness claim must use decision_rule='supporting_evidence_only'"
                )

        if self.decision_rule == ClaimDecisionRule.ABOVE_BASELINE_AND_PERMUTATION:
            if not self.baseline_required:
                raise ValueError(
                    "above_baseline_and_permutation claim must set baseline_required=true"
                )
            if not self.permutation_required:
                raise ValueError(
                    "above_baseline_and_permutation claim must set permutation_required=true"
                )

        if self.role == ClaimRole.SUPPORTING:
            if self.decision_rule == ClaimDecisionRule.ABOVE_BASELINE_AND_PERMUTATION:
                raise ValueError(
                    "supporting claim cannot use decision_rule='above_baseline_and_permutation'"
                )

        if self.role != ClaimRole.PRIMARY and not self.interpretation_limits:
            raise ValueError("non-primary claim must declare at least one interpretation limit")

        return self


class SuccessCriteria(_ProtocolModel):
    primary_claim_id: str = Field(min_length=1)
    require_dummy_baseline_outperformance: bool = True
    require_permutation_pass: bool = True
    permutation_alpha: float = 0.05
    require_complete_primary_suite_evidence: bool = True
    secondary_cannot_substitute_for_primary: bool = True
    supporting_cannot_substitute_for_primary: bool = True
    transfer_is_secondary_only: bool = True
    interpretability_is_supporting_only: bool = True

    @model_validator(mode="after")
    def _validate_success_criteria(self) -> SuccessCriteria:
        if not (0.0 < float(self.permutation_alpha) <= 1.0):
            raise ValueError("permutation_alpha must be in (0, 1]")

        if self.transfer_is_secondary_only is not True:
            raise ValueError("transfer_is_secondary_only must be true in this protocol version")

        if self.interpretability_is_supporting_only is not True:
            raise ValueError(
                "interpretability_is_supporting_only must be true in this protocol version"
            )

        if self.secondary_cannot_substitute_for_primary is not True:
            raise ValueError(
                "secondary_cannot_substitute_for_primary must be true in this protocol version"
            )

        if self.supporting_cannot_substitute_for_primary is not True:
            raise ValueError(
                "supporting_cannot_substitute_for_primary must be true in this protocol version"
            )

        return self


class SuiteSpec(_ProtocolModel):
    suite_id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    enabled: bool = True
    suite_type: SuiteType
    claim_ids: list[str] = Field(min_length=1)
    split_mode: Literal["within_subject_loso_session", "frozen_cross_person_transfer"]
    models: list[str] | None = None
    subject_source: SubjectSource = SubjectSource.ALL_FROM_INDEX
    subjects: list[str] = Field(default_factory=list)
    transfer_pair_source: TransferPairSource = TransferPairSource.ALL_ORDERED_PAIRS_FROM_INDEX
    transfer_pairs: list[TransferPair] = Field(default_factory=list)
    filter_task: str | None = None
    filter_modality: str | None = None
    seed_override: int | None = None
    controls_required: bool = False
    interpretability_requested: bool = False

    @model_validator(mode="after")
    def _validate_suite(self) -> SuiteSpec:
        if len(set(self.claim_ids)) != len(self.claim_ids):
            raise ValueError(f"Suite '{self.suite_id}' defines duplicate claim_ids.")
        if self.models is not None:
            supported_models = set(ALL_MODEL_NAMES)
            if not self.models:
                raise ValueError(
                    f"Suite '{self.suite_id}' defines an empty models list; omit models to use model_policy.models."
                )
            for model_name in self.models:
                if model_name not in supported_models:
                    allowed = ", ".join(sorted(supported_models))
                    raise ValueError(
                        f"Suite '{self.suite_id}' references unsupported model '{model_name}'. "
                        f"Allowed values: {allowed}."
                    )
                _reject_exploratory_only_official_model(
                    model_name=model_name,
                    field_name=f"suite '{self.suite_id}' models",
                )
        if self.seed_override is not None and int(self.seed_override) < 0:
            raise ValueError(f"Suite '{self.suite_id}' seed_override must be >= 0.")

        if self.split_mode == "within_subject_loso_session":
            if self.transfer_pairs:
                raise ValueError(
                    f"Suite '{self.suite_id}' uses within_subject_loso_session and cannot define transfer_pairs."
                )
            if self.subject_source == SubjectSource.EXPLICIT and not self.subjects:
                raise ValueError(
                    f"Suite '{self.suite_id}' subject_source='explicit' requires subjects."
                )

        if self.split_mode == "frozen_cross_person_transfer":
            if self.subjects:
                raise ValueError(
                    f"Suite '{self.suite_id}' uses frozen_cross_person_transfer and cannot define subjects."
                )
            if self.transfer_pair_source == TransferPairSource.EXPLICIT and not self.transfer_pairs:
                raise ValueError(
                    f"Suite '{self.suite_id}' transfer_pair_source='explicit' requires transfer_pairs."
                )

        return self


class ThesisProtocol(_ProtocolModel):
    protocol_schema_version: str = THESIS_PROTOCOL_SCHEMA_VERSION
    framework_mode: Literal["confirmatory"] = FrameworkMode.CONFIRMATORY.value
    protocol_id: str = Field(min_length=1)
    protocol_version: str = Field(min_length=1)
    status: ProtocolStatus
    description: str = Field(min_length=1)
    notes: str | None = None
    confirmatory_lock: dict[str, Any] | None = None
    claims: list[ClaimSpec] = Field(min_length=1)
    success_criteria: SuccessCriteria
    scientific_contract: ScientificContract
    split_policy: SplitPolicy
    model_policy: ModelPolicy
    model_cost_policy: ModelCostPolicy = Field(default_factory=ModelCostPolicy)
    methodology_policy: MethodologyPolicy
    metric_policy: MetricPolicy = Field(default_factory=MetricPolicy)
    subgroup_reporting_policy: SubgroupReportingPolicy = Field(
        default_factory=SubgroupReportingPolicy
    )
    data_policy: DataPolicy = Field(default_factory=DataPolicy)
    evidence_policy: EvidencePolicy
    control_policy: ControlPolicy
    interpretability_policy: InterpretabilityPolicy
    feature_engineering_policy: FeatureEngineeringPolicy = Field(
        default_factory=FeatureEngineeringPolicy
    )
    sensitivity_policy: SensitivityPolicy
    artifact_contract: ArtifactContract
    official_run_suites: list[SuiteSpec] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_protocol(self) -> ThesisProtocol:
        if self.protocol_schema_version not in SUPPORTED_THESIS_PROTOCOL_SCHEMA_VERSIONS:
            allowed = ", ".join(sorted(SUPPORTED_THESIS_PROTOCOL_SCHEMA_VERSIONS))
            raise ValueError(
                f"Unsupported protocol_schema_version '{self.protocol_schema_version}'. "
                f"Allowed values: {allowed}."
            )

        if self.metric_policy.primary_metric != self.scientific_contract.primary_metric:
            raise ValueError(
                "metric_policy.primary_metric must match scientific_contract.primary_metric."
            )
        if set(self.metric_policy.secondary_metrics) != set(
            self.scientific_contract.secondary_metrics
        ):
            raise ValueError(
                "metric_policy.secondary_metrics must match scientific_contract.secondary_metrics."
            )

        if self.model_policy.class_weight_policy != self.methodology_policy.class_weight_policy:
            raise ValueError(
                "model_policy.class_weight_policy must match methodology_policy.class_weight_policy."
            )
        if (
            str(self.feature_engineering_policy.feature_recipe_id)
            != BASELINE_STANDARD_SCALER_RECIPE_ID
        ):
            raise ValueError(
                "Confirmatory protocol runs require "
                "feature_engineering_policy.feature_recipe_id='baseline_standard_scaler_v1'."
            )
        if not bool(self.feature_engineering_policy.emit_feature_qc_artifacts):
            raise ValueError(
                "Confirmatory protocol runs require "
                "feature_engineering_policy.emit_feature_qc_artifacts=true."
            )
        if self.methodology_policy.policy_name == MethodologyPolicyName.FIXED_BASELINES_ONLY:
            if self.model_policy.selection_strategy != ModelSelectionStrategy.FIXED_BASELINES:
                raise ValueError(
                    "fixed_baselines_only requires model_policy.selection_strategy='fixed_baselines'."
                )
            if self.model_policy.tuning_enabled:
                raise ValueError("fixed_baselines_only requires model_policy.tuning_enabled=false.")
            if self.model_policy.nested_grouped_cv:
                raise ValueError(
                    "fixed_baselines_only requires model_policy.nested_grouped_cv=false."
                )
        if self.methodology_policy.policy_name == MethodologyPolicyName.GROUPED_NESTED_TUNING:
            if self.model_policy.selection_strategy != ModelSelectionStrategy.NESTED_TUNED:
                raise ValueError(
                    "grouped_nested_tuning requires model_policy.selection_strategy='nested_tuned'."
                )
            if not self.model_policy.tuning_enabled:
                raise ValueError("grouped_nested_tuning requires model_policy.tuning_enabled=true.")
            if not self.model_policy.nested_grouped_cv:
                raise ValueError(
                    "grouped_nested_tuning requires model_policy.nested_grouped_cv=true."
                )
        allowed_tiers = set(self.model_cost_policy.allowed_tiers)
        max_runtime = int(self.model_cost_policy.max_projected_runtime_seconds_per_run)
        protocol_models = {
            str(model_name).strip()
            for model_name in list(self.model_policy.models)
            if str(model_name).strip()
        }
        for suite in self.official_run_suites:
            suite_models = suite.models if suite.models is not None else self.model_policy.models
            for model_name in suite_models:
                if str(model_name).strip():
                    protocol_models.add(str(model_name).strip())
        for model_name in sorted(protocol_models):
            catalog_entry = get_model_cost_entry(model_name)
            if catalog_entry.cost_tier not in allowed_tiers:
                allowed = ", ".join(sorted(tier.value for tier in allowed_tiers))
                raise ValueError(
                    "Confirmatory protocol model cost policy rejected model "
                    f"'{model_name}': tier='{catalog_entry.cost_tier.value}' is not allowed. "
                    f"Allowed tiers: {allowed}."
                )
            projected_runtime = projected_runtime_seconds(
                model_name=model_name,
                framework_mode=FrameworkMode.CONFIRMATORY,
                methodology_policy=self.methodology_policy.policy_name,
                tuning_enabled=bool(self.methodology_policy.tuning_enabled),
            )
            if int(projected_runtime) > max_runtime:
                raise ValueError(
                    "Confirmatory protocol model cost policy rejected model "
                    f"'{model_name}': projected_runtime_seconds={projected_runtime} exceeds "
                    "max_projected_runtime_seconds_per_run="
                    f"{max_runtime}."
                )

        suite_ids = [suite.suite_id for suite in self.official_run_suites]
        if len(set(suite_ids)) != len(suite_ids):
            raise ValueError("official_run_suites contains duplicate suite_id values.")

        suite_id_set = set(suite_ids)
        for listed_suite in self.control_policy.dummy_baseline.suites:
            if listed_suite not in suite_id_set:
                raise ValueError(
                    f"control_policy.dummy_baseline references unknown suite '{listed_suite}'."
                )
        for listed_suite in self.control_policy.permutation.suites:
            if listed_suite not in suite_id_set:
                raise ValueError(
                    f"control_policy.permutation references unknown suite '{listed_suite}'."
                )
        for listed_suite in self.interpretability_policy.suites:
            if listed_suite not in suite_id_set:
                raise ValueError(
                    f"interpretability_policy references unknown suite '{listed_suite}'."
                )
        for listed_suite in self.sensitivity_policy.suites:
            if listed_suite not in suite_id_set:
                raise ValueError(f"sensitivity_policy references unknown suite '{listed_suite}'.")

        permutation_metric = (
            self.control_policy.permutation.metric or self.metric_policy.primary_metric
        )
        if (
            self.control_policy.permutation.enabled
            and permutation_metric != self.metric_policy.primary_metric
        ):
            raise ValueError(
                "control_policy.permutation.metric must match metric_policy.primary_metric "
                "for confirmatory protocol runs."
            )
        if (
            self.evidence_policy.required_package.require_dummy_baseline
            and not self.control_policy.dummy_baseline.enabled
        ):
            raise ValueError(
                "evidence_policy.required_package.require_dummy_baseline=true "
                "requires control_policy.dummy_baseline.enabled=true."
            )
        if (
            self.evidence_policy.required_package.require_permutation_control
            and not self.control_policy.permutation.enabled
        ):
            raise ValueError(
                "evidence_policy.required_package.require_permutation_control=true "
                "requires control_policy.permutation.enabled=true."
            )
        if self.control_policy.permutation.enabled and int(
            self.control_policy.permutation.n_permutations
        ) < int(self.evidence_policy.permutation.minimum_permutations):
            raise ValueError(
                "control_policy.permutation.n_permutations must be >= "
                "evidence_policy.permutation.minimum_permutations."
            )

        for suite in self.official_run_suites:
            suite_models = suite.models if suite.models is not None else self.model_policy.models
            if suite.controls_required and suite.suite_type != SuiteType.CONTROL:
                raise ValueError(
                    f"Suite '{suite.suite_id}' sets controls_required=true but suite_type is not 'control'."
                )
            if suite.interpretability_requested:
                if not self.interpretability_policy.enabled:
                    raise ValueError(
                        f"Suite '{suite.suite_id}' requests interpretability but interpretability_policy.enabled is false."
                    )
                if suite.suite_id not in set(self.interpretability_policy.suites):
                    raise ValueError(
                        f"Suite '{suite.suite_id}' requests interpretability but is not listed in interpretability_policy.suites."
                    )
                if suite.split_mode not in set(self.interpretability_policy.modes):
                    raise ValueError(
                        f"Suite '{suite.suite_id}' requests interpretability for unsupported split_mode '{suite.split_mode}'."
                    )
                disallowed_models = [
                    model_name
                    for model_name in suite_models
                    if model_name not in set(self.interpretability_policy.models)
                ]
                if disallowed_models:
                    raise ValueError(
                        f"Suite '{suite.suite_id}' requests interpretability for unsupported models: "
                        + ", ".join(sorted(set(disallowed_models)))
                    )

            if suite.split_mode == self.split_policy.secondary_mode and suite.suite_type not in {
                SuiteType.SECONDARY,
                SuiteType.CONTROL,
            }:
                raise ValueError(
                    f"Suite '{suite.suite_id}' uses secondary split_mode "
                    f"'{self.split_policy.secondary_mode}' but suite_type='{suite.suite_type.value}'."
                )

        if self.protocol_id == "thesis_confirmatory_v1" and not isinstance(
            self.confirmatory_lock, dict
        ):
            raise ValueError(
                "thesis_confirmatory_v1 requires confirmatory_lock metadata for hard-gate enforcement."
            )

        claim_ids = [claim.claim_id for claim in self.claims]
        if len(set(claim_ids)) != len(claim_ids):
            raise ValueError("claims must have unique claim_id values")

        claim_by_id = {claim.claim_id: claim for claim in self.claims}
        suite_by_id = {suite.suite_id: suite for suite in self.official_run_suites}
        suite_id_set = set(suite_by_id.keys())

        primary_claims = [claim for claim in self.claims if claim.role == ClaimRole.PRIMARY]
        if len(primary_claims) != 1:
            raise ValueError("protocol must declare exactly one primary claim")
        primary_claim_id = self.success_criteria.primary_claim_id
        if primary_claim_id not in claim_by_id:
            raise ValueError("success_criteria.primary_claim_id must reference an existing claim")

        primary_claim = claim_by_id[primary_claim_id]
        if primary_claim.role != ClaimRole.PRIMARY:
            raise ValueError("success_criteria.primary_claim_id must reference the primary claim")

        if primary_claim.decision_metric != self.metric_policy.primary_metric:
            raise ValueError(
                "primary claim decision_metric must match metric_policy.primary_metric"
            )

        for suite in self.official_run_suites:
            for claim_id in suite.claim_ids:
                if claim_id not in claim_by_id:
                    raise ValueError(
                        f"suite '{suite.suite_id}' references unknown claim_id '{claim_id}'"
                    )

        for claim in self.claims:
            for suite_id in claim.suite_ids:
                if suite_id not in suite_id_set:
                    raise ValueError(
                        f"claim '{claim.claim_id}' references unknown suite_id '{suite_id}'"
                    )

        for suite in self.official_run_suites:
            for claim_id in suite.claim_ids:
                claim = claim_by_id[claim_id]
                if suite.suite_id not in claim.suite_ids:
                    raise ValueError(
                        f"suite '{suite.suite_id}' references claim '{claim_id}' but that claim does not reference the suite"
                    )

        for claim in self.claims:
            for suite_id in claim.suite_ids:
                suite = suite_by_id[suite_id]
                if claim.claim_id not in suite.claim_ids:
                    raise ValueError(
                        f"claim '{claim.claim_id}' references suite '{suite_id}' but that suite does not reference the claim"
                    )

        for claim in self.claims:
            for suite_id in claim.suite_ids:
                suite = suite_by_id[suite_id]

                if claim.estimand_scope == EstimandScope.WITHIN_SUBJECT_LOSO_SESSION:
                    if suite.split_mode != "within_subject_loso_session":
                        raise ValueError(
                            f"claim '{claim.claim_id}' requires split_mode='within_subject_loso_session' but suite '{suite_id}' uses '{suite.split_mode}'"
                        )

                if claim.estimand_scope == EstimandScope.FROZEN_CROSS_PERSON_TRANSFER:
                    if suite.split_mode != "frozen_cross_person_transfer":
                        raise ValueError(
                            f"claim '{claim.claim_id}' requires split_mode='frozen_cross_person_transfer' but suite '{suite_id}' uses '{suite.split_mode}'"
                        )

        for claim in self.claims:
            for suite_id in claim.suite_ids:
                suite = suite_by_id[suite_id]

                if claim.role == ClaimRole.PRIMARY:
                    if suite.suite_type != EvidenceRunRole.PRIMARY:
                        raise ValueError(
                            f"primary claim '{claim.claim_id}' must reference only PRIMARY suites"
                        )

                if claim.category == ClaimCategory.CROSS_PERSON_TRANSFER:
                    if suite.suite_type == EvidenceRunRole.PRIMARY:
                        raise ValueError(
                            f"cross_person_transfer claim '{claim.claim_id}' cannot reference PRIMARY suites"
                        )

        if primary_claim.baseline_required:
            if not self.control_policy.dummy_baseline.enabled:
                raise ValueError(
                    "primary claim requires dummy baseline but control_policy.dummy_baseline.enabled is false"
                )

        if primary_claim.permutation_required:
            if not self.control_policy.permutation.enabled:
                raise ValueError(
                    "primary claim requires permutation but control_policy.permutation.enabled is false"
                )

        if (
            (
                self.success_criteria.require_dummy_baseline_outperformance
                or self.success_criteria.require_permutation_pass
            )
            and self.protocol_id != "thesis_confirmatory_v1"
        ):
            supporting_control_claims = [
                claim
                for claim in self.claims
                if (
                    claim.role == ClaimRole.SUPPORTING
                    and claim.category == ClaimCategory.CONTROL_EVIDENCE
                    and claim.estimand_scope == primary_claim.estimand_scope
                )
            ]
            if not supporting_control_claims:
                if self.success_criteria.require_dummy_baseline_outperformance:
                    raise ValueError(
                        "success_criteria.require_dummy_baseline_outperformance=true requires "
                        "a supporting CONTROL_EVIDENCE claim that matches the primary estimand_scope."
                    )
                raise ValueError(
                    "success_criteria.require_permutation_pass=true requires "
                    "a supporting CONTROL_EVIDENCE claim that matches the primary estimand_scope."
                )

            supporting_control_suite_ids = {
                suite_id for claim in supporting_control_claims for suite_id in claim.suite_ids
            }

            if self.success_criteria.require_dummy_baseline_outperformance:
                dummy_control_suites = set(self.control_policy.dummy_baseline.suites)
                if not dummy_control_suites:
                    raise ValueError(
                        "success_criteria.require_dummy_baseline_outperformance=true requires "
                        "control_policy.dummy_baseline.suites to be non-empty."
                    )
                missing_dummy_control_suite_links = sorted(
                    dummy_control_suites - supporting_control_suite_ids
                )
                if missing_dummy_control_suite_links:
                    raise ValueError(
                        "control_policy.dummy_baseline.suites must be represented by supporting "
                        "CONTROL_EVIDENCE claims for primary-claim dummy-baseline evaluation. Missing: "
                        + ", ".join(missing_dummy_control_suite_links)
                    )

            if self.success_criteria.require_permutation_pass:
                permutation_control_suites = set(self.control_policy.permutation.suites)
                if not permutation_control_suites:
                    raise ValueError(
                        "success_criteria.require_permutation_pass=true requires "
                        "control_policy.permutation.suites to be non-empty."
                    )
                missing_permutation_control_suite_links = sorted(
                    permutation_control_suites - supporting_control_suite_ids
                )
                if missing_permutation_control_suite_links:
                    raise ValueError(
                        "control_policy.permutation.suites must be represented by supporting "
                        "CONTROL_EVIDENCE claims for primary-claim permutation evaluation. Missing: "
                        + ", ".join(missing_permutation_control_suite_links)
                    )

        has_interpretability_claim = any(
            claim.category == ClaimCategory.INTERPRETABILITY_ROBUSTNESS for claim in self.claims
        )

        if has_interpretability_claim and not self.interpretability_policy.enabled:
            raise ValueError(
                "interpretability claim declared but interpretability_policy.enabled is false"
            )

        if self.success_criteria.interpretability_is_supporting_only:
            for claim in self.claims:
                if claim.category == ClaimCategory.INTERPRETABILITY_ROBUSTNESS:
                    if claim.role != ClaimRole.SUPPORTING:
                        raise ValueError(
                            "interpretability claim must be supporting when interpretability_is_supporting_only=true"
                        )

        if self.success_criteria.transfer_is_secondary_only:
            for claim in self.claims:
                if claim.category == ClaimCategory.CROSS_PERSON_TRANSFER:
                    if claim.role == ClaimRole.PRIMARY:
                        raise ValueError(
                            "cross_person_transfer claim cannot be primary when transfer_is_secondary_only=true"
                        )

        return self


class CompiledRunControls(_ProtocolModel):
    dummy_baseline_run: bool = False
    permutation_enabled: bool = False
    permutation_metric: str | None = None
    n_permutations: int = 0

    @model_validator(mode="after")
    def _validate_controls(self) -> CompiledRunControls:
        if self.permutation_enabled:
            if self.permutation_metric is None:
                raise ValueError("CompiledRunControls.permutation_metric is required when enabled.")
            validate_metric_name(self.permutation_metric)
            if int(self.n_permutations) <= 0:
                raise ValueError("CompiledRunControls.n_permutations must be > 0 when enabled.")
        return self


class CompiledRunSpec(_ProtocolModel):
    run_id: str = Field(min_length=1)
    suite_id: str = Field(min_length=1)
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
    primary_metric_aggregation: Literal["mean_fold_scores", "pooled_held_out_predictions"] = (
        "mean_fold_scores"
    )
    controls: CompiledRunControls = Field(default_factory=CompiledRunControls)
    interpretability_enabled: bool = False
    methodology_policy_name: MethodologyPolicyName = MethodologyPolicyName.FIXED_BASELINES_ONLY
    class_weight_policy: ClassWeightPolicy = ClassWeightPolicy.NONE
    feature_recipe_id: str = BASELINE_STANDARD_SCALER_RECIPE_ID
    emit_feature_qc_artifacts: bool = True
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
    framework_mode: Literal["confirmatory"] = FrameworkMode.CONFIRMATORY.value
    canonical_run: bool = True
    artifact_requirements: list[str] = Field(
        default_factory=lambda: list(REQUIRED_RUN_ARTIFACTS_BASELINE)
    )
    protocol_id: str = Field(min_length=1)
    protocol_version: str = Field(min_length=1)
    protocol_schema_version: str = THESIS_PROTOCOL_SCHEMA_VERSION

    @model_validator(mode="after")
    def _validate_compiled_spec(self) -> CompiledRunSpec:
        if self.model not in set(ALL_MODEL_NAMES):
            allowed = ", ".join(sorted(ALL_MODEL_NAMES))
            raise ValueError(
                f"CompiledRunSpec model '{self.model}' is unsupported. Allowed values: {allowed}."
            )
        _reject_exploratory_only_official_model(
            model_name=self.model,
            field_name="CompiledRunSpec",
        )
        model_spec = get_model_spec(self.model)
        if str(self.class_weight_policy.value) not in set(model_spec.supported_class_weight_policies):
            allowed = ", ".join(model_spec.supported_class_weight_policies)
            raise ValueError(
                f"CompiledRunSpec '{self.run_id}' class_weight_policy='{self.class_weight_policy.value}' "
                f"is not supported by model '{self.model}'. Allowed values: {allowed}."
            )
        catalog_entry = get_model_cost_entry(self.model)
        if self.model_cost_tier != catalog_entry.cost_tier:
            raise ValueError(
                f"CompiledRunSpec '{self.run_id}' model_cost_tier='{self.model_cost_tier.value}' "
                f"does not match catalog tier '{catalog_entry.cost_tier.value}' for model "
                f"'{self.model}'."
            )
        expected_projected_runtime = projected_runtime_seconds(
            model_name=self.model,
            framework_mode=FrameworkMode.CONFIRMATORY,
            methodology_policy=self.methodology_policy_name,
            tuning_enabled=bool(self.tuning_enabled),
        )
        if int(self.projected_runtime_seconds) <= 0:
            raise ValueError("CompiledRunSpec.projected_runtime_seconds must be > 0.")
        if int(self.projected_runtime_seconds) != int(expected_projected_runtime):
            raise ValueError(
                f"CompiledRunSpec '{self.run_id}' projected_runtime_seconds="
                f"{self.projected_runtime_seconds} does not match expected "
                f"{expected_projected_runtime} for model '{self.model}'."
            )
        if self.cv_mode == "within_subject_loso_session":
            if self.subject is None:
                raise ValueError(
                    f"CompiledRunSpec '{self.run_id}' requires subject for within_subject_loso_session."
                )
            if self.train_subject is not None or self.test_subject is not None:
                raise ValueError(
                    f"CompiledRunSpec '{self.run_id}' cannot set train_subject/test_subject in within-subject mode."
                )
        if self.cv_mode == "frozen_cross_person_transfer":
            if self.train_subject is None or self.test_subject is None:
                raise ValueError(
                    f"CompiledRunSpec '{self.run_id}' requires train_subject and test_subject."
                )
            if self.subject is not None:
                raise ValueError(
                    f"CompiledRunSpec '{self.run_id}' cannot set subject in frozen transfer mode."
                )
            if self.train_subject == self.test_subject:
                raise ValueError(
                    f"CompiledRunSpec '{self.run_id}' requires different train_subject and test_subject."
                )
        if int(self.seed) < 0:
            raise ValueError("CompiledRunSpec.seed must be >= 0.")
        validate_metric_name(self.primary_metric)
        if self.controls.permutation_enabled:
            if self.controls.permutation_metric != self.primary_metric:
                raise ValueError(
                    f"CompiledRunSpec '{self.run_id}' requires controls.permutation_metric "
                    "to match primary_metric for confirmatory runs."
                )
        resolved_recipe_id = resolve_feature_recipe_id(self.feature_recipe_id)
        if resolved_recipe_id != BASELINE_STANDARD_SCALER_RECIPE_ID:
            raise ValueError(
                f"CompiledRunSpec '{self.run_id}' requires feature_recipe_id="
                f"'{BASELINE_STANDARD_SCALER_RECIPE_ID}' for confirmatory runs."
            )
        self.feature_recipe_id = resolved_recipe_id
        if not bool(self.emit_feature_qc_artifacts):
            raise ValueError(
                f"CompiledRunSpec '{self.run_id}' requires emit_feature_qc_artifacts=true "
                "for confirmatory runs."
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
            raise ValueError("CompiledRunSpec.repeat_count must be > 0.")
        if int(self.repeat_id) <= 0:
            raise ValueError("CompiledRunSpec.repeat_id must be > 0.")
        if int(self.repeat_id) > int(self.repeat_count):
            raise ValueError("CompiledRunSpec.repeat_id must be <= repeat_count.")
        if not str(self.base_run_id).strip():
            raise ValueError("CompiledRunSpec.base_run_id must be non-empty.")
        if self.framework_mode != FrameworkMode.CONFIRMATORY.value:
            raise ValueError(
                f"CompiledRunSpec '{self.run_id}' must use framework_mode='confirmatory'."
            )
        if self.canonical_run is not True:
            raise ValueError(
                f"CompiledRunSpec '{self.run_id}' must set canonical_run=true in confirmatory mode."
            )
        return self


class CompiledProtocolManifest(_ProtocolModel):
    compiled_schema_version: str = "thesis-protocol-compiled-v1"
    framework_mode: Literal["confirmatory"] = FrameworkMode.CONFIRMATORY.value
    protocol_schema_version: str = THESIS_PROTOCOL_SCHEMA_VERSION
    protocol_id: str = Field(min_length=1)
    protocol_version: str = Field(min_length=1)
    status: ProtocolStatus
    methodology_policy: MethodologyPolicy
    metric_policy: MetricPolicy
    subgroup_reporting_policy: SubgroupReportingPolicy
    feature_engineering_policy: FeatureEngineeringPolicy
    data_policy: DataPolicy
    evidence_policy: EvidencePolicy
    suite_ids: list[str] = Field(min_length=1)
    runs: list[CompiledRunSpec] = Field(min_length=1)
    claim_to_run_map: dict[str, list[str]]
    required_protocol_artifacts: list[str] = Field(
        default_factory=lambda: list(REQUIRED_PROTOCOL_ARTIFACTS)
    )
    required_run_artifacts: list[str] = Field(
        default_factory=lambda: list(REQUIRED_RUN_ARTIFACTS_BASELINE)
    )
    required_run_metadata_fields: list[str] = Field(
        default_factory=lambda: [
            "framework_mode",
            "canonical_run",
            "methodology_policy_name",
            "class_weight_policy",
            "tuning_enabled",
            "feature_recipe_id",
            "model_cost_tier",
            "projected_runtime_seconds",
            "evidence_run_role",
            "repeat_id",
            "repeat_count",
            "base_run_id",
            "primary_metric_name",
            "primary_metric_aggregation",
            "data_policy_effective",
            "protocol_id",
            "protocol_version",
            "protocol_schema_version",
            "suite_id",
            "claim_ids",
        ]
    )

    @model_validator(mode="after")
    def _validate_manifest(self) -> CompiledProtocolManifest:
        if self.framework_mode != FrameworkMode.CONFIRMATORY.value:
            raise ValueError("CompiledProtocolManifest.framework_mode must be 'confirmatory'.")
        if len(set(self.suite_ids)) != len(self.suite_ids):
            raise ValueError("CompiledProtocolManifest.suite_ids must be unique.")
        run_ids = [run.run_id for run in self.runs]
        if len(set(run_ids)) != len(run_ids):
            raise ValueError("CompiledProtocolManifest.runs contains duplicate run_id values.")
        for claim_id, mapped_runs in self.claim_to_run_map.items():
            if not mapped_runs:
                raise ValueError(
                    f"CompiledProtocolManifest claim_to_run_map['{claim_id}'] must list at least one run."
                )
        if not self.required_run_metadata_fields:
            raise ValueError(
                "CompiledProtocolManifest.required_run_metadata_fields must not be empty."
            )
        for key in (
            "framework_mode",
            "canonical_run",
            "feature_recipe_id",
            "model_cost_tier",
            "projected_runtime_seconds",
            "data_policy_effective",
            "primary_metric_aggregation",
            "protocol_id",
            "protocol_version",
            "suite_id",
        ):
            if key not in self.required_run_metadata_fields:
                raise ValueError(
                    "CompiledProtocolManifest.required_run_metadata_fields is missing required key: "
                    + key
                )
        return self


class ProtocolRunResult(_ProtocolModel):
    run_id: str = Field(min_length=1)
    suite_id: str = Field(min_length=1)
    framework_mode: Literal["confirmatory"] = FrameworkMode.CONFIRMATORY.value
    status: Literal[
        "planned",
        "success",
        "failed",
        "timed_out",
        "skipped_due_to_policy",
        "completed",
    ]
    report_dir: str | None = None
    metrics_path: str | None = None
    config_path: str | None = None
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
    def _validate_result(self) -> ProtocolRunResult:
        if self.framework_mode != FrameworkMode.CONFIRMATORY.value:
            raise ValueError("ProtocolRunResult.framework_mode must be 'confirmatory'.")
        normalized_status = normalize_run_status(self.status)
        if normalized_status == RUN_STATUS_SUCCESS and self.status == RUN_STATUS_COMPLETED_LEGACY:
            self.status = RUN_STATUS_SUCCESS

        if normalized_status in {RUN_STATUS_FAILED, RUN_STATUS_TIMED_OUT} and not self.error:
            raise ValueError(
                "ProtocolRunResult.error is required when status is failed or timed_out."
            )
        if normalized_status in {RUN_STATUS_FAILED, RUN_STATUS_TIMED_OUT}:
            if self.error_code is None:
                raise ValueError(
                    "ProtocolRunResult.error_code is required when status is failed or timed_out."
                )
            if self.error_type is None:
                raise ValueError(
                    "ProtocolRunResult.error_type is required when status is failed or timed_out."
                )
            if self.failure_stage is None:
                raise ValueError(
                    "ProtocolRunResult.failure_stage is required when status is failed or timed_out."
                )
        if normalized_status == RUN_STATUS_TIMED_OUT:
            if self.timeout_seconds is None:
                raise ValueError(
                    "ProtocolRunResult.timeout_seconds is required when status='timed_out'."
                )
            if self.elapsed_seconds is None:
                raise ValueError(
                    "ProtocolRunResult.elapsed_seconds is required when status='timed_out'."
                )
            if self.timeout_diagnostics_path is None:
                raise ValueError(
                    "ProtocolRunResult.timeout_diagnostics_path is required when status='timed_out'."
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
                        f"ProtocolRunResult.{field_name} must be null unless status is failed or timed_out."
                    )
        if normalized_status != RUN_STATUS_TIMED_OUT:
            for field_name in (
                "timeout_seconds",
                "elapsed_seconds",
                "timeout_diagnostics_path",
            ):
                if getattr(self, field_name) is not None:
                    raise ValueError(
                        f"ProtocolRunResult.{field_name} must be null unless status='timed_out'."
                    )
        if normalized_status != RUN_STATUS_SKIPPED_DUE_TO_POLICY and self.policy_reason is not None:
            raise ValueError(
                "ProtocolRunResult.policy_reason must be null unless status='skipped_due_to_policy'."
            )
        if (
            normalized_status == RUN_STATUS_SKIPPED_DUE_TO_POLICY
            and not str(self.policy_reason or "").strip()
        ):
            raise ValueError(
                "ProtocolRunResult.policy_reason is required when status='skipped_due_to_policy'."
            )
        return self
