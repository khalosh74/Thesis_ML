from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from Thesis_ML.config.framework_mode import FrameworkMode
from Thesis_ML.config.methodology import (
    ClassWeightPolicy,
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
from Thesis_ML.experiments.model_factory import ALL_MODEL_NAMES

SUPPORTED_CV_MODES = frozenset({"within_subject_loso_session", "frozen_cross_person_transfer"})
SUPPORTED_PRIMARY_METRICS = SUPPORTED_CLASSIFICATION_METRICS
REQUIRED_PROTOCOL_ARTIFACTS = (
    "protocol.json",
    "compiled_protocol_manifest.json",
    "claim_to_run_map.json",
    "suite_summary.json",
    "execution_status.json",
    "report_index.csv",
)
REQUIRED_RUN_ARTIFACTS_BASELINE = (
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


class _ProtocolModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


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
            for mode in self.modes:
                if mode not in SUPPORTED_CV_MODES:
                    allowed = ", ".join(sorted(SUPPORTED_CV_MODES))
                    raise ValueError(
                        f"Unsupported interpretability mode '{mode}'. Allowed values: {allowed}."
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
            "primary_metric_name",
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
            for name in ("config.json", "metrics.json")
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
            "primary_metric_name",
            "protocol_id",
            "protocol_version",
            "suite_id",
        ):
            if key not in self.required_run_metadata_fields:
                raise ValueError(
                    "artifact_contract.required_run_metadata_fields is missing required key: " + key
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
    scientific_contract: ScientificContract
    split_policy: SplitPolicy
    model_policy: ModelPolicy
    methodology_policy: MethodologyPolicy
    metric_policy: MetricPolicy = Field(default_factory=MetricPolicy)
    subgroup_reporting_policy: SubgroupReportingPolicy = Field(
        default_factory=SubgroupReportingPolicy
    )
    control_policy: ControlPolicy
    interpretability_policy: InterpretabilityPolicy
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
        if self.methodology_policy.policy_name == MethodologyPolicyName.FIXED_BASELINES_ONLY:
            if self.model_policy.selection_strategy != ModelSelectionStrategy.FIXED_BASELINES:
                raise ValueError(
                    "fixed_baselines_only requires model_policy.selection_strategy='fixed_baselines'."
                )
            if self.model_policy.tuning_enabled:
                raise ValueError("fixed_baselines_only requires model_policy.tuning_enabled=false.")
            if self.model_policy.nested_grouped_cv:
                raise ValueError("fixed_baselines_only requires model_policy.nested_grouped_cv=false.")
        if self.methodology_policy.policy_name == MethodologyPolicyName.GROUPED_NESTED_TUNING:
            if self.model_policy.selection_strategy != ModelSelectionStrategy.NESTED_TUNED:
                raise ValueError(
                    "grouped_nested_tuning requires model_policy.selection_strategy='nested_tuned'."
                )
            if not self.model_policy.tuning_enabled:
                raise ValueError("grouped_nested_tuning requires model_policy.tuning_enabled=true.")
            if not self.model_policy.nested_grouped_cv:
                raise ValueError("grouped_nested_tuning requires model_policy.nested_grouped_cv=true.")

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
    cv_mode: Literal["within_subject_loso_session", "frozen_cross_person_transfer"]
    subject: str | None = None
    train_subject: str | None = None
    test_subject: str | None = None
    filter_task: str | None = None
    filter_modality: str | None = None
    seed: int
    primary_metric: str = "balanced_accuracy"
    controls: CompiledRunControls = Field(default_factory=CompiledRunControls)
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
            "primary_metric_name",
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
            raise ValueError("CompiledProtocolManifest.required_run_metadata_fields must not be empty.")
        return self


class ProtocolRunResult(_ProtocolModel):
    run_id: str = Field(min_length=1)
    suite_id: str = Field(min_length=1)
    framework_mode: Literal["confirmatory"] = FrameworkMode.CONFIRMATORY.value
    status: Literal["planned", "completed", "failed"]
    report_dir: str | None = None
    metrics_path: str | None = None
    config_path: str | None = None
    error: str | None = None
    metrics: dict[str, float | int | str | bool | None | dict[str, Any]] | None = None

    @model_validator(mode="after")
    def _validate_result(self) -> ProtocolRunResult:
        if self.framework_mode != FrameworkMode.CONFIRMATORY.value:
            raise ValueError("ProtocolRunResult.framework_mode must be 'confirmatory'.")
        if self.status == "failed" and not self.error:
            raise ValueError("ProtocolRunResult.error is required when status='failed'.")
        if self.status != "failed" and self.error is not None:
            raise ValueError("ProtocolRunResult.error must be null unless status='failed'.")
        return self
