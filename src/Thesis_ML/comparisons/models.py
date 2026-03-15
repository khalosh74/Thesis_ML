from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from Thesis_ML.config.framework_mode import FrameworkMode
from Thesis_ML.experiments.model_factory import MODEL_NAMES
from Thesis_ML.protocols.models import SUPPORTED_CV_MODES, SUPPORTED_PRIMARY_METRICS

COMPARISON_SCHEMA_VERSION = "comparison-spec-v1"
SUPPORTED_COMPARISON_SCHEMA_VERSIONS = frozenset({COMPARISON_SCHEMA_VERSION})
REQUIRED_COMPARISON_ARTIFACTS = (
    "comparison.json",
    "compiled_comparison_manifest.json",
    "comparison_summary.json",
    "execution_status.json",
    "report_index.csv",
)
REQUIRED_COMPARISON_RUN_ARTIFACTS = (
    "config.json",
    "metrics.json",
    "fold_metrics.csv",
    "fold_splits.csv",
    "predictions.csv",
    "spatial_compatibility_report.json",
    "interpretability_summary.json",
)


class _ComparisonModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class ComparisonStatus(StrEnum):
    DRAFT = "draft"
    LOCKED = "locked"
    EXECUTED = "executed"
    RETIRED = "retired"


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
            raise ValueError("seed_policy.global_seed must be >= 0.")
        return self


class ComparisonControlPolicy(_ComparisonModel):
    permutation_enabled: bool = False
    permutation_metric: str = "balanced_accuracy"
    n_permutations: int = 0
    dummy_baseline_enabled: bool = False

    @model_validator(mode="after")
    def _validate_controls(self) -> ComparisonControlPolicy:
        if self.permutation_metric not in SUPPORTED_PRIMARY_METRICS:
            allowed = ", ".join(sorted(SUPPORTED_PRIMARY_METRICS))
            raise ValueError(
                f"Unsupported permutation_metric '{self.permutation_metric}'. Allowed values: {allowed}."
            )
        if self.permutation_enabled and int(self.n_permutations) <= 0:
            raise ValueError("n_permutations must be > 0 when permutation_enabled is true.")
        if not self.permutation_enabled and int(self.n_permutations) < 0:
            raise ValueError("n_permutations must be >= 0.")
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
            supported = set(MODEL_NAMES)
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
    primary_metric: str = "balanced_accuracy"
    seed_policy: ComparisonSeedPolicy = Field(default_factory=ComparisonSeedPolicy)
    control_policy: ComparisonControlPolicy = Field(default_factory=ComparisonControlPolicy)
    interpretability_policy: ComparisonInterpretabilityPolicy = Field(
        default_factory=ComparisonInterpretabilityPolicy
    )
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
        if self.primary_metric not in SUPPORTED_PRIMARY_METRICS:
            allowed = ", ".join(sorted(SUPPORTED_PRIMARY_METRICS))
            raise ValueError(
                f"Unsupported primary_metric '{self.primary_metric}'. Allowed values: {allowed}."
            )
        if (
            self.control_policy.permutation_enabled
            and self.control_policy.permutation_metric != self.primary_metric
        ):
            raise ValueError(
                "permutation_metric must match primary_metric for locked comparison runs."
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
        if self.model not in set(MODEL_NAMES):
            allowed = ", ".join(sorted(MODEL_NAMES))
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
        if self.cv_mode == "within_subject_loso_session":
            if self.subject is None:
                raise ValueError(f"CompiledComparisonRunSpec '{self.run_id}' requires subject.")
        if self.cv_mode == "frozen_cross_person_transfer":
            if self.train_subject is None or self.test_subject is None:
                raise ValueError(
                    f"CompiledComparisonRunSpec '{self.run_id}' requires train_subject and test_subject."
                )
        return self


class CompiledComparisonManifest(_ComparisonModel):
    compiled_schema_version: str = "comparison-compiled-v1"
    framework_mode: Literal["locked_comparison"] = FrameworkMode.LOCKED_COMPARISON.value
    comparison_id: str = Field(min_length=1)
    comparison_version: str = Field(min_length=1)
    status: ComparisonStatus
    comparison_dimension: str = Field(min_length=1)
    variant_ids: list[str] = Field(min_length=1)
    runs: list[CompiledComparisonRunSpec] = Field(min_length=1)
    claim_to_run_map: dict[str, list[str]]
    required_comparison_artifacts: list[str] = Field(
        default_factory=lambda: list(REQUIRED_COMPARISON_ARTIFACTS)
    )
    required_run_artifacts: list[str] = Field(
        default_factory=lambda: list(REQUIRED_COMPARISON_RUN_ARTIFACTS)
    )

    @model_validator(mode="after")
    def _validate_manifest(self) -> CompiledComparisonManifest:
        run_ids = [run.run_id for run in self.runs]
        if len(set(run_ids)) != len(run_ids):
            raise ValueError("CompiledComparisonManifest.runs contains duplicate run_id values.")
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
    metrics: dict[str, float | int | str | bool | None] | None = None
