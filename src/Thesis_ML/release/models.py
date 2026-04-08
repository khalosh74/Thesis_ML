from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _ReleaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class RunClass(StrEnum):
    SCRATCH = "scratch"
    EXPLORATORY = "exploratory"
    CANDIDATE = "candidate"
    OFFICIAL = "official"


class RunStatus(StrEnum):
    CREATED = "created"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    PROMOTED = "promoted"


class ReleaseStatus(StrEnum):
    FROZEN_CANDIDATE_ONLY = "frozen_candidate_only"
    OFFICIAL = "official"


class TransferPair(_ReleaseModel):
    train_subject: str = Field(min_length=1)
    test_subject: str = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_pair(self) -> TransferPair:
        if self.train_subject == self.test_subject:
            raise ValueError("transfer pair requires different train_subject/test_subject")
        return self


class ReleaseDatasetContract(_ReleaseModel):
    dataset_id: str = Field(min_length=1)
    dataset_contract_version: str = Field(min_length=1)
    dataset_fingerprint_required: bool = True
    require_spatial_compatibility_checks: bool = True
    minimum_subjects: int = Field(ge=1)
    minimum_sessions_per_subject: int = Field(ge=1)
    required_columns: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_required_columns(self) -> ReleaseDatasetContract:
        if len(set(self.required_columns)) != len(self.required_columns):
            raise ValueError("dataset_contract.required_columns must be unique")
        return self


class ReleaseTarget(_ReleaseModel):
    name: str = Field(min_length=1)
    mapping_path: str = Field(min_length=1)
    mapping_hash: str = Field(min_length=64, max_length=64)


class ReleaseScope(_ReleaseModel):
    main_tasks: list[str] = Field(min_length=1)
    main_modality: str = Field(min_length=1)
    within_subjects: list[str] = Field(min_length=1)
    tasks: list[str] | None = None
    modality: str | None = None
    subjects: list[str] | None = None
    target: str | None = None
    strict_scope: bool = True
    transfer_pairs: list[TransferPair] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_scope(self) -> ReleaseScope:
        if len(set(self.main_tasks)) != len(self.main_tasks):
            raise ValueError("scope.main_tasks must be unique")
        if len(set(self.within_subjects)) != len(self.within_subjects):
            raise ValueError("scope.within_subjects must be unique")
        if self.tasks is not None:
            if len(set(self.tasks)) != len(self.tasks):
                raise ValueError("scope.tasks must be unique when provided")
            if sorted(self.tasks) != sorted(self.main_tasks):
                raise ValueError("scope.tasks must match scope.main_tasks when both are provided")
        if self.subjects is not None:
            if len(set(self.subjects)) != len(self.subjects):
                raise ValueError("scope.subjects must be unique when provided")
            if sorted(self.subjects) != sorted(self.within_subjects):
                raise ValueError(
                    "scope.subjects must match scope.within_subjects when both are provided"
                )
        if self.modality is not None and self.modality != self.main_modality:
            raise ValueError(
                "scope.modality must match scope.main_modality when both are provided"
            )
        if self.strict_scope is not True:
            raise ValueError("scope.strict_scope must be true for release-science-v1")
        return self

    def effective_tasks(self) -> list[str]:
        return list(self.tasks if self.tasks is not None else self.main_tasks)

    def effective_subjects(self) -> list[str]:
        return list(self.subjects if self.subjects is not None else self.within_subjects)

    def effective_modality(self) -> str:
        return str(self.modality if self.modality is not None else self.main_modality)


class ReleaseFeaturePolicy(_ReleaseModel):
    feature_space: str = Field(min_length=1)
    preprocessing_strategy: str = Field(min_length=1)
    dimensionality_strategy: str = Field(min_length=1)


class ReleaseModelPolicy(_ReleaseModel):
    model_family: str = Field(min_length=1)
    class_weight_policy: str = Field(min_length=1)
    methodology_policy_name: str = Field(min_length=1)
    tuning_enabled: bool = False
    deterministic_execution: bool = True


class ReleasePrimaryAnalysis(_ReleaseModel):
    split: str = Field(min_length=1)
    metric: str = Field(min_length=1)
    secondary_metrics: list[str] = Field(default_factory=list)
    seed: int = Field(ge=0)

    @model_validator(mode="after")
    def _validate_metrics(self) -> ReleasePrimaryAnalysis:
        if len(set(self.secondary_metrics)) != len(self.secondary_metrics):
            raise ValueError("primary_analysis.secondary_metrics must be unique")
        return self


class ReleaseSecondaryAnalysis(_ReleaseModel):
    split: str = Field(min_length=1)


class ReleaseSplitPolicy(_ReleaseModel):
    primary_analysis: ReleasePrimaryAnalysis
    secondary_analysis: ReleaseSecondaryAnalysis


class ReleaseControls(_ReleaseModel):
    dummy_baseline: bool = True
    permutation_test: bool = True
    n_permutations: int = Field(ge=0)


class ReleaseMultiplicity(_ReleaseModel):
    primary_hypotheses: int = Field(ge=1)
    primary_alpha: float = Field(gt=0.0, le=1.0)
    secondary_policy: str = Field(min_length=1)
    exploratory_claims_allowed: bool = False


class ReleaseInterpretationLimits(_ReleaseModel):
    no_causal_claims: bool = True
    no_clinical_claims: bool = True
    no_localization_claims_from_coefficients: bool = True
    no_external_generalization_claim: bool = True
    secondary_results_not_primary_evidence: bool = True


class ReleaseReportingRequirements(_ReleaseModel):
    require_protocol_id: bool = True
    require_dataset_fingerprint: bool = True
    require_mapping_hash: bool = True
    require_interpretation_limits: bool = True
    require_deviation_log: bool = True


class ReleaseScience(_ReleaseModel):
    schema_version: str = "release-science-v1"
    release_id: str = Field(min_length=1)
    sample_unit: str = Field(min_length=1)
    dataset_contract: ReleaseDatasetContract
    target: ReleaseTarget
    scope: ReleaseScope
    feature_policy: ReleaseFeaturePolicy
    model_policy: ReleaseModelPolicy
    split_policy: ReleaseSplitPolicy
    controls: ReleaseControls
    multiplicity: ReleaseMultiplicity
    interpretation_limits: ReleaseInterpretationLimits
    reporting_requirements: ReleaseReportingRequirements


class ReleaseClaims(_ReleaseModel):
    schema_version: str = "release-claims-v1"
    primary_claim: str = Field(min_length=1)
    secondary_claim: str = Field(min_length=1)
    prohibited_claims: list[str] = Field(default_factory=list)
    evidence_boundaries: list[str] = Field(default_factory=list)
    result_usage_rules: list[str] = Field(default_factory=list)


class ReleaseExecution(_ReleaseModel):
    schema_version: str = "release-execution-v1"
    run_classes_allowed: list[RunClass]
    official_creation_policy: str
    hardware_mode: str
    max_parallel_runs: int = Field(ge=1)
    deterministic_compute: bool
    allow_backend_fallback: bool
    cache_policy: str = Field(min_length=1)
    run_root: str = Field(min_length=1)
    candidate_root: str = Field(min_length=1)
    official_root: str = Field(min_length=1)
    exploratory_root: str = Field(min_length=1)
    scratch_root: str = Field(min_length=1)
    default_behavior: str = Field(min_length=1)
    allow_resume_for: list[RunClass] = Field(default_factory=list)
    allow_force_for: list[RunClass] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_execution_policy(self) -> ReleaseExecution:
        if self.official_creation_policy != "promotion_only":
            raise ValueError("official_creation_policy must be 'promotion_only'")
        if self.hardware_mode != "cpu_only":
            raise ValueError("official release hardware_mode must be 'cpu_only'")
        if RunClass.OFFICIAL in set(self.run_classes_allowed):
            raise ValueError("execution.run_classes_allowed must not include 'official'")
        return self


class ReleaseEnvironment(_ReleaseModel):
    schema_version: str = "release-environment-v1"
    official_python: str = Field(min_length=1)
    requires_uv_lock: bool = True
    cpu_only_official: bool = True
    supported_os: list[str] = Field(default_factory=list)
    required_project_scripts: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_supported_os(self) -> ReleaseEnvironment:
        allowed = {"windows", "linux"}
        invalid = sorted(set(self.supported_os) - allowed)
        if invalid:
            raise ValueError(
                "environment.supported_os contains unsupported values: " + ", ".join(invalid)
            )
        return self


class ReleaseEvidence(_ReleaseModel):
    schema_version: str = "release-evidence-v1"
    required_run_artifacts: list[str] = Field(default_factory=list)
    required_release_artifacts: list[str] = Field(default_factory=list)
    required_control_artifacts: list[str] = Field(default_factory=list)
    required_manifest_fields: list[str] = Field(default_factory=list)
    verify_dataset_fingerprint: bool = True
    verify_science_hash: bool = True
    verify_execution_hash: bool = True
    verify_environment_hash: bool = True
    verify_claims_hash: bool = True

    @model_validator(mode="after")
    def _validate_artifacts(self) -> ReleaseEvidence:
        for field_name in (
            "required_run_artifacts",
            "required_release_artifacts",
            "required_control_artifacts",
            "required_manifest_fields",
        ):
            values = list(getattr(self, field_name))
            if not values:
                raise ValueError(f"{field_name} must not be empty")
            if len(set(values)) != len(values):
                raise ValueError(f"{field_name} must contain unique values")
        return self


class ReleaseBundle(_ReleaseModel):
    schema_version: str = "release-bundle-v1"
    release_id: str = Field(min_length=1)
    release_version: str = Field(min_length=1)
    status: ReleaseStatus
    title: str = Field(min_length=1)
    aliases: list[str] = Field(default_factory=list)
    science_path: str = Field(min_length=1)
    execution_path: str = Field(min_length=1)
    environment_path: str = Field(min_length=1)
    evidence_path: str = Field(min_length=1)
    claims_path: str = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_aliases(self) -> ReleaseBundle:
        if len(set(self.aliases)) != len(self.aliases):
            raise ValueError("release.aliases must be unique")
        return self


class ReleaseRegistry(_ReleaseModel):
    schema_version: str = "release-registry-v1"
    aliases: dict[str, str]


class DatasetManifest(_ReleaseModel):
    schema_version: str = "dataset-instance-v1"
    dataset_id: str = Field(min_length=1)
    dataset_contract_version: str = Field(min_length=1)
    dataset_fingerprint: str = Field(min_length=1)
    index_csv: str = Field(min_length=1)
    data_root: str = Field(min_length=1)
    cache_dir: str | None = None
    created_at: str = Field(min_length=1)
    source_extraction_version: str = Field(min_length=1)
    sample_unit: str = Field(min_length=1)
    required_columns: list[str] = Field(min_length=1)
    subject_count: int = Field(ge=1)
    session_counts_by_subject: dict[str, int] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_sessions(self) -> DatasetManifest:
        if len(set(self.required_columns)) != len(self.required_columns):
            raise ValueError("dataset_manifest.required_columns must be unique")
        for subject, count in self.session_counts_by_subject.items():
            if not str(subject).strip():
                raise ValueError("dataset_manifest.session_counts_by_subject has blank subject key")
            if int(count) <= 0:
                raise ValueError(
                    "dataset_manifest.session_counts_by_subject values must be positive"
                )
        return self


class RunManifest(_ReleaseModel):
    schema_version: str = "release-run-manifest-v1"
    run_id: str = Field(min_length=1)
    run_class: RunClass
    release_id: str = Field(min_length=1)
    release_version: str = Field(min_length=1)
    release_hash: str = Field(min_length=64, max_length=64)
    science_hash: str = Field(min_length=64, max_length=64)
    execution_hash: str = Field(min_length=64, max_length=64)
    environment_hash: str = Field(min_length=64, max_length=64)
    evidence_hash: str = Field(min_length=64, max_length=64)
    claims_hash: str = Field(min_length=64, max_length=64)
    dataset_manifest_path: str = Field(min_length=1)
    dataset_fingerprint: str = Field(min_length=1)
    git_commit: str = ""
    git_dirty: bool = False
    python_version: str = Field(min_length=1)
    platform: str = Field(min_length=1)
    timestamp_utc: str = Field(min_length=1)
    parent_run_id: str | None = None
    status: RunStatus
    promotable: bool = False
    official: bool = False
    cache_policy: str = Field(min_length=1)
    command_line: str = ""
    evidence_verified: bool = False
    compiled_scope_manifest_path: str | None = None
    selected_samples_path: str | None = None
    selected_sample_ids_sha256: str | None = Field(default=None, min_length=64, max_length=64)
    scope_alignment_passed: bool = False


class ReleaseManifest(_ReleaseModel):
    schema_version: str = "release-manifest-v1"
    release_id: str = Field(min_length=1)
    release_version: str = Field(min_length=1)
    generated_at_utc: str = Field(min_length=1)
    release_json_path: str = Field(min_length=1)
    science_json_path: str = Field(min_length=1)
    execution_json_path: str = Field(min_length=1)
    environment_json_path: str = Field(min_length=1)
    evidence_json_path: str = Field(min_length=1)
    claims_json_path: str = Field(min_length=1)
    release_hash: str = Field(min_length=64, max_length=64)
    science_hash: str = Field(min_length=64, max_length=64)
    execution_hash: str = Field(min_length=64, max_length=64)
    environment_hash: str = Field(min_length=64, max_length=64)
    evidence_hash: str = Field(min_length=64, max_length=64)
    claims_hash: str = Field(min_length=64, max_length=64)


class PromotionManifest(_ReleaseModel):
    schema_version: str = "release-promotion-manifest-v1"
    release_id: str = Field(min_length=1)
    release_version: str = Field(min_length=1)
    official_run_id: str = Field(min_length=1)
    candidate_run_id: str = Field(min_length=1)
    candidate_run_path: str = Field(min_length=1)
    official_run_path: str = Field(min_length=1)
    timestamp_utc: str = Field(min_length=1)
    release_hash: str = Field(min_length=64, max_length=64)
    science_hash: str = Field(min_length=64, max_length=64)
    execution_hash: str = Field(min_length=64, max_length=64)
    environment_hash: str = Field(min_length=64, max_length=64)
    evidence_hash: str = Field(min_length=64, max_length=64)
    claims_hash: str = Field(min_length=64, max_length=64)
    dataset_fingerprint: str = Field(min_length=1)


def utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


__all__ = [
    "DatasetManifest",
    "PromotionManifest",
    "ReleaseBundle",
    "ReleaseClaims",
    "ReleaseEnvironment",
    "ReleaseEvidence",
    "ReleaseExecution",
    "ReleaseManifest",
    "ReleaseRegistry",
    "ReleaseScience",
    "RunClass",
    "RunManifest",
    "RunStatus",
    "TransferPair",
    "utc_now_iso",
]
