from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from Thesis_ML.config.metric_policy import validate_metric_name
from Thesis_ML.config.schema_versions import (
    COMPILED_MANIFEST_SCHEMA_VERSION,
    SUMMARY_RESULT_SCHEMA_VERSION,
    SUPPORTED_COMPILED_MANIFEST_SCHEMA_VERSIONS,
    SUPPORTED_SUMMARY_RESULT_SCHEMA_VERSIONS,
)


class SectionName(StrEnum):
    DATASET_SELECTION = "dataset_selection"
    FEATURE_CACHE_BUILD = "feature_cache_build"
    FEATURE_MATRIX_LOAD = "feature_matrix_load"
    SPATIAL_VALIDATION = "spatial_validation"
    MODEL_FIT = "model_fit"
    EVALUATION = "evaluation"
    INTERPRETABILITY = "interpretability"


class ReusePolicy(StrEnum):
    AUTO = "auto"
    REQUIRE_EXPLICIT_BASE = "require_explicit_base"
    DISALLOW = "disallow"


class SearchMode(StrEnum):
    DETERMINISTIC_GRID = "deterministic_grid"
    OPTUNA = "optuna"


class StudyType(StrEnum):
    SINGLE_EXPERIMENT = "single_experiment"
    FULL_FACTORIAL = "full_factorial"
    FRACTIONAL_FACTORIAL = "fractional_factorial"
    CUSTOM_MATRIX = "custom_matrix"


class StudyIntent(StrEnum):
    EXPLORATORY = "exploratory"
    CONFIRMATORY = "confirmatory"


class AggregationLevel(StrEnum):
    TRIAL = "trial"
    CELL = "cell"
    SUBJECT = "subject"
    FOLD = "fold"
    GROUP = "group"


class UncertaintyMethod(StrEnum):
    NONE = "none"
    STANDARD_ERROR = "standard_error"
    CONFIDENCE_INTERVAL = "confidence_interval"
    BOOTSTRAP = "bootstrap"
    PERMUTATION = "permutation"


class FactorType(StrEnum):
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"
    ORDINAL = "ordinal"
    NUMERIC = "numeric"


def supported_sections() -> list[SectionName]:
    return list(SectionName)


_SECTION_ORDER_MAP = {section.value: idx for idx, section in enumerate(SectionName)}


def _section_value(value: SectionName | str) -> str:
    if isinstance(value, SectionName):
        return value.value
    return str(value)


class _ContractModel(BaseModel):
    model_config = ConfigDict(
        extra="allow",
        use_enum_values=True,
        str_strip_whitespace=True,
    )


class ArtifactRef(_ContractModel):
    name: str = Field(min_length=1)
    path: str | None = None
    section: SectionName | None = None
    description: str | None = None
    required: bool = False


class TrialSpec(_ContractModel):
    experiment_id: str = Field(min_length=1)
    template_id: str = Field(default="template", min_length=1)
    supported: bool = True
    unsupported_reason: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    expand: dict[str, str] = Field(default_factory=dict)
    sections: list[SectionName] = Field(default_factory=supported_sections)
    artifacts: list[ArtifactRef] = Field(default_factory=list)
    start_section: SectionName = SectionName.DATASET_SELECTION
    end_section: SectionName = SectionName.EVALUATION
    base_artifact_id: str | None = None
    reuse_policy: ReusePolicy = ReusePolicy.AUTO
    search_space_id: str | None = None
    study_id: str | None = None
    trial_id: str | None = None
    cell_id: str | None = None
    repeat_id: int | None = None
    seed: int | None = None
    factor_settings: dict[str, Any] = Field(default_factory=dict)
    fixed_controls: dict[str, Any] = Field(default_factory=dict)
    design_metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_supported_template_params(self) -> TrialSpec:
        if self.supported:
            missing = [
                name
                for name in ("target", "model", "cv")
                if not str(self.params.get(name, "")).strip()
            ]
            if missing:
                raise ValueError(
                    "Supported trial template must define params keys: " + ", ".join(missing)
                )
        if self.base_artifact_id is not None and not self.base_artifact_id.strip():
            raise ValueError("base_artifact_id must be non-empty when provided.")
        if self.search_space_id is not None and not self.search_space_id.strip():
            raise ValueError("search_space_id must be non-empty when provided.")
        if self.study_id is not None and not self.study_id.strip():
            raise ValueError("study_id must be non-empty when provided.")
        if self.trial_id is not None and not self.trial_id.strip():
            raise ValueError("trial_id must be non-empty when provided.")
        if self.cell_id is not None and not self.cell_id.strip():
            raise ValueError("cell_id must be non-empty when provided.")
        if self.repeat_id is not None and int(self.repeat_id) <= 0:
            raise ValueError("repeat_id must be > 0 when provided.")
        if self.seed is not None and int(self.seed) < 0:
            raise ValueError("seed must be >= 0 when provided.")
        for key in (
            "persist_models",
            "persist_fold_models",
            "persist_final_refit_model",
        ):
            if key in self.params and not isinstance(self.params.get(key), bool):
                raise ValueError(f"TrialSpec.params['{key}'] must be boolean when provided.")
        return self


class FactorSpec(_ContractModel):
    study_id: str = Field(min_length=1)
    factor_name: str = Field(min_length=1)
    section_name: SectionName | None = None
    parameter_path: str = Field(min_length=1)
    factor_type: FactorType = FactorType.CATEGORICAL
    levels: list[Any] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_levels(self) -> FactorSpec:
        if not self.levels:
            raise ValueError(
                f"Factor '{self.factor_name}' in study '{self.study_id}' must define levels."
            )
        if self.factor_type == FactorType.BOOLEAN:
            normalized = {str(value).strip().lower() for value in self.levels}
            if not normalized.issubset({"true", "false", "0", "1", "yes", "no"}):
                raise ValueError(
                    f"Boolean factor '{self.factor_name}' in study '{self.study_id}' "
                    "must use boolean-like levels."
                )
        return self


class FixedControlSpec(_ContractModel):
    study_id: str = Field(min_length=1)
    parameter_path: str = Field(min_length=1)
    value: Any


class ConstraintSpec(_ContractModel):
    study_id: str = Field(min_length=1)
    if_factor: str = Field(min_length=1)
    if_level: Any
    disallow_factor: str = Field(min_length=1)
    disallow_level: Any
    reason: str | None = None


class BlockingReplicationSpec(_ContractModel):
    study_id: str = Field(min_length=1)
    block_type: str = Field(min_length=1)
    block_value: str | None = None
    repeat_id: int = 1
    seed: int | None = None

    @model_validator(mode="after")
    def _validate_repeat_seed(self) -> BlockingReplicationSpec:
        if int(self.repeat_id) <= 0:
            raise ValueError("repeat_id must be > 0.")
        if self.seed is not None and int(self.seed) < 0:
            raise ValueError("seed must be >= 0 when provided.")
        return self


class StudyDesignSpec(_ContractModel):
    study_id: str = Field(min_length=1)
    study_name: str = Field(min_length=1)
    enabled: bool = False
    study_type: StudyType = StudyType.SINGLE_EXPERIMENT
    intent: StudyIntent = StudyIntent.EXPLORATORY
    question: str | None = None
    generalization_claim: str | None = None
    start_section: SectionName = SectionName.DATASET_SELECTION
    end_section: SectionName = SectionName.EVALUATION
    base_artifact_id: str | None = None
    primary_metric: str = "balanced_accuracy"
    secondary_metrics: str | None = None
    cv_scheme: str | None = None
    nested_cv: bool | None = None
    external_validation_planned: bool | None = None
    blocking_strategy: str | None = None
    randomization_strategy: str | None = None
    replication_mode: str = "none"
    replication_strategy: str | None = None
    num_repeats: int = 1
    random_seed_policy: str = "fixed"
    stopping_rule: str | None = None
    notes: str | None = None
    factors: list[FactorSpec] = Field(default_factory=list)
    fixed_controls: list[FixedControlSpec] = Field(default_factory=list)
    constraints: list[ConstraintSpec] = Field(default_factory=list)
    blocking_replication: list[BlockingReplicationSpec] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_study(self) -> StudyDesignSpec:
        if "primary_metric" not in self.model_fields_set:
            raise ValueError(f"Study '{self.study_id}' must explicitly declare primary_metric.")
        if not str(self.primary_metric).strip():
            raise ValueError(f"Study '{self.study_id}' requires non-empty primary_metric.")
        self.primary_metric = validate_metric_name(str(self.primary_metric).strip())
        start_key = _section_value(self.start_section)
        end_key = _section_value(self.end_section)
        start_idx = _SECTION_ORDER_MAP[start_key]
        end_idx = _SECTION_ORDER_MAP[end_key]
        if start_idx > end_idx:
            raise ValueError(
                f"Invalid section range in study '{self.study_id}': "
                f"start_section='{start_key}' is after "
                f"end_section='{end_key}'."
            )
        if int(self.num_repeats) <= 0:
            raise ValueError(f"Study '{self.study_id}' requires num_repeats > 0.")
        if self.base_artifact_id is not None and not self.base_artifact_id.strip():
            raise ValueError("base_artifact_id must be non-empty when provided.")
        if self.study_type == StudyType.FULL_FACTORIAL and not self.factors:
            raise ValueError(
                f"Study '{self.study_id}' uses full_factorial but no factors were defined."
            )
        return self


class StudyRigorChecklistSpec(_ContractModel):
    study_id: str = Field(min_length=1)
    leakage_risk_reviewed: bool
    deployment_boundary_defined: bool
    unit_of_analysis_defined: bool
    data_hierarchy_defined: bool
    missing_data_plan: str = Field(min_length=1)
    class_imbalance_plan: str = Field(min_length=1)
    subgroup_plan: str = Field(min_length=1)
    fairness_or_applicability_notes: str | None = None
    reporting_checklist_completed: bool
    risk_of_bias_reviewed: bool
    confirmatory_lock_applied: bool
    analysis_notes: str | None = None


class AnalysisPlanSpec(_ContractModel):
    study_id: str = Field(min_length=1)
    primary_contrast: str | None = None
    secondary_contrasts: str | None = None
    aggregation_level: AggregationLevel = AggregationLevel.CELL
    uncertainty_method: UncertaintyMethod = UncertaintyMethod.NONE
    multiplicity_handling: str | None = None
    interaction_reporting_policy: str | None = None
    interpretation_rules: str | None = None
    notes: str | None = None


class StudyReviewSummary(_ContractModel):
    study_id: str = Field(min_length=1)
    study_name: str = Field(min_length=1)
    intent: StudyIntent = StudyIntent.EXPLORATORY
    question: str | None = None
    generalization_claim: str | None = None
    start_section: SectionName = SectionName.DATASET_SELECTION
    end_section: SectionName = SectionName.EVALUATION
    factors: dict[str, list[Any]] = Field(default_factory=dict)
    fixed_controls: dict[str, Any] = Field(default_factory=dict)
    blocked_constraints: list[str] = Field(default_factory=list)
    excluded_combination_count: int = 0
    expected_design_cells: int = 0
    expected_trials: int = 0
    primary_metric: str | None = None
    secondary_metrics: str | None = None
    cv_scheme: str | None = None
    nested_cv: bool | None = None
    external_validation_planned: bool | None = None
    blocking_strategy: str | None = None
    randomization_strategy: str | None = None
    replication_strategy: str | None = None
    replication_mode: str | None = None
    num_repeats: int = 1
    random_seed_policy: str | None = None
    rigor_checklist_status: str = "missing"
    analysis_plan_status: str = "missing"
    execution_eligibility_status: str = "blocked"
    execution_disposition: Literal["allowed", "warning", "blocked"] = "blocked"
    warning_count: int = 0
    error_count: int = 0
    missing_fields: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_counts(self) -> StudyReviewSummary:
        if self.expected_design_cells < 0:
            raise ValueError("expected_design_cells must be >= 0.")
        if self.expected_trials < 0:
            raise ValueError("expected_trials must be >= 0.")
        if self.excluded_combination_count < 0:
            raise ValueError("excluded_combination_count must be >= 0.")
        if self.warning_count < 0:
            raise ValueError("warning_count must be >= 0.")
        if self.error_count < 0:
            raise ValueError("error_count must be >= 0.")
        return self


class GeneratedDesignCell(_ContractModel):
    study_id: str = Field(min_length=1)
    trial_id: str = Field(min_length=1)
    cell_id: str = Field(min_length=1)
    factor_settings: dict[str, Any] = Field(default_factory=dict)
    start_section: SectionName = SectionName.DATASET_SELECTION
    end_section: SectionName = SectionName.EVALUATION
    base_artifact_id: str | None = None
    resolved_params: dict[str, Any] = Field(default_factory=dict)
    status: Literal["planned", "dry_run", "completed", "failed", "blocked"] = "planned"
    repeat_id: int = 1
    seed: int | None = None
    notes: str | None = None

    @model_validator(mode="after")
    def _validate_generated_cell(self) -> GeneratedDesignCell:
        start_key = _section_value(self.start_section)
        end_key = _section_value(self.end_section)
        start_idx = _SECTION_ORDER_MAP[start_key]
        end_idx = _SECTION_ORDER_MAP[end_key]
        if start_idx > end_idx:
            raise ValueError(
                f"Invalid section range for trial '{self.trial_id}': {start_key}->{end_key}."
            )
        missing = [
            name
            for name in ("target", "cv", "model")
            if not str(self.resolved_params.get(name, "")).strip()
        ]
        if missing:
            raise ValueError(
                f"Generated design cell '{self.trial_id}' is missing required params: "
                + ", ".join(missing)
            )
        if int(self.repeat_id) <= 0:
            raise ValueError("repeat_id must be > 0.")
        if self.seed is not None and int(self.seed) < 0:
            raise ValueError("seed must be >= 0 when provided.")
        return self


class EffectSummary(_ContractModel):
    study_id: str = Field(min_length=1)
    summary_type: str = Field(min_length=1)
    factor_keys: list[str] = Field(default_factory=list)
    factor_levels: dict[str, Any] = Field(default_factory=dict)
    primary_metric_name: str = "balanced_accuracy"
    primary_metric_value: float | None = None
    best_trial_id: str | None = None
    n_trials: int = 0
    descriptive_only: bool = True
    notes: str | None = None

    @model_validator(mode="after")
    def _validate_effect_summary(self) -> EffectSummary:
        if "primary_metric_name" not in self.model_fields_set:
            raise ValueError("EffectSummary must explicitly declare primary_metric_name.")
        self.primary_metric_name = validate_metric_name(str(self.primary_metric_name).strip())
        if int(self.n_trials) < 0:
            raise ValueError("n_trials must be >= 0.")
        if self.best_trial_id is not None and not self.best_trial_id.strip():
            raise ValueError("best_trial_id must be non-empty when provided.")
        return self


class SearchDimensionSpec(_ContractModel):
    parameter_name: str = Field(min_length=1)
    values: list[Any] = Field(default_factory=list)
    parameter_scope: str = "parameter"

    @model_validator(mode="after")
    def _validate_values(self) -> SearchDimensionSpec:
        if not self.values:
            raise ValueError("Search dimension values must contain at least one value.")
        return self


class SearchSpaceSpec(_ContractModel):
    search_space_id: str = Field(min_length=1)
    enabled: bool = True
    optimization_mode: SearchMode = SearchMode.DETERMINISTIC_GRID
    objective_metric: str = "balanced_accuracy"
    max_trials: int | None = None
    dimensions: list[SearchDimensionSpec] = Field(default_factory=list)
    notes: str | None = None

    @model_validator(mode="after")
    def _validate_shape(self) -> SearchSpaceSpec:
        if "objective_metric" not in self.model_fields_set:
            raise ValueError(
                f"Search space '{self.search_space_id}' must explicitly declare objective_metric."
            )
        self.objective_metric = validate_metric_name(str(self.objective_metric).strip())
        if self.enabled and not self.dimensions:
            raise ValueError(
                f"Enabled search space '{self.search_space_id}' must define at least one dimension."
            )
        if self.max_trials is not None and self.max_trials <= 0:
            raise ValueError("max_trials must be > 0 when provided.")
        return self


class ExperimentSpec(_ContractModel):
    experiment_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    stage: str = Field(min_length=1)
    decision_id: str | None = None
    manipulated_factor: str | None = None
    primary_metric: str = "balanced_accuracy"
    executable_now: bool = True
    execution_status: str = "unknown"
    blocked_reasons: list[str] = Field(default_factory=list)
    notes: str | None = None
    section_plan: list[SectionName] = Field(default_factory=supported_sections)
    variant_templates: list[TrialSpec] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_trial_experiment_ids(self) -> ExperimentSpec:
        if "primary_metric" not in self.model_fields_set:
            raise ValueError(
                f"Experiment '{self.experiment_id}' must explicitly declare primary_metric."
            )
        if not str(self.primary_metric).strip():
            raise ValueError(
                f"Experiment '{self.experiment_id}' requires non-empty primary_metric."
            )
        self.primary_metric = validate_metric_name(str(self.primary_metric).strip())
        mismatched = [
            trial.template_id
            for trial in self.variant_templates
            if trial.experiment_id != self.experiment_id
        ]
        if mismatched:
            raise ValueError(
                "All trial specs must match parent experiment_id; mismatched templates: "
                + ", ".join(mismatched)
            )
        return self


class CompiledStudyManifest(_ContractModel):
    schema_version: str
    compiled_manifest_schema_version: str = COMPILED_MANIFEST_SCHEMA_VERSION
    description: str | None = None
    source_registry_path: str | None = None
    compiled_at_utc: str
    supported_sections: list[SectionName] = Field(default_factory=supported_sections)
    experiments: list[ExperimentSpec]
    trial_specs: list[TrialSpec]
    search_spaces: list[SearchSpaceSpec] = Field(default_factory=list)
    study_designs: list[StudyDesignSpec] = Field(default_factory=list)
    study_rigor_checklists: list[StudyRigorChecklistSpec] = Field(default_factory=list)
    analysis_plans: list[AnalysisPlanSpec] = Field(default_factory=list)
    study_reviews: list[StudyReviewSummary] = Field(default_factory=list)
    generated_design_matrix: list[GeneratedDesignCell] = Field(default_factory=list)
    effect_summaries: list[EffectSummary] = Field(default_factory=list)
    validation_warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_trial_coverage(self) -> CompiledStudyManifest:
        if self.compiled_manifest_schema_version not in SUPPORTED_COMPILED_MANIFEST_SCHEMA_VERSIONS:
            supported = ", ".join(sorted(SUPPORTED_COMPILED_MANIFEST_SCHEMA_VERSIONS))
            raise ValueError(
                "Unsupported compiled manifest schema version "
                f"'{self.compiled_manifest_schema_version}'. Supported versions: {supported}"
            )
        known_experiments = {experiment.experiment_id for experiment in self.experiments}
        unknown_trials = sorted(
            {
                trial.template_id
                for trial in self.trial_specs
                if trial.experiment_id not in known_experiments
            }
        )
        if unknown_trials:
            raise ValueError(
                "Compiled trial specs reference unknown experiments for templates: "
                + ", ".join(unknown_trials)
            )
        known_spaces = {space.search_space_id for space in self.search_spaces}
        unknown_search_space_refs = sorted(
            {
                str(trial.search_space_id)
                for trial in self.trial_specs
                if trial.search_space_id and str(trial.search_space_id) not in known_spaces
            }
        )
        if unknown_search_space_refs:
            raise ValueError(
                "Compiled trial specs reference unknown search_space_id values: "
                + ", ".join(unknown_search_space_refs)
            )
        known_studies = {study.study_id for study in self.study_designs}
        unknown_study_refs = sorted(
            {
                str(trial.study_id)
                for trial in self.trial_specs
                if trial.study_id and str(trial.study_id) not in known_studies
            }
        )
        if unknown_study_refs:
            raise ValueError(
                "Compiled trial specs reference unknown study_id values: "
                + ", ".join(unknown_study_refs)
            )
        unknown_generated_studies = sorted(
            {
                cell.study_id
                for cell in self.generated_design_matrix
                if cell.study_id not in known_studies
            }
        )
        if unknown_generated_studies:
            raise ValueError(
                "Generated design matrix references unknown study_id values: "
                + ", ".join(unknown_generated_studies)
            )
        unknown_rigor_studies = sorted(
            {
                checklist.study_id
                for checklist in self.study_rigor_checklists
                if checklist.study_id not in known_studies
            }
        )
        if unknown_rigor_studies:
            raise ValueError(
                "Study rigor checklist references unknown study_id values: "
                + ", ".join(unknown_rigor_studies)
            )
        unknown_analysis_studies = sorted(
            {plan.study_id for plan in self.analysis_plans if plan.study_id not in known_studies}
        )
        if unknown_analysis_studies:
            raise ValueError(
                "Analysis plans reference unknown study_id values: "
                + ", ".join(unknown_analysis_studies)
            )
        duplicate_rigor_ids = sorted(
            {
                checklist.study_id
                for checklist in self.study_rigor_checklists
                if sum(
                    1
                    for other in self.study_rigor_checklists
                    if other.study_id == checklist.study_id
                )
                > 1
            }
        )
        if duplicate_rigor_ids:
            raise ValueError(
                "Study rigor checklist has duplicate entries for study_id values: "
                + ", ".join(duplicate_rigor_ids)
            )
        duplicate_analysis_ids = sorted(
            {
                plan.study_id
                for plan in self.analysis_plans
                if sum(1 for other in self.analysis_plans if other.study_id == plan.study_id) > 1
            }
        )
        if duplicate_analysis_ids:
            raise ValueError(
                "Analysis plans have duplicate entries for study_id values: "
                + ", ".join(duplicate_analysis_ids)
            )
        unknown_review_studies = sorted(
            {
                review.study_id
                for review in self.study_reviews
                if review.study_id not in known_studies
            }
        )
        if unknown_review_studies:
            raise ValueError(
                "Study review summaries reference unknown study_id values: "
                + ", ".join(unknown_review_studies)
            )
        duplicate_review_ids = sorted(
            {
                review.study_id
                for review in self.study_reviews
                if sum(1 for other in self.study_reviews if other.study_id == review.study_id) > 1
            }
        )
        if duplicate_review_ids:
            raise ValueError(
                "Study review summaries have duplicate entries for study_id values: "
                + ", ".join(duplicate_review_ids)
            )
        known_trial_ids = {str(trial.trial_id) for trial in self.trial_specs if trial.trial_id}
        unknown_effect_trials = sorted(
            {
                str(summary.best_trial_id)
                for summary in self.effect_summaries
                if summary.best_trial_id and str(summary.best_trial_id) not in known_trial_ids
            }
        )
        if unknown_effect_trials:
            raise ValueError(
                "Effect summaries reference unknown best_trial_id values: "
                + ", ".join(unknown_effect_trials)
            )
        return self


class TrialResultSummary(_ContractModel):
    summary_result_schema_version: str = SUMMARY_RESULT_SCHEMA_VERSION
    experiment_id: str = Field(min_length=1)
    trial_id: str = Field(min_length=1)
    status: Literal["planned", "dry_run", "completed", "failed", "blocked"]
    primary_metric_name: str = "balanced_accuracy"
    primary_metric_value: float | None = None
    artifact_refs: list[ArtifactRef] = Field(default_factory=list)
    error: str | None = None
    notes: str | None = None

    @model_validator(mode="after")
    def _validate_summary_schema(self) -> TrialResultSummary:
        if "primary_metric_name" not in self.model_fields_set:
            raise ValueError("TrialResultSummary must explicitly declare primary_metric_name.")
        self.primary_metric_name = validate_metric_name(str(self.primary_metric_name).strip())
        if self.summary_result_schema_version not in SUPPORTED_SUMMARY_RESULT_SCHEMA_VERSIONS:
            supported = ", ".join(sorted(SUPPORTED_SUMMARY_RESULT_SCHEMA_VERSIONS))
            raise ValueError(
                "Unsupported trial result summary schema version "
                f"'{self.summary_result_schema_version}'. Supported versions: {supported}"
            )
        return self
