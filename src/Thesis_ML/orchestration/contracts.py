from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SectionName(StrEnum):
    DATASET_SELECTION = "dataset_selection"
    FEATURE_CACHE_BUILD = "feature_cache_build"
    FEATURE_MATRIX_LOAD = "feature_matrix_load"
    SPATIAL_VALIDATION = "spatial_validation"
    MODEL_FIT = "model_fit"
    EVALUATION = "evaluation"
    INTERPRETABILITY = "interpretability"


def supported_sections() -> list[str]:
    return [section.value for section in SectionName]


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
                    "Supported trial template must define params keys: "
                    + ", ".join(missing)
                )
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
    description: str | None = None
    source_registry_path: str | None = None
    compiled_at_utc: str
    supported_sections: list[SectionName] = Field(default_factory=supported_sections)
    experiments: list[ExperimentSpec]
    trial_specs: list[TrialSpec]

    @model_validator(mode="after")
    def _validate_trial_coverage(self) -> CompiledStudyManifest:
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
        return self


class TrialResultSummary(_ContractModel):
    experiment_id: str = Field(min_length=1)
    trial_id: str = Field(min_length=1)
    status: Literal["planned", "dry_run", "completed", "failed", "blocked"]
    primary_metric_name: str = "balanced_accuracy"
    primary_metric_value: float | None = None
    artifact_refs: list[ArtifactRef] = Field(default_factory=list)
    error: str | None = None
    notes: str | None = None
