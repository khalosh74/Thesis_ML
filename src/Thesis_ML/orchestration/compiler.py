from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from Thesis_ML.config.schema_versions import COMPILED_MANIFEST_SCHEMA_VERSION
from Thesis_ML.orchestration.contracts import (
    CompiledStudyManifest,
    EffectSummary,
    ExperimentSpec,
    GeneratedDesignCell,
    SearchSpaceSpec,
    StudyDesignSpec,
    TrialSpec,
    supported_sections,
)


def _compile_trials_for_experiment(
    experiment_id: str,
    raw_templates: list[dict[str, Any]],
) -> list[TrialSpec]:
    compiled_trials: list[TrialSpec] = []
    for template in raw_templates:
        payload = dict(template)
        payload["experiment_id"] = experiment_id
        payload.setdefault("sections", supported_sections())
        payload.setdefault("artifacts", [])
        payload.setdefault("start_section", "dataset_selection")
        payload.setdefault("end_section", "evaluation")
        payload.setdefault("base_artifact_id", None)
        payload.setdefault("reuse_policy", "auto")
        try:
            compiled_trials.append(TrialSpec.model_validate(payload))
        except ValidationError as exc:
            template_id = str(payload.get("template_id", "<missing-template-id>"))
            raise ValueError(
                f"Invalid trial template '{template_id}' for experiment '{experiment_id}': {exc}"
            ) from exc
    return compiled_trials


def compile_registry_payload(
    payload: dict[str, Any],
    *,
    source_registry_path: Path | None = None,
) -> CompiledStudyManifest:
    experiments_payload = payload.get("experiments")
    if not isinstance(experiments_payload, list):
        raise ValueError("Invalid registry payload: expected an 'experiments' list.")

    compiled_experiments: list[ExperimentSpec] = []
    compiled_trials: list[TrialSpec] = []
    compiled_search_spaces: list[SearchSpaceSpec] = []
    compiled_study_designs: list[StudyDesignSpec] = []
    compiled_generated_design_matrix: list[GeneratedDesignCell] = []
    compiled_effect_summaries: list[EffectSummary] = []

    raw_search_spaces = payload.get("search_spaces", [])
    if raw_search_spaces is None:
        raw_search_spaces = []
    if not isinstance(raw_search_spaces, list):
        raise ValueError("Invalid registry payload: expected 'search_spaces' to be a list.")
    for raw_space in raw_search_spaces:
        if not isinstance(raw_space, dict):
            raise ValueError("Invalid search space payload: each search space must be an object.")
        try:
            compiled_search_spaces.append(SearchSpaceSpec.model_validate(dict(raw_space)))
        except ValidationError as exc:
            space_id = str(raw_space.get("search_space_id", "<missing-search-space-id>"))
            raise ValueError(f"Invalid search space '{space_id}': {exc}") from exc

    raw_study_designs = payload.get("study_designs", [])
    if raw_study_designs is None:
        raw_study_designs = []
    if not isinstance(raw_study_designs, list):
        raise ValueError("Invalid registry payload: expected 'study_designs' to be a list.")
    for raw_study in raw_study_designs:
        if not isinstance(raw_study, dict):
            raise ValueError("Invalid study design payload: each study must be an object.")
        try:
            compiled_study_designs.append(StudyDesignSpec.model_validate(dict(raw_study)))
        except ValidationError as exc:
            study_id = str(raw_study.get("study_id", "<missing-study-id>"))
            raise ValueError(f"Invalid study design '{study_id}': {exc}") from exc

    raw_generated_matrix = payload.get("generated_design_matrix", [])
    if raw_generated_matrix is None:
        raw_generated_matrix = []
    if not isinstance(raw_generated_matrix, list):
        raise ValueError("Invalid registry payload: expected 'generated_design_matrix' to be a list.")
    for raw_cell in raw_generated_matrix:
        if not isinstance(raw_cell, dict):
            raise ValueError("Invalid generated design cell payload: each cell must be an object.")
        try:
            compiled_generated_design_matrix.append(GeneratedDesignCell.model_validate(dict(raw_cell)))
        except ValidationError as exc:
            trial_id = str(raw_cell.get("trial_id", "<missing-trial-id>"))
            raise ValueError(f"Invalid generated design cell '{trial_id}': {exc}") from exc

    raw_effect_summaries = payload.get("effect_summaries", [])
    if raw_effect_summaries is None:
        raw_effect_summaries = []
    if not isinstance(raw_effect_summaries, list):
        raise ValueError("Invalid registry payload: expected 'effect_summaries' to be a list.")
    for raw_summary in raw_effect_summaries:
        if not isinstance(raw_summary, dict):
            raise ValueError("Invalid effect summary payload: each summary must be an object.")
        try:
            compiled_effect_summaries.append(EffectSummary.model_validate(dict(raw_summary)))
        except ValidationError as exc:
            summary_type = str(raw_summary.get("summary_type", "<missing-summary-type>"))
            raise ValueError(f"Invalid effect summary '{summary_type}': {exc}") from exc

    for raw_experiment in experiments_payload:
        if not isinstance(raw_experiment, dict):
            raise ValueError("Invalid experiment payload: each experiment entry must be an object.")
        experiment_payload = dict(raw_experiment)
        experiment_id = str(experiment_payload.get("experiment_id", "")).strip()
        raw_templates = experiment_payload.get("variant_templates", [])
        if raw_templates is None:
            raw_templates = []
        if not isinstance(raw_templates, list):
            raise ValueError(
                f"Invalid variant_templates for experiment '{experiment_id or '<missing-id>'}': "
                "expected a list."
            )

        trial_specs = _compile_trials_for_experiment(
            experiment_id=experiment_id,
            raw_templates=raw_templates,
        )
        experiment_payload["variant_templates"] = trial_specs
        experiment_payload.setdefault("section_plan", supported_sections())

        try:
            experiment_spec = ExperimentSpec.model_validate(experiment_payload)
        except ValidationError as exc:
            raise ValueError(
                f"Invalid experiment spec '{experiment_id or '<missing-id>'}': {exc}"
            ) from exc

        compiled_experiments.append(experiment_spec)
        compiled_trials.extend(trial_specs)

    manifest_payload = {
        **payload,
        "schema_version": str(payload.get("schema_version", "unspecified")),
        "compiled_manifest_schema_version": COMPILED_MANIFEST_SCHEMA_VERSION,
        "source_registry_path": (
            str(source_registry_path.resolve()) if source_registry_path else None
        ),
        "compiled_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "supported_sections": supported_sections(),
        "experiments": compiled_experiments,
        "trial_specs": compiled_trials,
        "search_spaces": compiled_search_spaces,
        "study_designs": compiled_study_designs,
        "generated_design_matrix": compiled_generated_design_matrix,
        "effect_summaries": compiled_effect_summaries,
    }

    try:
        return CompiledStudyManifest.model_validate(manifest_payload)
    except ValidationError as exc:
        raise ValueError(f"Invalid compiled study manifest: {exc}") from exc


def compile_registry_file(path: Path) -> CompiledStudyManifest:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON in registry '{path}': {exc.msg} (line {exc.lineno}, col {exc.colno})"
        ) from exc

    if not isinstance(payload, dict):
        raise ValueError(f"Invalid registry root in '{path}': expected JSON object.")

    return compile_registry_payload(payload, source_registry_path=path)
