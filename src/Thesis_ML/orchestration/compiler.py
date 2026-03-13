from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from Thesis_ML.orchestration.contracts import (
    CompiledStudyManifest,
    ExperimentSpec,
    SearchSpaceSpec,
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
        "source_registry_path": (
            str(source_registry_path.resolve()) if source_registry_path else None
        ),
        "compiled_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "supported_sections": supported_sections(),
        "experiments": compiled_experiments,
        "trial_specs": compiled_trials,
        "search_spaces": compiled_search_spaces,
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
