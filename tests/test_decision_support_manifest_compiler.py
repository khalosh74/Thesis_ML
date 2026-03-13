from __future__ import annotations

import json
from pathlib import Path

import pytest

from Thesis_ML.orchestration.compiler import compile_registry_file, compile_registry_payload
from Thesis_ML.orchestration.contracts import SectionName


def _minimal_registry_payload() -> dict[str, object]:
    return {
        "schema_version": "test",
        "description": "unit-test registry",
        "experiments": [
            {
                "experiment_id": "E01",
                "title": "Target granularity experiment",
                "stage": "Stage 1 - Target lock",
                "decision_id": "D01",
                "manipulated_factor": "Target definition",
                "primary_metric": "balanced_accuracy",
                "variant_templates": [
                    {
                        "template_id": "coarse_affect_within_subject",
                        "supported": True,
                        "params": {
                            "target": "coarse_affect",
                            "model": "ridge",
                            "cv": "within_subject_loso_session",
                        },
                        "expand": {"subject": "subjects"},
                    },
                    {
                        "template_id": "binary_valence_like",
                        "supported": False,
                        "unsupported_reason": "not implemented",
                    },
                ],
            }
        ],
    }


def test_compile_registry_payload_success() -> None:
    manifest = compile_registry_payload(_minimal_registry_payload())
    assert manifest.schema_version == "test"
    assert len(manifest.experiments) == 1
    assert len(manifest.trial_specs) == 2
    expected_sections = [section.value for section in SectionName]
    assert manifest.supported_sections == expected_sections

    experiment = manifest.experiments[0]
    assert experiment.experiment_id == "E01"
    assert experiment.variant_templates[0].template_id == "coarse_affect_within_subject"
    assert experiment.variant_templates[0].sections == expected_sections
    assert experiment.variant_templates[0].start_section == "dataset_selection"
    assert experiment.variant_templates[0].end_section == "evaluation"
    assert experiment.variant_templates[0].reuse_policy == "auto"
    assert experiment.variant_templates[1].supported == False  # noqa: E712


def test_compile_registry_file_sets_source_path(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(
        json.dumps(
            {
                "schema_version": "test",
                "experiments": [
                    {
                        "experiment_id": "E01",
                        "title": "Target granularity experiment",
                        "stage": "Stage 1 - Target lock",
                        "variant_templates": [
                            {
                                "template_id": "t1",
                                "supported": True,
                                "params": {
                                    "target": "coarse_affect",
                                    "model": "ridge",
                                    "cv": "within_subject_loso_session",
                                },
                            }
                        ],
                    }
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    manifest = compile_registry_file(registry_path)
    assert manifest.source_registry_path == str(registry_path.resolve())
    assert manifest.experiments[0].experiment_id == "E01"


def test_compile_registry_payload_missing_experiments_raises() -> None:
    with pytest.raises(ValueError, match="experiments"):
        compile_registry_payload({"schema_version": "test"})


def test_compile_registry_payload_supported_trial_missing_required_params_raises() -> None:
    payload = _minimal_registry_payload()
    experiments = payload["experiments"]
    assert isinstance(experiments, list)
    experiment = experiments[0]
    assert isinstance(experiment, dict)
    templates = experiment["variant_templates"]
    assert isinstance(templates, list)
    template = templates[0]
    assert isinstance(template, dict)
    template["params"] = {"target": "coarse_affect", "model": "ridge"}

    with pytest.raises(ValueError, match="params keys: cv"):
        compile_registry_payload(payload)


def test_compile_registry_payload_invalid_reuse_policy_raises() -> None:
    payload = _minimal_registry_payload()
    experiments = payload["experiments"]
    assert isinstance(experiments, list)
    experiment = experiments[0]
    assert isinstance(experiment, dict)
    templates = experiment["variant_templates"]
    assert isinstance(templates, list)
    template = templates[0]
    assert isinstance(template, dict)
    template["reuse_policy"] = "invalid_policy"

    with pytest.raises(ValueError, match="reuse_policy"):
        compile_registry_payload(payload)
