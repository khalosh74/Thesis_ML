from __future__ import annotations

import json
from pathlib import Path


def _registry_payload() -> dict[str, object]:
    path = Path("configs/decision_support_registry_revised_execution.json")
    return json.loads(path.read_text(encoding="utf-8"))


def _experiment_templates(experiment_id: str) -> list[dict[str, object]]:
    payload = _registry_payload()
    experiments = list(payload.get("experiments") or [])
    for row in experiments:
        if not isinstance(row, dict):
            continue
        if str(row.get("experiment_id")) != str(experiment_id):
            continue
        templates = list(row.get("variant_templates") or [])
        return [template for template in templates if isinstance(template, dict)]
    raise AssertionError(f"experiment_id '{experiment_id}' was not found in runtime registry")


def test_e17_templates_use_no_preprocessing() -> None:
    templates = _experiment_templates("E17")
    assert templates
    template_ids = {str(template.get("template_id")) for template in templates}
    assert {"wb_row_015", "wb_row_015_reverse"} <= template_ids
    for template in templates:
        params = dict(template.get("params") or {})
        assert str(params.get("preprocessing_strategy")) == "none"


def test_e15_template_declares_explicit_locked_core() -> None:
    templates = _experiment_templates("E15")
    e15_template = next(
        (template for template in templates if str(template.get("template_id")) == "wb_row_013"),
        None,
    )
    if e15_template is None:
        raise AssertionError("E15 template 'wb_row_013' is missing from runtime registry")
    params = dict(e15_template.get("params") or {})
    assert str(params.get("feature_space")) == "whole_brain_masked"
    assert str(params.get("preprocessing_strategy")) == "none"
    assert str(params.get("dimensionality_strategy")) == "none"
    assert str(params.get("methodology_policy_name")) == "fixed_baselines_only"
    assert str(params.get("class_weight_policy")) == "none"
