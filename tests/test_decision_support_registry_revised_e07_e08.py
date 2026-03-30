from __future__ import annotations

import json
from pathlib import Path


def _load_revised_registry() -> dict:
    registry_path = Path("configs/decision_support_registry_revised_execution.json")
    return json.loads(registry_path.read_text(encoding="utf-8"))


def _get_experiment(payload: dict, experiment_id: str) -> dict:
    for experiment in payload.get("experiments", []):
        if experiment.get("experiment_id") == experiment_id:
            return experiment
    raise AssertionError(f"Missing experiment '{experiment_id}' in revised decision-support registry.")


def test_revised_registry_e07_is_explicit_narrowed_weighting() -> None:
    payload = _load_revised_registry()
    e07 = _get_experiment(payload, "E07")
    templates = list(e07.get("variant_templates", []))

    assert len(templates) == 8
    assert all(template.get("search_space_id") is None for template in templates)
    assert all(str(template.get("search_space_id")) != "SS05" for template in templates)

    class_weight_values = {
        str(template.get("params", {}).get("class_weight_policy")) for template in templates
    }
    assert class_weight_values == {"none", "balanced"}
    assert {
        str(template.get("params", {}).get("methodology_policy_name")) for template in templates
    } == {"fixed_baselines_only"}
    assert {
        str(template.get("params", {}).get("model")) for template in templates
    } == {"ridge"}
    assert all(str(template.get("params", {}).get("model")) != "VARIES" for template in templates)


def test_revised_registry_e08_is_explicit_narrowed_tuning() -> None:
    payload = _load_revised_registry()
    e08 = _get_experiment(payload, "E08")
    templates = list(e08.get("variant_templates", []))

    assert len(templates) == 8
    assert all(template.get("search_space_id") is None for template in templates)
    assert all(str(template.get("search_space_id")) != "SS05" for template in templates)

    methodology_values = {
        str(template.get("params", {}).get("methodology_policy_name")) for template in templates
    }
    assert methodology_values == {"fixed_baselines_only", "grouped_nested_tuning"}
    assert {str(template.get("params", {}).get("class_weight_policy")) for template in templates} == {
        "none"
    }
    assert {str(template.get("params", {}).get("model")) for template in templates} == {"ridge"}
    assert all(str(template.get("params", {}).get("model")) != "VARIES" for template in templates)

    grouped_nested_templates = [
        template
        for template in templates
        if str(template.get("params", {}).get("methodology_policy_name")) == "grouped_nested_tuning"
    ]
    assert grouped_nested_templates
    for template in grouped_nested_templates:
        params = template.get("params", {})
        assert params.get("tuning_search_space_id") == "official-linear-grouped-nested-v2"
        assert params.get("tuning_search_space_version") == "2.0.0"
        assert params.get("tuning_inner_cv_scheme") == "grouped_leave_one_group_out"
        assert params.get("tuning_inner_group_field") == "session"
