from __future__ import annotations

import json
from pathlib import Path


def _load_registry() -> dict:
    registry_path = Path("configs/decision_support_registry_revised_execution.json")
    return json.loads(registry_path.read_text(encoding="utf-8"))


def test_e09_registry_scope_is_narrowed_to_four_recog_templates() -> None:
    registry = _load_registry()
    experiments = registry.get("experiments", [])
    e09 = next(item for item in experiments if item.get("experiment_id") == "E09")
    templates = list(e09.get("variant_templates", []))

    assert len(templates) == 4

    subjects = {str(template["params"].get("subject")) for template in templates}
    assert subjects == {"sub-001", "sub-002"}

    feature_spaces = {str(template["params"].get("feature_space")) for template in templates}
    assert feature_spaces == {"whole_brain_masked", "roi_masked_predefined"}

    assert all(str(template["params"].get("filter_task")) == "recog" for template in templates)
    assert all(
        str(template["params"].get("filter_modality")) == "audiovisual" for template in templates
    )

    roi_templates = [
        template
        for template in templates
        if str(template["params"].get("feature_space")) == "roi_masked_predefined"
    ]
    assert len(roi_templates) == 2
    assert all(
        str(template["params"].get("roi_spec_path"))
        == "configs/feature_spaces/julich_sensory_union_voxels_v1.json"
        for template in roi_templates
    )
