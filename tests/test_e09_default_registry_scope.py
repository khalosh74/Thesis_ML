from __future__ import annotations

import json
from pathlib import Path


def _load_registry(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_e09_scope(registry: dict) -> None:
    experiments = list(registry.get("experiments", []))
    e09 = next(item for item in experiments if str(item.get("experiment_id")) == "E09")
    templates = list(e09.get("variant_templates", []))

    assert e09.get("executable_now") is True
    assert str(e09.get("execution_status")) == "executable"
    assert len(templates) == 4

    subjects = {str(template["params"].get("subject")) for template in templates}
    feature_spaces = {str(template["params"].get("feature_space")) for template in templates}
    assert subjects == {"sub-001", "sub-002"}
    assert feature_spaces == {"whole_brain_masked", "roi_masked_predefined"}

    assert all(str(template["params"].get("target")) == "coarse_affect" for template in templates)
    assert all(
        str(template["params"].get("cv")) == "within_subject_loso_session" for template in templates
    )
    assert all(str(template["params"].get("model")) == "ridge" for template in templates)
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


def test_e09_scope_is_updated_in_default_registry_and_packaged_asset() -> None:
    repo_registry = _load_registry(Path("configs/decision_support_registry.json"))
    packaged_registry = _load_registry(Path("src/Thesis_ML/assets/configs/decision_support_registry.json"))

    _assert_e09_scope(repo_registry)
    _assert_e09_scope(packaged_registry)
