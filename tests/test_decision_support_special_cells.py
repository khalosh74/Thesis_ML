from __future__ import annotations

from Thesis_ML.orchestration.variant_expansion import materialize_experiment_cells


def _base_variant() -> dict[str, object]:
    return {
        "template_id": "base",
        "variant_index": 1,
        "supported": True,
        "blocked_reason": None,
        "params": {"target": "coarse_affect", "model": "ridge", "cv": "within_subject_loso_session"},
        "factor_settings": {},
        "fixed_controls": {},
        "design_metadata": {},
    }


def _registry_experiments_for_e12_anchor() -> list[dict[str, object]]:
    return [
        {
            "experiment_id": "E12",
            "executable_now": True,
            "execution_status": "unknown",
            "variant_templates": [
                {
                    "template_id": "e12_template",
                    "supported": True,
                    "params": {
                        "target": "coarse_affect",
                        "model": "ridge",
                        "cv": "within_subject_loso_session",
                    },
                }
            ],
        },
        {
            "experiment_id": "E16",
            "title": "Final within-person confirmatory analysis",
            "stage": "Stage 5 - Confirmatory analysis",
            "executable_now": True,
            "execution_status": "unknown",
            "variant_templates": [
                {
                    "template_id": "e16_anchor",
                    "supported": True,
                    "params": {
                        "target": "coarse_affect",
                        "model": "ridge",
                        "cv": "within_subject_loso_session",
                        "subject": "sub-001",
                        "feature_space": "whole_brain_masked",
                    },
                }
            ],
        },
    ]


def _registry_experiments_for_e12_multi_anchor() -> list[dict[str, object]]:
    return [
        {
            "experiment_id": "E16",
            "title": "Final within-person confirmatory analysis",
            "stage": "Stage 5 - Confirmatory analysis",
            "executable_now": True,
            "execution_status": "unknown",
            "variant_templates": [
                {
                    "template_id": "e16_anchor",
                    "supported": True,
                    "params": {
                        "target": "coarse_affect",
                        "model": "ridge",
                        "cv": "within_subject_loso_session",
                        "subject": "sub-001",
                        "feature_space": "whole_brain_masked",
                    },
                }
            ],
        },
        {
            "experiment_id": "E18",
            "title": "Final cross-person confirmatory analysis",
            "stage": "Stage 5 - Confirmatory analysis",
            "executable_now": True,
            "execution_status": "unknown",
            "variant_templates": [
                {
                    "template_id": "e18_anchor",
                    "supported": True,
                    "params": {
                        "target": "coarse_affect",
                        "model": "ridge",
                        "cv": "frozen_cross_person_transfer",
                        "train_subject": "sub-001",
                        "test_subject": "sub-002",
                        "feature_space": "whole_brain_masked",
                    },
                }
            ],
        },
    ]


def test_e12_materialization_chunks_permutations_deterministically() -> None:
    experiment = {"experiment_id": "E12"}
    cells, warnings = materialize_experiment_cells(
        experiment=experiment,
        variants=[_base_variant()],
        dataset_scope={},
        n_permutations=120,
        registry_experiments=_registry_experiments_for_e12_anchor(),
    )
    assert warnings == []
    assert len(cells) == 3
    assert [int(cell["n_permutations_override"]) for cell in cells] == [50, 50, 20]
    assert [int(cell["design_metadata"]["chunk_index"]) for cell in cells] == [1, 2, 3]
    assert {str(cell["params"]["subject"]) for cell in cells} == {"sub-001"}
    assert {str(cell["params"]["feature_space"]) for cell in cells} == {"whole_brain_masked"}
    required_chunk_keys = {
        "permutation_group_id",
        "expected_chunk_count",
        "total_permutations_requested",
        "anchor_experiment_id",
        "anchor_template_id",
        "anchor_identity",
    }
    for cell in cells:
        metadata = dict(cell["design_metadata"])
        assert required_chunk_keys <= set(metadata)
        assert int(metadata["expected_chunk_count"]) == 3
        assert int(metadata["total_permutations_requested"]) == 120
        assert str(metadata["anchor_experiment_id"]) == "E16"
        factor_settings = dict(cell["factor_settings"])
        assert str(factor_settings["permutation_group_id"])
        assert int(factor_settings["expected_chunk_count"]) == 3
        assert int(factor_settings["total_permutations_requested"]) == 120
        assert str(factor_settings["anchor_experiment_id"]) == "E16"


def test_e12_materialization_creates_separate_groups_for_within_and_transfer_anchors() -> None:
    experiment = {"experiment_id": "E12"}
    cells, warnings = materialize_experiment_cells(
        experiment=experiment,
        variants=[_base_variant()],
        dataset_scope={},
        n_permutations=100,
        registry_experiments=_registry_experiments_for_e12_multi_anchor(),
    )
    assert warnings == []
    assert len(cells) == 4
    group_ids = {
        str(cell["design_metadata"]["permutation_group_id"])
        for cell in cells
        if isinstance(cell.get("design_metadata"), dict)
    }
    assert len(group_ids) == 2
    within_cells = [
        cell for cell in cells if str(cell["params"].get("cv")) == "within_subject_loso_session"
    ]
    transfer_cells = [
        cell for cell in cells if str(cell["params"].get("cv")) == "frozen_cross_person_transfer"
    ]
    assert len(within_cells) == 2
    assert len(transfer_cells) == 2
    assert {str(cell["params"].get("subject")) for cell in within_cells} == {"sub-001"}
    assert {
        (str(cell["params"].get("train_subject")), str(cell["params"].get("test_subject")))
        for cell in transfer_cells
    } == {("sub-001", "sub-002")}


def test_e23_materialization_expands_omitted_sessions() -> None:
    experiment = {"experiment_id": "E23"}
    dataset_scope = {
        "subjects": ["sub-001"],
        "tasks": ["emo", "recog"],
        "sessions_by_subject_task_modality": {
            "sub-001": {
                "emo": {"audiovisual": ["ses-01", "ses-02"]},
                "recog": {"audiovisual": ["ses-01"]},
            }
        },
    }
    variant = _base_variant()
    variant["params"] = {
        "target": "coarse_affect",
        "model": "ridge",
        "cv": "session_influence_jackknife",
        "filter_modality": "audiovisual",
    }
    cells, warnings = materialize_experiment_cells(
        experiment=experiment,
        variants=[variant],
        dataset_scope=dataset_scope,
        n_permutations=0,
    )
    assert warnings == []
    assert len(cells) == 3
    omitted = sorted(str(cell["design_metadata"]["omitted_session"]) for cell in cells)
    assert omitted == ["ses-01", "ses-01", "ses-02"]
    assert all(bool(cell["supported"]) is False for cell in cells)
    assert all(
        "not supported by thesisml-run-experiment" in str(cell.get("blocked_reason", ""))
        for cell in cells
    )


def test_e24_materialization_marks_cells_as_sequential_only() -> None:
    experiment = {"experiment_id": "E24"}
    cells, warnings = materialize_experiment_cells(
        experiment=experiment,
        variants=[_base_variant()],
        dataset_scope={},
        n_permutations=0,
    )
    assert warnings == []
    assert len(cells) == 1
    assert bool(cells[0]["sequential_only"]) is True
    assert bool(cells[0]["design_metadata"]["sequential_only"]) is True
