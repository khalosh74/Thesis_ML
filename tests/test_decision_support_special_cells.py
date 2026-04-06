from __future__ import annotations

from Thesis_ML.orchestration.variant_expansion import materialize_experiment_cells


def _base_variant() -> dict[str, object]:
    return {
        "template_id": "base",
        "variant_index": 1,
        "supported": True,
        "blocked_reason": None,
        "params": {
            "target": "coarse_affect",
            "model": "ridge",
            "cv": "within_subject_loso_session",
        },
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
                },
                {
                    "template_id": "e16_anchor_sub002",
                    "supported": True,
                    "params": {
                        "target": "coarse_affect",
                        "model": "ridge",
                        "cv": "within_subject_loso_session",
                        "subject": "sub-002",
                        "feature_space": "whole_brain_masked",
                    },
                },
            ],
        },
        {
            "experiment_id": "E17",
            "title": "Final cross-person confirmatory analysis",
            "stage": "Stage 5 - Confirmatory analysis",
            "executable_now": True,
            "execution_status": "unknown",
            "variant_templates": [
                {
                    "template_id": "e17_anchor",
                    "supported": True,
                    "params": {
                        "target": "coarse_affect",
                        "model": "ridge",
                        "cv": "frozen_cross_person_transfer",
                        "train_subject": "sub-001",
                        "test_subject": "sub-002",
                        "feature_space": "whole_brain_masked",
                    },
                },
                {
                    "template_id": "e17_anchor_reverse",
                    "supported": True,
                    "params": {
                        "target": "coarse_affect",
                        "model": "ridge",
                        "cv": "frozen_cross_person_transfer",
                        "train_subject": "sub-002",
                        "test_subject": "sub-001",
                        "feature_space": "whole_brain_masked",
                    },
                },
            ],
        },
    ]


def _registry_experiments_for_e13_multi_anchor() -> list[dict[str, object]]:
    return [
        {
            "experiment_id": "E13",
            "executable_now": True,
            "execution_status": "unknown",
            "variant_templates": [
                {
                    "template_id": "e13_template",
                    "supported": True,
                    "params": {
                        "target": "coarse_affect",
                        "model": "dummy_or_majority",
                        "cv": "within_subject_loso_session",
                        "subject": "None",
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
                },
                {
                    "template_id": "e16_anchor_sub002",
                    "supported": True,
                    "params": {
                        "target": "coarse_affect",
                        "model": "ridge",
                        "cv": "within_subject_loso_session",
                        "subject": "sub-002",
                        "feature_space": "whole_brain_masked",
                    },
                },
            ],
        },
        {
            "experiment_id": "E17",
            "title": "Final cross-person confirmatory analysis",
            "stage": "Stage 5 - Confirmatory analysis",
            "executable_now": True,
            "execution_status": "unknown",
            "variant_templates": [
                {
                    "template_id": "e17_anchor",
                    "supported": True,
                    "params": {
                        "target": "coarse_affect",
                        "model": "ridge",
                        "cv": "frozen_cross_person_transfer",
                        "train_subject": "sub-001",
                        "test_subject": "sub-002",
                        "feature_space": "whole_brain_masked",
                    },
                },
                {
                    "template_id": "e17_anchor_reverse",
                    "supported": True,
                    "params": {
                        "target": "coarse_affect",
                        "model": "ridge",
                        "cv": "frozen_cross_person_transfer",
                        "train_subject": "sub-002",
                        "test_subject": "sub-001",
                        "feature_space": "whole_brain_masked",
                    },
                },
            ],
        },
    ]


def _registry_experiments_for_e13_single_anchor() -> list[dict[str, object]]:
    return [
        {
            "experiment_id": "E13",
            "executable_now": True,
            "execution_status": "unknown",
            "variant_templates": [
                {
                    "template_id": "e13_template",
                    "supported": True,
                    "params": {
                        "target": "coarse_affect",
                        "model": "dummy",
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
                    },
                }
            ],
        },
    ]


def _registry_experiments_for_e14_multi_anchor() -> list[dict[str, object]]:
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
                },
                {
                    "template_id": "e16_anchor_sub002",
                    "supported": True,
                    "params": {
                        "target": "coarse_affect",
                        "model": "ridge",
                        "cv": "within_subject_loso_session",
                        "subject": "sub-002",
                        "feature_space": "whole_brain_masked",
                    },
                },
            ],
        },
        {
            "experiment_id": "E17",
            "title": "Final cross-person confirmatory analysis",
            "stage": "Stage 5 - Confirmatory analysis",
            "executable_now": True,
            "execution_status": "unknown",
            "variant_templates": [
                {
                    "template_id": "e17_anchor",
                    "supported": True,
                    "params": {
                        "target": "coarse_affect",
                        "model": "ridge",
                        "cv": "frozen_cross_person_transfer",
                        "train_subject": "sub-001",
                        "test_subject": "sub-002",
                    },
                }
            ],
        },
    ]


def _registry_experiments_for_e14_non_linear_anchor_only() -> list[dict[str, object]]:
    return [
        {
            "experiment_id": "E16",
            "title": "Final within-person confirmatory analysis",
            "stage": "Stage 5 - Confirmatory analysis",
            "executable_now": True,
            "execution_status": "unknown",
            "variant_templates": [
                {
                    "template_id": "e16_anchor_tree",
                    "supported": True,
                    "params": {
                        "target": "coarse_affect",
                        "model": "xgboost",
                        "cv": "within_subject_loso_session",
                        "subject": "sub-001",
                    },
                }
            ],
        },
        {
            "experiment_id": "E17",
            "title": "Final cross-person confirmatory analysis",
            "stage": "Stage 5 - Confirmatory analysis",
            "executable_now": True,
            "execution_status": "unknown",
            "variant_templates": [
                {
                    "template_id": "e17_anchor",
                    "supported": True,
                    "params": {
                        "target": "coarse_affect",
                        "model": "ridge",
                        "cv": "frozen_cross_person_transfer",
                        "train_subject": "sub-001",
                        "test_subject": "sub-002",
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
    seeds = [int(cell["seed"]) for cell in cells]
    assert len(set(seeds)) == 3
    assert seeds == sorted(seeds)
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
    assert len(cells) == 8
    group_ids = {
        str(cell["design_metadata"]["permutation_group_id"])
        for cell in cells
        if isinstance(cell.get("design_metadata"), dict)
    }
    assert len(group_ids) == 4
    within_cells = [
        cell for cell in cells if str(cell["params"].get("cv")) == "within_subject_loso_session"
    ]
    transfer_cells = [
        cell for cell in cells if str(cell["params"].get("cv")) == "frozen_cross_person_transfer"
    ]
    assert len(within_cells) == 4
    assert len(transfer_cells) == 4
    assert {str(cell["params"].get("subject")) for cell in within_cells} == {
        "sub-001",
        "sub-002",
    }
    assert {
        (str(cell["params"].get("train_subject")), str(cell["params"].get("test_subject")))
        for cell in transfer_cells
    } == {("sub-001", "sub-002"), ("sub-002", "sub-001")}


def test_e12_materialization_honors_design_metadata_chunk_size_override() -> None:
    experiment = {"experiment_id": "E12"}
    variant = _base_variant()
    variant["design_metadata"] = {"permutation_chunk_size": 200}
    cells, warnings = materialize_experiment_cells(
        experiment=experiment,
        variants=[variant],
        dataset_scope={},
        n_permutations=450,
        registry_experiments=_registry_experiments_for_e12_anchor(),
    )
    assert warnings == []
    assert len(cells) == 3
    assert [int(cell["n_permutations_override"]) for cell in cells] == [200, 200, 50]
    assert [int(cell["design_metadata"]["chunk_index"]) for cell in cells] == [1, 2, 3]
    for cell in cells:
        metadata = dict(cell["design_metadata"])
        assert int(metadata["expected_chunk_count"]) == 3
        assert int(metadata["total_permutations_requested"]) == 450


def test_e13_materialization_matches_confirmatory_anchors_and_enforces_dummy_model() -> None:
    experiment = {"experiment_id": "E13"}
    cells, warnings = materialize_experiment_cells(
        experiment=experiment,
        variants=[_base_variant()],
        dataset_scope={},
        n_permutations=0,
        registry_experiments=_registry_experiments_for_e13_multi_anchor(),
    )
    assert warnings == []
    assert len(cells) == 4
    assert {str(cell["params"].get("model")) for cell in cells} == {"dummy"}
    assert all(cell["params"].get("subject") != "None" for cell in cells)

    within_cells = [
        cell for cell in cells if str(cell["params"].get("cv")) == "within_subject_loso_session"
    ]
    transfer_cells = [
        cell for cell in cells if str(cell["params"].get("cv")) == "frozen_cross_person_transfer"
    ]
    assert len(within_cells) == 2
    assert len(transfer_cells) == 2
    assert {str(cell["params"].get("subject")) for cell in within_cells} == {
        "sub-001",
        "sub-002",
    }
    assert {
        (str(cell["params"].get("train_subject")), str(cell["params"].get("test_subject")))
        for cell in transfer_cells
    } == {("sub-001", "sub-002"), ("sub-002", "sub-001")}

    metadata_kinds = {
        str(cell.get("design_metadata", {}).get("special_cell_kind"))
        for cell in cells
        if isinstance(cell.get("design_metadata"), dict)
    }
    assert metadata_kinds == {"confirmatory_dummy_baseline"}
    group_ids = {
        str(cell.get("design_metadata", {}).get("baseline_group_id"))
        for cell in cells
        if isinstance(cell.get("design_metadata"), dict)
    }
    assert len(group_ids) == 4


def test_e13_materialization_adapts_when_confirmatory_anchor_set_changes() -> None:
    experiment = {"experiment_id": "E13"}
    cells, warnings = materialize_experiment_cells(
        experiment=experiment,
        variants=[_base_variant()],
        dataset_scope={},
        n_permutations=0,
        registry_experiments=_registry_experiments_for_e13_single_anchor(),
    )
    assert warnings == []
    assert len(cells) == 1
    assert str(cells[0]["params"].get("model")) == "dummy"
    assert str(cells[0]["params"].get("cv")) == "within_subject_loso_session"
    assert str(cells[0]["params"].get("subject")) == "sub-001"


def test_e14_materialization_matches_e16_within_person_anchors_only() -> None:
    experiment = {"experiment_id": "E14"}
    cells, warnings = materialize_experiment_cells(
        experiment=experiment,
        variants=[_base_variant()],
        dataset_scope={},
        n_permutations=0,
        registry_experiments=_registry_experiments_for_e14_multi_anchor(),
    )
    assert warnings == []
    assert len(cells) == 2
    assert all(bool(cell.get("supported")) is False for cell in cells)
    assert {
        str(cell.get("design_metadata", {}).get("special_cell_kind"))
        for cell in cells
        if isinstance(cell.get("design_metadata"), dict)
    } == {"interpretability_stability"}
    assert {str(cell["params"].get("cv")) for cell in cells} == {"within_subject_loso_session"}
    assert {str(cell["params"].get("subject")) for cell in cells} == {"sub-001", "sub-002"}
    assert {
        str(cell.get("design_metadata", {}).get("anchor_experiment_id"))
        for cell in cells
        if isinstance(cell.get("design_metadata"), dict)
    } == {"E16"}


def test_e14_materialization_requires_within_subject_linear_anchor_conditions() -> None:
    experiment = {"experiment_id": "E14"}
    cells, warnings = materialize_experiment_cells(
        experiment=experiment,
        variants=[_base_variant()],
        dataset_scope={},
        n_permutations=0,
        registry_experiments=_registry_experiments_for_e14_non_linear_anchor_only(),
    )
    assert cells == []
    assert warnings
    assert "no eligible E16 within-subject linear-interpretability anchors" in str(warnings[0])


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


def test_e15_materialization_emits_restricted_and_full_control_cells() -> None:
    experiment = {"experiment_id": "E15"}
    variant = _base_variant()
    variant["params"] = {
        "target": "coarse_affect",
        "model": "ridge",
        "cv": "within_subject_loso_session",
        "subject": "sub-001",
        "filter_task": "emo",
        "filter_modality": "audiovisual",
        "feature_space": "whole_brain_masked",
        "preprocessing_strategy": "none",
        "dimensionality_strategy": "none",
        "methodology_policy_name": "fixed_baselines_only",
        "class_weight_policy": "none",
    }

    cells, warnings = materialize_experiment_cells(
        experiment=experiment,
        variants=[variant],
        dataset_scope={},
        n_permutations=0,
    )

    assert warnings == []
    assert len(cells) == 2
    subset_arms = {
        str(cell.get("design_metadata", {}).get("subset_arm"))
        for cell in cells
        if isinstance(cell.get("design_metadata"), dict)
    }
    assert subset_arms == {"task_restricted", "full_control"}
    restricted = [
        cell
        for cell in cells
        if str(cell.get("design_metadata", {}).get("subset_arm")) == "task_restricted"
    ][0]
    full_control = [
        cell
        for cell in cells
        if str(cell.get("design_metadata", {}).get("subset_arm")) == "full_control"
    ][0]
    assert str(restricted["params"].get("filter_task")) == "emo"
    assert full_control["params"].get("filter_task") is None


def test_e15_materialization_blocks_missing_subject_for_within_subject_cv() -> None:
    experiment = {"experiment_id": "E15"}
    variant = _base_variant()
    variant["params"] = {
        "target": "coarse_affect",
        "model": "ridge",
        "cv": "within_subject_loso_session",
        "filter_task": "emo",
    }

    cells, warnings = materialize_experiment_cells(
        experiment=experiment,
        variants=[variant],
        dataset_scope={},
        n_permutations=0,
    )

    assert warnings == []
    assert len(cells) == 1
    assert bool(cells[0]["supported"]) is False
    assert "requires subject" in str(cells[0].get("blocked_reason", ""))


def test_e15_materialization_blocks_missing_locked_core_parameters() -> None:
    experiment = {"experiment_id": "E15"}
    variant = _base_variant()
    variant["params"] = {
        "target": "coarse_affect",
        "model": "ridge",
        "cv": "within_subject_loso_session",
        "subject": "sub-001",
        "filter_task": "emo",
        "filter_modality": "audiovisual",
    }

    cells, warnings = materialize_experiment_cells(
        experiment=experiment,
        variants=[variant],
        dataset_scope={},
        n_permutations=0,
    )

    assert warnings == []
    assert len(cells) == 1
    assert bool(cells[0]["supported"]) is False
    assert "requires explicit locked-core parameters" in str(cells[0].get("blocked_reason", ""))
