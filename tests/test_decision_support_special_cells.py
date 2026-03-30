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


def test_e12_materialization_chunks_permutations_deterministically() -> None:
    experiment = {"experiment_id": "E12"}
    cells, warnings = materialize_experiment_cells(
        experiment=experiment,
        variants=[_base_variant()],
        dataset_scope={},
        n_permutations=120,
    )
    assert warnings == []
    assert len(cells) == 3
    assert [int(cell["n_permutations_override"]) for cell in cells] == [50, 50, 20]
    assert [int(cell["design_metadata"]["chunk_index"]) for cell in cells] == [1, 2, 3]


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
