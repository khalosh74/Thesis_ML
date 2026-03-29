from __future__ import annotations

from Thesis_ML.orchestration import campaign_engine


def _experiment(experiment_id: str) -> dict[str, str]:
    return {
        "experiment_id": experiment_id,
        "stage": "synthetic",
        "title": experiment_id,
    }


def test_phase_batches_follow_native_order_and_grouping() -> None:
    selected = [
        _experiment("E01"),
        _experiment("E02"),
        _experiment("E03"),
        _experiment("E16"),
        _experiment("E17"),
        _experiment("E24"),
    ]
    batches = campaign_engine._build_phase_batches(  # type: ignore[attr-defined]
        selected_experiments=selected,
        phase_plan="auto",
    )
    phase_names = [str(item["phase_name"]) for item in batches]
    assert phase_names[:4] == [
        "Preflight",
        "Stage 1 target/scope lock",
        "Stage 2 split/transfer lock",
        "Stage 3 model lock",
    ]
    stage1 = next(batch for batch in batches if batch["phase_name"] == "Stage 1 target/scope lock")
    assert [[exp["experiment_id"] for exp in group] for group in stage1["groups"]] == [
        ["E01"],
        ["E02", "E03"],
    ]
    confirmatory = next(batch for batch in batches if batch["phase_name"] == "Confirmatory")
    assert [[exp["experiment_id"] for exp in group] for group in confirmatory["groups"]] == [
        ["E16", "E17"],
    ]
    repro = next(batch for batch in batches if batch["phase_name"] == "Reproducibility audit")
    assert [[exp["experiment_id"] for exp in group] for group in repro["groups"]] == [["E24"]]


def test_flat_phase_plan_collapses_to_single_batch() -> None:
    selected = [_experiment("E03"), _experiment("E01")]
    batches = campaign_engine._build_phase_batches(  # type: ignore[attr-defined]
        selected_experiments=selected,
        phase_plan="flat",
    )
    assert len(batches) == 1
    assert batches[0]["phase_name"] == "Flat selected sequence"
    assert [[exp["experiment_id"] for exp in group] for group in batches[0]["groups"]] == [
        ["E03"],
        ["E01"],
    ]
