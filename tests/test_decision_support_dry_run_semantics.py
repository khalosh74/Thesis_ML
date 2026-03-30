from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from Thesis_ML.orchestration import campaign_engine


def _write_index(path: Path) -> None:
    df = pd.DataFrame(
        [
            {
                "sample_id": "s1",
                "subject": "sub-001",
                "session": "ses-01",
                "task": "emo",
                "modality": "audiovisual",
            },
            {
                "sample_id": "s2",
                "subject": "sub-001",
                "session": "ses-02",
                "task": "emo",
                "modality": "audiovisual",
            },
            {
                "sample_id": "s3",
                "subject": "sub-001",
                "session": "ses-03",
                "task": "recog",
                "modality": "audiovisual",
            },
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _write_registry(path: Path, experiments: list[dict[str, object]]) -> None:
    payload = {"schema_version": "test", "experiments": experiments}
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _stub_run_experiment(**kwargs: object) -> dict[str, object]:
    run_id = str(kwargs.get("run_id"))
    return {
        "run_id": run_id,
        "report_dir": f"/tmp/{run_id}",
        "config_path": f"/tmp/{run_id}/config.json",
        "metrics_path": f"/tmp/{run_id}/metrics.json",
        "fold_metrics_path": f"/tmp/{run_id}/fold_metrics.csv",
        "fold_splits_path": f"/tmp/{run_id}/fold_splits.csv",
        "predictions_path": f"/tmp/{run_id}/predictions.csv",
        "spatial_compatibility_report_path": f"/tmp/{run_id}/spatial.json",
        "metrics": {
            "balanced_accuracy": 0.5,
            "macro_f1": 0.5,
            "accuracy": 0.5,
            "n_folds": 2,
        },
    }


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def test_selected_experiment_without_materialized_cells_is_explicitly_skipped(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.json"
    index_csv = tmp_path / "index.csv"
    _write_index(index_csv)
    _write_registry(
        registry_path,
        experiments=[
            {
                "experiment_id": "E12",
                "title": "Permutation robustness",
                "stage": "Blocking robustness",
                "decision_id": "D12",
                "manipulated_factor": "permutation_chunk",
                "primary_metric": "balanced_accuracy",
                "variant_templates": [
                    {
                        "template_id": "e12_base",
                        "supported": True,
                        "params": {
                            "target": "coarse_affect",
                            "model": "ridge",
                            "cv": "within_subject_loso_session",
                            "subject": "sub-001",
                        },
                        "expand": {},
                    }
                ],
            }
        ],
    )

    result = campaign_engine.run_decision_support_campaign(
        registry_path=registry_path,
        index_csv=index_csv,
        data_root=tmp_path / "Data",
        cache_dir=tmp_path / "cache",
        output_root=tmp_path / "outputs",
        experiment_id=None,
        stage=None,
        run_all=True,
        seed=42,
        n_permutations=0,
        dry_run=True,
        run_experiment_fn=_stub_run_experiment,
    )

    campaign_root = Path(result["campaign_root"])
    events = _read_jsonl(campaign_root / "execution_events.jsonl")
    skip_events = [
        row
        for row in events
        if str(row.get("event_name")) == "experiment_skipped"
        and str(row.get("experiment_id")) == "E12"
    ]
    assert skip_events
    assert "no materialized cells" in str(skip_events[0].get("message", ""))

    skip_summary = json.loads((campaign_root / "phase_skip_summary.json").read_text(encoding="utf-8"))
    skipped_rows = [
        row
        for row in skip_summary.get("skipped_experiments", [])
        if str(row.get("experiment_id")) == "E12"
    ]
    assert skipped_rows
    assert any("--n-permutations > 0" in str(row.get("reason", "")) for row in skipped_rows)

    summary_df = pd.read_csv(campaign_root / "decision_support_summary.csv")
    e12_row = summary_df[summary_df["experiment_id"] == "E12"].iloc[0]
    assert str(e12_row["status"]) == "skipped"


def test_dry_run_phase_event_sequence_is_consistent_for_later_phase_runs(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.json"
    index_csv = tmp_path / "index.csv"
    _write_index(index_csv)
    _write_registry(
        registry_path,
        experiments=[
            {
                "experiment_id": "E24",
                "title": "Reproducibility audit",
                "stage": "Reproducibility audit",
                "decision_id": "D24",
                "manipulated_factor": "repro_rerun",
                "primary_metric": "balanced_accuracy",
                "variant_templates": [
                    {
                        "template_id": "e24_base",
                        "supported": True,
                        "params": {
                            "target": "coarse_affect",
                            "model": "ridge",
                            "cv": "within_subject_loso_session",
                            "subject": "sub-001",
                        },
                        "expand": {},
                    }
                ],
            }
        ],
    )

    result = campaign_engine.run_decision_support_campaign(
        registry_path=registry_path,
        index_csv=index_csv,
        data_root=tmp_path / "Data",
        cache_dir=tmp_path / "cache",
        output_root=tmp_path / "outputs",
        experiment_id=None,
        stage=None,
        run_all=True,
        seed=42,
        n_permutations=0,
        dry_run=True,
        run_experiment_fn=_stub_run_experiment,
    )

    events = _read_jsonl(Path(result["campaign_root"]) / "execution_events.jsonl")
    run_event_names = {
        "run_planned",
        "run_dispatched",
        "run_started",
        "run_finished",
        "run_failed",
        "run_blocked",
        "run_dry_run",
    }
    started_order: dict[str, int] = {}
    finished_order: dict[str, int] = {}
    current_phase: str | None = None

    for idx, event in enumerate(events):
        event_name = str(event.get("event_name"))
        phase_name = str(event.get("phase_name")) if event.get("phase_name") is not None else None
        if event_name == "phase_started" and phase_name is not None:
            started_order[phase_name] = idx
            current_phase = phase_name
        elif event_name in run_event_names:
            assert phase_name is not None
            assert phase_name in started_order
            assert current_phase == phase_name
        elif event_name == "phase_finished" and phase_name is not None:
            assert phase_name in started_order
            assert current_phase == phase_name
            finished_order[phase_name] = idx
            current_phase = None

    assert started_order
    assert finished_order
    assert set(started_order) == set(finished_order)
    assert all(finished_order[name] > started_order[name] for name in started_order)


def test_e23_blocked_cells_remain_counted_in_dry_run(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.json"
    index_csv = tmp_path / "index.csv"
    _write_index(index_csv)
    _write_registry(
        registry_path,
        experiments=[
            {
                "experiment_id": "E23",
                "title": "Context robustness omitted-session",
                "stage": "Context robustness",
                "decision_id": "D23",
                "manipulated_factor": "omitted_session",
                "primary_metric": "balanced_accuracy",
                "variant_templates": [
                    {
                        "template_id": "e23_base",
                        "supported": True,
                        "params": {
                            "target": "coarse_affect",
                            "model": "ridge",
                            "cv": "session_influence_jackknife",
                            "filter_modality": "audiovisual",
                        },
                        "expand": {},
                    }
                ],
            }
        ],
    )

    result = campaign_engine.run_decision_support_campaign(
        registry_path=registry_path,
        index_csv=index_csv,
        data_root=tmp_path / "Data",
        cache_dir=tmp_path / "cache",
        output_root=tmp_path / "outputs",
        experiment_id=None,
        stage=None,
        run_all=True,
        seed=42,
        n_permutations=0,
        dry_run=True,
        run_experiment_fn=_stub_run_experiment,
    )

    campaign_root = Path(result["campaign_root"])
    events = _read_jsonl(campaign_root / "execution_events.jsonl")
    blocked_event_count = sum(1 for row in events if str(row.get("event_name")) == "run_blocked")

    live_payload = json.loads((campaign_root / "campaign_live_status.json").read_text(encoding="utf-8"))
    assert blocked_event_count > 0
    assert int(live_payload["counts"]["runs_blocked"]) == blocked_event_count
