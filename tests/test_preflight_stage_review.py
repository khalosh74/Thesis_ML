from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd


def _load_script_module(script_path: Path):
    spec = importlib.util.spec_from_file_location(script_path.stem, script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load script module: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_config(path: Path, *, index_csv: Path, task: str, train_subject: str, test_subject: str) -> None:
    payload = {
        "index_csv": str(index_csv),
        "data_root": "Data",
        "cache_dir": "Data/processed/feature_cache",
        "target": "coarse_affect",
        "cv": "frozen_cross_person_transfer",
        "model": "ridge",
        "train_subject": train_subject,
        "test_subject": test_subject,
        "filter_task": task,
        "seed": 42,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_metrics(report_dir: Path, balanced_accuracy: float) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "primary_metric_name": "balanced_accuracy",
        "primary_metric_value": balanced_accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": balanced_accuracy,
        "accuracy": balanced_accuracy,
        "n_folds": 1,
    }
    (report_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(
        [
            {
                "fold": 0,
                "balanced_accuracy": balanced_accuracy,
                "macro_f1": balanced_accuracy,
                "accuracy": balanced_accuracy,
            }
        ]
    ).to_csv(report_dir / "fold_metrics.csv", index=False)


def test_preflight_stage_review_writes_outputs_and_summaries(tmp_path: Path) -> None:
    campaign_root = tmp_path / "campaign"
    campaign_root.mkdir(parents=True, exist_ok=True)

    index_csv = tmp_path / "dataset_index.csv"
    index_df = pd.DataFrame(
        [
            # emo task
            {"sample_id": "e1", "subject": "sub-001", "session": "ses-01", "task": "emo", "modality": "audio", "emotion": "anger"},
            {"sample_id": "e2", "subject": "sub-001", "session": "ses-01", "task": "emo", "modality": "audio", "emotion": "anger"},
            {"sample_id": "e3", "subject": "sub-001", "session": "ses-02", "task": "emo", "modality": "audio", "emotion": "anger"},
            {"sample_id": "e4", "subject": "sub-001", "session": "ses-02", "task": "emo", "modality": "audio", "emotion": "happiness"},
            {"sample_id": "e5", "subject": "sub-002", "session": "ses-01", "task": "emo", "modality": "audio", "emotion": "happiness"},
            {"sample_id": "e6", "subject": "sub-002", "session": "ses-01", "task": "emo", "modality": "audio", "emotion": "happiness"},
            {"sample_id": "e7", "subject": "sub-002", "session": "ses-02", "task": "emo", "modality": "audio", "emotion": "happiness"},
            {"sample_id": "e8", "subject": "sub-002", "session": "ses-02", "task": "emo", "modality": "audio", "emotion": "happiness"},
            # recog task
            {"sample_id": "r1", "subject": "sub-001", "session": "ses-01", "task": "recog", "modality": "audio", "emotion": "anger"},
            {"sample_id": "r2", "subject": "sub-001", "session": "ses-01", "task": "recog", "modality": "audio", "emotion": "anger"},
            {"sample_id": "r3", "subject": "sub-001", "session": "ses-02", "task": "recog", "modality": "audio", "emotion": "anger"},
            {"sample_id": "r4", "subject": "sub-001", "session": "ses-02", "task": "recog", "modality": "audio", "emotion": "happiness"},
            {"sample_id": "r5", "subject": "sub-002", "session": "ses-01", "task": "recog", "modality": "audio", "emotion": "happiness"},
            {"sample_id": "r6", "subject": "sub-002", "session": "ses-01", "task": "recog", "modality": "audio", "emotion": "happiness"},
            {"sample_id": "r7", "subject": "sub-002", "session": "ses-02", "task": "recog", "modality": "audio", "emotion": "happiness"},
            {"sample_id": "r8", "subject": "sub-002", "session": "ses-02", "task": "recog", "modality": "audio", "emotion": "happiness"},
        ]
    )
    index_df.to_csv(index_csv, index=False)

    registry = {
        "schema_version": "test",
        "experiments": [{"experiment_id": "E05", "title": "Transfer direction"}],
    }
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")

    run_rows = []
    runs = [
        ("run1", "emo", "sub-001", "sub-002", 0.68),
        ("run2", "emo", "sub-002", "sub-001", 0.63),
        ("run3", "recog", "sub-001", "sub-002", 0.69),
        ("run4", "recog", "sub-002", "sub-001", 0.64),
    ]
    for run_id, task, train_subject, test_subject, score in runs:
        report_dir = campaign_root / "reports" / run_id
        config_path = campaign_root / "configs" / f"{run_id}.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        _write_config(
            config_path,
            index_csv=index_csv,
            task=task,
            train_subject=train_subject,
            test_subject=test_subject,
        )
        _write_metrics(report_dir, score)

        run_rows.append(
            {
                "Run_ID": run_id,
                "Experiment_ID": "E05",
                "Config_File_or_Path": str(config_path),
                "Artifact_Path": str(report_dir),
                "Result_Summary": "completed",
            }
        )

    pd.DataFrame(run_rows).to_csv(campaign_root / "run_log_export.csv", index=False)
    pd.DataFrame(
        [
            {
                "experiment_id": "E05",
                "completed_variants": 4,
                "failed_variants": 0,
                "blocked_variants": 0,
            }
        ]
    ).to_csv(campaign_root / "decision_support_summary.csv", index=False)

    script = _load_script_module(Path("scripts") / "review_preflight_stage.py")
    exit_code = script.main(
        [
            "--campaign-root",
            str(campaign_root),
            "--experiment-id",
            "E05",
            "--registry",
            str(registry_path),
        ]
    )
    assert exit_code == 0

    reviews_dir = campaign_root / "preflight_reviews"
    review_json = reviews_dir / "E05_review.json"
    review_csv = reviews_dir / "E05_review.csv"
    review_md = reviews_dir / "E05_review.md"
    lock_md = reviews_dir / "E05_lock_review.md"

    assert review_json.exists()
    assert review_csv.exists()
    assert review_md.exists()
    assert lock_md.exists()

    payload = json.loads(review_json.read_text(encoding="utf-8"))
    assert payload["expected_completed_variants"] == 4
    assert payload["completed_variants"] == 4
    assert payload["uncertainty_status"] == "single_split_no_fold_variance"
    assert "expected_completed_variants" in review_md.read_text(encoding="utf-8")

    # Dummy baseline should be reconstructed from TRAIN folds only.
    # For run1 (sub-001 -> sub-002), train majority is negative while test is all positive.
    run1 = next(row for row in payload["completed_runs"] if row["run_id"] == "run1")
    assert run1["dummy_baseline_mean_balanced_accuracy"] == 0.0

    # Fold metrics summary should be populated from fold_metrics.csv.
    assert run1["uncertainty_summary"]["mean"] == 0.68
    assert run1["uncertainty_summary"]["fold_count"] == 1

    csv_frame = pd.read_csv(review_csv)
    assert "summary" in set(csv_frame["record_type"].astype(str).tolist())
