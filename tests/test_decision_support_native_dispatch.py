from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from Thesis_ML.experiments.parallel_execution import OfficialRunJob
from Thesis_ML.orchestration import campaign_engine


def _write_index(path: Path) -> None:
    df = pd.DataFrame(
        [
            {
                "sample_id": "s1",
                "subject": "sub-001",
                "session": "ses-01",
                "task": "emo",
                "modality": "audio",
            },
            {
                "sample_id": "s2",
                "subject": "sub-002",
                "session": "ses-01",
                "task": "emo",
                "modality": "video",
            },
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _write_registry(path: Path) -> None:
    payload = {
        "schema_version": "test",
        "experiments": [
            {
                "experiment_id": "E02",
                "title": "Task pooling experiment",
                "stage": "Stage 1 - Target lock",
                "decision_id": "D01",
                "manipulated_factor": "task",
                "primary_metric": "balanced_accuracy",
                "variant_templates": [
                    {
                        "template_id": "e02_template",
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
            {
                "experiment_id": "E03",
                "title": "Modality pooling experiment",
                "stage": "Stage 1 - Target lock",
                "decision_id": "D01",
                "manipulated_factor": "modality",
                "primary_metric": "balanced_accuracy",
                "variant_templates": [
                    {
                        "template_id": "e03_template",
                        "supported": True,
                        "params": {
                            "target": "coarse_affect",
                            "model": "ridge",
                            "cv": "within_subject_loso_session",
                            "subject": "sub-002",
                        },
                    }
                ],
            },
            {
                "experiment_id": "E24",
                "title": "Reproducibility rerun audit",
                "stage": "Stage 6 - Robustness analysis",
                "decision_id": "D09",
                "manipulated_factor": "rerun",
                "primary_metric": "balanced_accuracy",
                "variant_templates": [
                    {
                        "template_id": "e24_template",
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
        ],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_reuse_registry(path: Path) -> None:
    payload = {
        "schema_version": "test",
        "experiments": [
            {
                "experiment_id": "E06",
                "title": "Model family lock",
                "stage": "Stage 3 - Model lock",
                "decision_id": "D03",
                "manipulated_factor": "model",
                "primary_metric": "balanced_accuracy",
                "variant_templates": [
                    {
                        "template_id": "ridge_variant",
                        "supported": True,
                        "params": {
                            "target": "coarse_affect",
                            "model": "ridge",
                            "cv": "within_subject_loso_session",
                            "subject": "sub-001",
                        },
                    },
                    {
                        "template_id": "balanced_variant",
                        "supported": True,
                        "n_permutations_override": 3,
                        "params": {
                            "target": "coarse_affect",
                            "model": "ridge",
                            "cv": "within_subject_loso_session",
                            "subject": "sub-001",
                        },
                    },
                ],
            }
        ],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_non_reuse_registry(path: Path, roi_spec_path: Path) -> None:
    payload = {
        "schema_version": "test",
        "experiments": [
            {
                "experiment_id": "E06",
                "title": "Representation lock",
                "stage": "Stage 3 - Model lock",
                "decision_id": "D03",
                "manipulated_factor": "feature_space",
                "primary_metric": "balanced_accuracy",
                "variant_templates": [
                    {
                        "template_id": "whole_brain_variant",
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
                        "template_id": "roi_variant",
                        "supported": True,
                        "params": {
                            "target": "coarse_affect",
                            "model": "ridge",
                            "cv": "within_subject_loso_session",
                            "subject": "sub-001",
                            "feature_space": "roi_masked",
                            "roi_spec_path": str(roi_spec_path),
                        },
                    },
                ],
            }
        ],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_cross_experiment_reuse_registry(path: Path) -> None:
    payload = {
        "schema_version": "test",
        "experiments": [
            {
                "experiment_id": "E02",
                "title": "Task pooling experiment",
                "stage": "Stage 1 - Target lock",
                "decision_id": "D01",
                "manipulated_factor": "task",
                "primary_metric": "balanced_accuracy",
                "variant_templates": [
                    {
                        "template_id": "e02_anchor",
                        "supported": True,
                        "params": {
                            "target": "coarse_affect",
                            "model": "ridge",
                            "cv": "within_subject_loso_session",
                            "subject": "sub-001",
                        },
                    },
                    {
                        "template_id": "e02_dependent",
                        "supported": True,
                        "n_permutations_override": 5,
                        "params": {
                            "target": "coarse_affect",
                            "model": "ridge",
                            "cv": "within_subject_loso_session",
                            "subject": "sub-001",
                        },
                    },
                ],
            },
            {
                "experiment_id": "E03",
                "title": "Modality pooling experiment",
                "stage": "Stage 1 - Target lock",
                "decision_id": "D01",
                "manipulated_factor": "modality",
                "primary_metric": "balanced_accuracy",
                "variant_templates": [
                    {
                        "template_id": "e03_anchor",
                        "supported": True,
                        "params": {
                            "target": "coarse_affect",
                            "model": "ridge",
                            "cv": "within_subject_loso_session",
                            "subject": "sub-001",
                        },
                    },
                    {
                        "template_id": "e03_dependent",
                        "supported": True,
                        "n_permutations_override": 7,
                        "params": {
                            "target": "coarse_affect",
                            "model": "ridge",
                            "cv": "within_subject_loso_session",
                            "subject": "sub-001",
                        },
                    },
                ],
            },
        ],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_e12_confirmatory_anchor_registry(path: Path) -> None:
    payload = {
        "schema_version": "test",
        "experiments": [
            {
                "experiment_id": "E12",
                "title": "Permutation test experiment",
                "stage": "Stage 6 - Robustness analysis",
                "decision_id": "D08",
                "manipulated_factor": "Label structure",
                "primary_metric": "balanced_accuracy",
                "variant_templates": [
                    {
                        "template_id": "e12_template",
                        "supported": True,
                        "params": {
                            "target": "coarse_affect",
                            "model": "ridge",
                            "cv": "within_subject_loso_session",
                            "anchor_experiment_id": "E16",
                        },
                    }
                ],
            },
            {
                "experiment_id": "E16",
                "title": "Final within-person confirmatory analysis",
                "stage": "Stage 5 - Confirmatory analysis",
                "decision_id": "CFM_LOCK_V1",
                "manipulated_factor": "none",
                "primary_metric": "balanced_accuracy",
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
            {
                "experiment_id": "E17",
                "title": "Final within-person confirmatory analysis",
                "stage": "Stage 5 - Confirmatory analysis",
                "decision_id": "CFM_LOCK_V1",
                "manipulated_factor": "none",
                "primary_metric": "balanced_accuracy",
                "variant_templates": [
                    {
                        "template_id": "e17_anchor",
                        "supported": True,
                        "params": {
                            "target": "coarse_affect",
                            "model": "ridge",
                            "cv": "within_subject_loso_session",
                            "subject": "sub-002",
                        },
                    }
                ],
            },
            {
                "experiment_id": "E18",
                "title": "Final cross-person confirmatory analysis",
                "stage": "Stage 5 - Confirmatory analysis",
                "decision_id": "CFM_LOCK_V1",
                "manipulated_factor": "none",
                "primary_metric": "balanced_accuracy",
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
                        },
                    }
                ],
            },
            {
                "experiment_id": "E19",
                "title": "Final cross-person confirmatory analysis",
                "stage": "Stage 5 - Confirmatory analysis",
                "decision_id": "CFM_LOCK_V1",
                "manipulated_factor": "none",
                "primary_metric": "balanced_accuracy",
                "variant_templates": [
                    {
                        "template_id": "e19_anchor",
                        "supported": True,
                        "params": {
                            "target": "coarse_affect",
                            "model": "ridge",
                            "cv": "frozen_cross_person_transfer",
                            "train_subject": "sub-002",
                            "test_subject": "sub-001",
                        },
                    }
                ],
            },
        ],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_e13_confirmatory_anchor_registry(path: Path) -> None:
    payload = {
        "schema_version": "test",
        "experiments": [
            {
                "experiment_id": "E13",
                "title": "Trivial baseline experiment",
                "stage": "Stage 6 - Robustness analysis",
                "decision_id": "D08",
                "manipulated_factor": "Baseline type",
                "primary_metric": "balanced_accuracy",
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
                "decision_id": "CFM_LOCK_V1",
                "manipulated_factor": "none",
                "primary_metric": "balanced_accuracy",
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
            {
                "experiment_id": "E17",
                "title": "Final cross-person confirmatory analysis",
                "stage": "Stage 5 - Confirmatory analysis",
                "decision_id": "CFM_LOCK_V1",
                "manipulated_factor": "none",
                "primary_metric": "balanced_accuracy",
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
        ],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _success_payload_for_jobs(jobs: list[OfficialRunJob]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for job in jobs:
        run_id = str(job.run_id)
        rows.append(
            {
                "order_index": int(job.order_index),
                "run_id": run_id,
                "started_at_utc": "2026-01-01T00:00:00+00:00",
                "ended_at_utc": "2026-01-01T00:00:01+00:00",
                "watchdog_result": {
                    "status": "success",
                    "run_payload": {
                        "run_id": run_id,
                        "report_dir": f"/tmp/{run_id}",
                        "config_path": f"/tmp/{run_id}/config.json",
                        "metrics_path": f"/tmp/{run_id}/metrics.json",
                        "fold_metrics_path": f"/tmp/{run_id}/fold_metrics.csv",
                        "fold_splits_path": f"/tmp/{run_id}/fold_splits.csv",
                        "predictions_path": f"/tmp/{run_id}/predictions.csv",
                        "spatial_compatibility_report_path": f"/tmp/{run_id}/spatial.json",
                        "metrics": {
                            "balanced_accuracy": 0.6,
                            "macro_f1": 0.58,
                            "accuracy": 0.62,
                            "n_folds": 2,
                        },
                    },
                },
                "execution_error": None,
            }
        )
    return rows


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
            "balanced_accuracy": 0.6,
            "macro_f1": 0.58,
            "accuracy": 0.62,
            "n_folds": 2,
        },
    }


def test_native_dispatch_uses_official_jobs_for_same_phase_group(
    tmp_path: Path,
    monkeypatch,
) -> None:
    registry_path = tmp_path / "registry.json"
    index_csv = tmp_path / "index.csv"
    _write_registry(registry_path)
    _write_index(index_csv)

    captured: list[dict[str, Any]] = []

    def _fake_execute_official_jobs(
        *,
        jobs: list[OfficialRunJob],
        max_parallel_runs: int,
        max_parallel_gpu_runs: int,
        run_experiment_fn=None,
    ):
        captured.append(
            {
                "jobs": jobs,
                "max_parallel_runs": int(max_parallel_runs),
                "max_parallel_gpu_runs": int(max_parallel_gpu_runs),
                "run_ids": [str(job.run_id) for job in jobs],
                "experiment_ids": [str(job.run_identity.get("experiment_id")) for job in jobs],
                "worker_execution_modes": [str(job.worker_execution_mode) for job in jobs],
            }
        )
        return _success_payload_for_jobs(jobs)

    monkeypatch.setattr(campaign_engine, "_execute_official_jobs", _fake_execute_official_jobs)

    campaign_engine.run_decision_support_campaign(
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
        dry_run=False,
        max_parallel_runs=3,
        max_parallel_gpu_runs=2,
        run_experiment_fn=_stub_run_experiment,
    )

    assert captured
    first_dispatch = captured[0]
    assert first_dispatch["max_parallel_runs"] == 3
    assert first_dispatch["max_parallel_gpu_runs"] == 2
    assert all(isinstance(job, OfficialRunJob) for job in first_dispatch["jobs"])
    assert {"E02", "E03"} <= set(first_dispatch["experiment_ids"])
    assert set(first_dispatch["worker_execution_modes"]) == {"native_worker"}


def test_e24_dispatch_forces_serial_execution(
    tmp_path: Path,
    monkeypatch,
) -> None:
    registry_path = tmp_path / "registry.json"
    index_csv = tmp_path / "index.csv"
    _write_registry(registry_path)
    _write_index(index_csv)

    captured_parallelism: list[tuple[int, int]] = []

    def _fake_execute_official_jobs(
        *,
        jobs: list[OfficialRunJob],
        max_parallel_runs: int,
        max_parallel_gpu_runs: int,
        run_experiment_fn=None,
    ):
        if any(str(job.run_identity.get("experiment_id")) == "E24" for job in jobs):
            captured_parallelism.append((int(max_parallel_runs), int(max_parallel_gpu_runs)))
        return _success_payload_for_jobs(jobs)

    monkeypatch.setattr(campaign_engine, "_execute_official_jobs", _fake_execute_official_jobs)

    campaign_engine.run_decision_support_campaign(
        registry_path=registry_path,
        index_csv=index_csv,
        data_root=tmp_path / "Data",
        cache_dir=tmp_path / "cache",
        output_root=tmp_path / "outputs",
        experiment_id="E24",
        stage=None,
        run_all=False,
        seed=42,
        n_permutations=0,
        dry_run=False,
        max_parallel_runs=4,
        max_parallel_gpu_runs=2,
        run_experiment_fn=_stub_run_experiment,
    )

    assert captured_parallelism == [(1, 0)]


def test_campaign_reuse_fan_out_starts_dependent_runs_from_shared_feature_matrix(
    tmp_path: Path,
    monkeypatch,
) -> None:
    registry_path = tmp_path / "registry_reuse.json"
    index_csv = tmp_path / "index.csv"
    _write_reuse_registry(registry_path)
    _write_index(index_csv)

    captured_dispatches: list[list[OfficialRunJob]] = []

    def _fake_execute_official_jobs(
        *,
        jobs: list[OfficialRunJob],
        max_parallel_runs: int,
        max_parallel_gpu_runs: int,
        run_experiment_fn=None,
    ):
        captured_dispatches.append(list(jobs))
        rows: list[dict[str, Any]] = []
        for job in jobs:
            run_id = str(job.run_id)
            rows.append(
                {
                    "run_id": run_id,
                    "watchdog_result": {
                        "status": "success",
                        "run_payload": {
                            "run_id": run_id,
                            "report_dir": f"/tmp/{run_id}",
                            "config_path": f"/tmp/{run_id}/config.json",
                            "metrics_path": f"/tmp/{run_id}/metrics.json",
                            "fold_metrics_path": f"/tmp/{run_id}/fold_metrics.csv",
                            "fold_splits_path": f"/tmp/{run_id}/fold_splits.csv",
                            "predictions_path": f"/tmp/{run_id}/predictions.csv",
                            "spatial_compatibility_report_path": f"/tmp/{run_id}/spatial.json",
                            "artifact_ids": {
                                "feature_cache": f"fc_{run_id}",
                                "feature_matrix_bundle": f"fm_{run_id}",
                            },
                            "metrics": {
                                "balanced_accuracy": 0.6,
                                "macro_f1": 0.58,
                                "accuracy": 0.62,
                                "n_folds": 2,
                            },
                        },
                    },
                    "execution_error": None,
                }
            )
        return rows

    monkeypatch.setattr(campaign_engine, "_execute_official_jobs", _fake_execute_official_jobs)

    result = campaign_engine.run_decision_support_campaign(
        registry_path=registry_path,
        index_csv=index_csv,
        data_root=tmp_path / "Data",
        cache_dir=tmp_path / "cache",
        output_root=tmp_path / "outputs",
        experiment_id="E06",
        stage=None,
        run_all=False,
        seed=42,
        n_permutations=0,
        dry_run=False,
        max_parallel_runs=4,
        max_parallel_gpu_runs=0,
        run_experiment_fn=_stub_run_experiment,
    )

    assert len(captured_dispatches) == 2
    assert len(captured_dispatches[0]) == 1
    assert len(captured_dispatches[1]) == 1

    anchor_job = captured_dispatches[0][0]
    dependent_job = captured_dispatches[1][0]
    assert dependent_job.run_kwargs["start_section"] == "spatial_validation"
    assert dependent_job.run_kwargs["base_artifact_id"] == f"fm_{anchor_job.run_id}"
    assert dependent_job.run_kwargs["reuse_policy"] == "require_explicit_base"

    campaign_id = Path(str(result["campaign_root"])).name
    manifests_dir = tmp_path / "outputs" / "E06" / campaign_id / "run_manifests"
    payloads = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(manifests_dir.glob("*.json"))
    ]
    assert any(
        str(row.get("config_used", {}).get("start_section")) == "spatial_validation"
        and str(row.get("config_used", {}).get("base_artifact_id", "")).startswith("fm_")
        for row in payloads
    )


def test_campaign_reuse_planner_blocks_cross_experiment_feature_matrix_reuse(
    tmp_path: Path,
    monkeypatch,
) -> None:
    registry_path = tmp_path / "registry_cross_experiment_reuse.json"
    index_csv = tmp_path / "index.csv"
    _write_cross_experiment_reuse_registry(registry_path)
    _write_index(index_csv)

    captured_dispatches: list[list[OfficialRunJob]] = []

    def _fake_execute_official_jobs(
        *,
        jobs: list[OfficialRunJob],
        max_parallel_runs: int,
        max_parallel_gpu_runs: int,
        run_experiment_fn=None,
    ):
        captured_dispatches.append(list(jobs))
        rows: list[dict[str, Any]] = []
        for job in jobs:
            run_id = str(job.run_id)
            rows.append(
                {
                    "run_id": run_id,
                    "watchdog_result": {
                        "status": "success",
                        "run_payload": {
                            "run_id": run_id,
                            "report_dir": f"/tmp/{run_id}",
                            "config_path": f"/tmp/{run_id}/config.json",
                            "metrics_path": f"/tmp/{run_id}/metrics.json",
                            "fold_metrics_path": f"/tmp/{run_id}/fold_metrics.csv",
                            "fold_splits_path": f"/tmp/{run_id}/fold_splits.csv",
                            "predictions_path": f"/tmp/{run_id}/predictions.csv",
                            "spatial_compatibility_report_path": f"/tmp/{run_id}/spatial.json",
                            "artifact_ids": {
                                "feature_cache": f"fc_{run_id}",
                                "feature_matrix_bundle": f"fm_{run_id}",
                            },
                            "metrics": {
                                "balanced_accuracy": 0.6,
                                "macro_f1": 0.58,
                                "accuracy": 0.62,
                                "n_folds": 2,
                            },
                        },
                    },
                    "execution_error": None,
                }
            )
        return rows

    monkeypatch.setattr(campaign_engine, "_execute_official_jobs", _fake_execute_official_jobs)

    campaign_engine.run_decision_support_campaign(
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
        dry_run=False,
        max_parallel_runs=4,
        max_parallel_gpu_runs=0,
        run_experiment_fn=_stub_run_experiment,
    )

    all_jobs = [job for dispatch in captured_dispatches for job in dispatch]
    assert len(all_jobs) == 4

    experiment_by_run_id = {
        str(job.run_id): str(job.run_identity.get("experiment_id")) for job in all_jobs
    }
    sections_by_experiment: dict[str, list[str]] = {"E02": [], "E03": []}
    for job in all_jobs:
        sections_by_experiment[str(job.run_identity.get("experiment_id"))].append(
            str(job.run_kwargs.get("start_section"))
        )

    assert sorted(sections_by_experiment["E02"]) == [
        "dataset_selection",
        "spatial_validation",
    ]
    assert sorted(sections_by_experiment["E03"]) == [
        "dataset_selection",
        "spatial_validation",
    ]

    dependent_jobs = [
        job for job in all_jobs if str(job.run_kwargs.get("start_section")) == "spatial_validation"
    ]
    assert {experiment_by_run_id[str(job.run_id)] for job in dependent_jobs} == {
        "E02",
        "E03",
    }
    for job in dependent_jobs:
        base_artifact_id = str(job.run_kwargs.get("base_artifact_id"))
        assert base_artifact_id.startswith("fm_")
        anchor_run_id = base_artifact_id.removeprefix("fm_")
        assert experiment_by_run_id[anchor_run_id] == experiment_by_run_id[str(job.run_id)]


def test_campaign_reuse_planner_does_not_coalesce_different_feature_spaces(
    tmp_path: Path,
    monkeypatch,
) -> None:
    registry_path = tmp_path / "registry_no_reuse.json"
    index_csv = tmp_path / "index.csv"
    roi_spec_path = tmp_path / "roi_spec.json"
    roi_spec_path.write_text(json.dumps({"name": "roi_a", "voxels": [1, 2, 3]}), encoding="utf-8")
    _write_non_reuse_registry(registry_path, roi_spec_path)
    _write_index(index_csv)

    captured_dispatches: list[list[OfficialRunJob]] = []

    def _fake_execute_official_jobs(
        *,
        jobs: list[OfficialRunJob],
        max_parallel_runs: int,
        max_parallel_gpu_runs: int,
        run_experiment_fn=None,
    ):
        captured_dispatches.append(list(jobs))
        return _success_payload_for_jobs(jobs)

    monkeypatch.setattr(campaign_engine, "_execute_official_jobs", _fake_execute_official_jobs)

    campaign_engine.run_decision_support_campaign(
        registry_path=registry_path,
        index_csv=index_csv,
        data_root=tmp_path / "Data",
        cache_dir=tmp_path / "cache",
        output_root=tmp_path / "outputs",
        experiment_id="E06",
        stage=None,
        run_all=False,
        seed=42,
        n_permutations=0,
        dry_run=False,
        max_parallel_runs=4,
        max_parallel_gpu_runs=0,
        run_experiment_fn=_stub_run_experiment,
    )

    assert captured_dispatches
    dispatched_jobs = captured_dispatches[0]
    assert len(dispatched_jobs) == 2
    assert all(job.run_kwargs["start_section"] == "dataset_selection" for job in dispatched_jobs)
    assert all(job.run_kwargs["base_artifact_id"] is None for job in dispatched_jobs)


def test_native_dispatch_e12_inherits_anchor_subject_for_chunks(
    tmp_path: Path,
    monkeypatch,
) -> None:
    registry_path = tmp_path / "registry_e12.json"
    index_csv = tmp_path / "index.csv"
    _write_e12_confirmatory_anchor_registry(registry_path)
    _write_index(index_csv)

    captured_jobs: list[OfficialRunJob] = []

    def _fake_execute_official_jobs(
        *,
        jobs: list[OfficialRunJob],
        max_parallel_runs: int,
        max_parallel_gpu_runs: int,
        run_experiment_fn=None,
    ):
        captured_jobs.extend(jobs)
        return _success_payload_for_jobs(jobs)

    monkeypatch.setattr(campaign_engine, "_execute_official_jobs", _fake_execute_official_jobs)

    campaign_engine.run_decision_support_campaign(
        registry_path=registry_path,
        index_csv=index_csv,
        data_root=tmp_path / "Data",
        cache_dir=tmp_path / "cache",
        output_root=tmp_path / "outputs",
        experiment_id="E12",
        stage=None,
        run_all=False,
        seed=42,
        n_permutations=120,
        dry_run=False,
        max_parallel_runs=3,
        max_parallel_gpu_runs=0,
        run_experiment_fn=_stub_run_experiment,
    )

    assert len(captured_jobs) == 12
    assert sorted(int(job.run_kwargs.get("n_permutations", 0)) for job in captured_jobs) == (
        [20] * 4 + [50] * 8
    )
    within_subjects = {
        str(job.run_kwargs.get("subject"))
        for job in captured_jobs
        if str(job.run_kwargs.get("cv")) == "within_subject_loso_session"
    }
    transfer_pairs = {
        (str(job.run_kwargs.get("train_subject")), str(job.run_kwargs.get("test_subject")))
        for job in captured_jobs
        if str(job.run_kwargs.get("cv")) == "frozen_cross_person_transfer"
    }
    assert within_subjects == {"sub-001", "sub-002"}
    assert transfer_pairs == {("sub-001", "sub-002"), ("sub-002", "sub-001")}


def test_native_dispatch_e13_inherits_anchor_identities_and_uses_dummy_model(
    tmp_path: Path,
    monkeypatch,
) -> None:
    registry_path = tmp_path / "registry_e13.json"
    index_csv = tmp_path / "index.csv"
    _write_e13_confirmatory_anchor_registry(registry_path)
    _write_index(index_csv)

    captured_jobs: list[OfficialRunJob] = []

    def _fake_execute_official_jobs(
        *,
        jobs: list[OfficialRunJob],
        max_parallel_runs: int,
        max_parallel_gpu_runs: int,
        run_experiment_fn=None,
    ):
        captured_jobs.extend(jobs)
        return _success_payload_for_jobs(jobs)

    monkeypatch.setattr(campaign_engine, "_execute_official_jobs", _fake_execute_official_jobs)

    campaign_engine.run_decision_support_campaign(
        registry_path=registry_path,
        index_csv=index_csv,
        data_root=tmp_path / "Data",
        cache_dir=tmp_path / "cache",
        output_root=tmp_path / "outputs",
        experiment_id="E13",
        stage=None,
        run_all=False,
        seed=42,
        n_permutations=0,
        dry_run=False,
        max_parallel_runs=2,
        max_parallel_gpu_runs=0,
        run_experiment_fn=_stub_run_experiment,
    )

    assert len(captured_jobs) == 2
    assert {str(job.run_kwargs.get("model")) for job in captured_jobs} == {"dummy"}
    assert {
        str(job.run_kwargs.get("subject")) for job in captured_jobs if job.run_kwargs.get("subject")
    } == {"sub-001"}
    assert {
        (str(job.run_kwargs.get("train_subject")), str(job.run_kwargs.get("test_subject")))
        for job in captured_jobs
        if job.run_kwargs.get("train_subject") or job.run_kwargs.get("test_subject")
    } == {("sub-001", "sub-002")}
