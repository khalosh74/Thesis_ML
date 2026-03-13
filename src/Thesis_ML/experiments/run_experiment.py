"""Leakage-safe experiment runner with grouped cross-validation."""

from __future__ import annotations

import argparse
import json
import logging
import platform
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from Thesis_ML.artifacts.registry import (
    ARTIFACT_TYPE_EXPERIMENT_REPORT,
    ARTIFACT_TYPE_INTERPRETABILITY_BUNDLE,
    ARTIFACT_TYPE_METRICS_BUNDLE,
    compute_config_hash,
    register_artifact,
)
from Thesis_ML.config.paths import DEFAULT_EXPERIMENT_REPORTS_ROOT
from Thesis_ML.experiments.cache_loading import load_features_from_cache
from Thesis_ML.experiments.execution_policy import (
    prepare_report_dir,
    write_run_status,
)
from Thesis_ML.experiments.metrics import (
    compute_interpretability_stability,
    evaluate_permutations,
    extract_linear_coefficients,
    scores_for_predictions,
)
from Thesis_ML.experiments.model_factory import MODEL_NAMES, make_model
from Thesis_ML.experiments.segment_execution import (
    SegmentExecutionRequest,
    execute_section_segment,
)
from Thesis_ML.experiments.spatial_validation import SPATIAL_AFFINE_ATOL

LOGGER = logging.getLogger(__name__)

_CV_MODES = ("loso_session", "within_subject_loso_session", "frozen_cross_person_transfer")
_TARGET_ALIASES = {
    "emotion": "emotion",
    "coarse_affect": "coarse_affect",
    "modality": "modality",
    "task": "task",
    "regressor_label": "regressor_label",
}


def _current_git_commit() -> str | None:
    try:
        process = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return process.stdout.strip() or None


def _make_model(name: str, seed: int) -> Any:
    return make_model(name=name, seed=seed)


def _build_pipeline(model_name: str, seed: int):
    model = _make_model(name=model_name, seed=seed)
    # fMRI voxel vectors are dense numeric arrays; centered scaling is appropriate.
    return Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("model", model),
        ]
    )


def _resolve_target_column(target: str) -> str:
    if target not in _TARGET_ALIASES:
        allowed = ", ".join(sorted(_TARGET_ALIASES))
        raise ValueError(f"Unsupported target '{target}'. Allowed values: {allowed}")
    return _TARGET_ALIASES[target]


def _resolve_cv_mode(cv: str) -> str:
    if cv not in _CV_MODES:
        allowed = ", ".join(_CV_MODES)
        raise ValueError(f"Unsupported cv '{cv}'. Allowed values: {allowed}")
    return cv


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(f"{json.dumps(payload, indent=2)}\n", encoding="utf-8")


def _load_features_from_cache(
    index_df: pd.DataFrame,
    cache_manifest_path: Path,
    spatial_report_path: Path | None = None,
    affine_atol: float = SPATIAL_AFFINE_ATOL,
) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any]]:
    return load_features_from_cache(
        index_df=index_df,
        cache_manifest_path=cache_manifest_path,
        spatial_report_path=spatial_report_path,
        affine_atol=affine_atol,
    )


def _scores_for_predictions(estimator, x_test: np.ndarray) -> dict[str, list[Any]]:
    return scores_for_predictions(estimator=estimator, x_test=x_test)


def _evaluate_permutations(
    pipeline_template,
    x_matrix: np.ndarray,
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    seed: int,
    n_permutations: int,
    observed_accuracy: float,
) -> dict[str, Any]:
    return evaluate_permutations(
        pipeline_template=pipeline_template,
        x_matrix=x_matrix,
        y=y,
        splits=splits,
        seed=seed,
        n_permutations=n_permutations,
        observed_accuracy=observed_accuracy,
    )


def _extract_linear_coefficients(estimator) -> tuple[np.ndarray, np.ndarray, list[str]]:
    return extract_linear_coefficients(estimator=estimator)


def _compute_interpretability_stability(coef_vectors: list[np.ndarray]) -> dict[str, Any]:
    return compute_interpretability_stability(coef_vectors=coef_vectors)


def run_experiment(
    index_csv: Path,
    data_root: Path,
    cache_dir: Path,
    target: str,
    model: str,
    cv: str | None = None,
    subject: str | None = None,
    train_subject: str | None = None,
    test_subject: str | None = None,
    seed: int = 42,
    filter_task: str | None = None,
    filter_modality: str | None = None,
    n_permutations: int = 0,
    run_id: str | None = None,
    reports_root: Path | str = DEFAULT_EXPERIMENT_REPORTS_ROOT,
    start_section: str | None = None,
    end_section: str | None = None,
    base_artifact_id: str | None = None,
    reuse_policy: str | None = None,
    force: bool = False,
    resume: bool = False,
    reuse_completed_artifacts: bool = False,
) -> dict[str, Any]:
    """Run one leakage-safe grouped-CV experiment and write standardized artifacts."""
    index_csv = Path(index_csv)
    data_root = Path(data_root)
    cache_dir = Path(cache_dir)
    reports_root = Path(reports_root)

    if cv is None or not str(cv).strip():
        allowed = ", ".join(_CV_MODES)
        raise ValueError(
            f"run_experiment requires explicit cv mode selection. Provide one of: {allowed}"
        )
    cv_mode = _resolve_cv_mode(str(cv).strip())
    if cv_mode == "within_subject_loso_session":
        if subject is None or not str(subject).strip():
            raise ValueError("cv='within_subject_loso_session' requires a non-empty subject.")
        subject = str(subject).strip()
    if cv_mode == "frozen_cross_person_transfer":
        if train_subject is None or not str(train_subject).strip():
            raise ValueError(
                "cv='frozen_cross_person_transfer' requires a non-empty train_subject."
            )
        if test_subject is None or not str(test_subject).strip():
            raise ValueError("cv='frozen_cross_person_transfer' requires a non-empty test_subject.")
        train_subject = str(train_subject).strip()
        test_subject = str(test_subject).strip()
        if train_subject == test_subject:
            raise ValueError("train_subject and test_subject must be different.")

    target_column = _resolve_target_column(target)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    resolved_run_id = run_id or f"{timestamp}_{model}_{target_column}"
    report_dir = reports_root / resolved_run_id
    run_mode = prepare_report_dir(
        report_dir,
        run_id=resolved_run_id,
        force=bool(force),
        resume=bool(resume),
    )
    should_reuse_completed_artifacts = bool(resume or reuse_completed_artifacts)
    artifact_registry_path = reports_root / "artifact_registry.sqlite3"
    code_ref = _current_git_commit()

    fold_metrics_path = report_dir / "fold_metrics.csv"
    fold_splits_path = report_dir / "fold_splits.csv"
    predictions_path = report_dir / "predictions.csv"
    metrics_path = report_dir / "metrics.json"
    config_path = report_dir / "config.json"
    spatial_compatibility_report_path = report_dir / "spatial_compatibility_report.json"
    interpretability_summary_path = report_dir / "interpretability_summary.json"
    interpretability_fold_artifacts_path = report_dir / "interpretability_fold_explanations.csv"

    write_run_status(
        report_dir,
        run_id=resolved_run_id,
        status="running",
        message=f"run_mode={run_mode}",
    )
    try:
        segment_result = execute_section_segment(
            SegmentExecutionRequest(
                index_csv=index_csv,
                data_root=data_root,
                cache_dir=cache_dir,
                target_column=target_column,
                cv_mode=cv_mode,
                model=model,
                subject=subject,
                train_subject=train_subject,
                test_subject=test_subject,
                filter_task=filter_task,
                filter_modality=filter_modality,
                seed=seed,
                n_permutations=n_permutations,
                run_id=resolved_run_id,
                config_filename=config_path.name,
                report_dir=report_dir,
                artifact_registry_path=artifact_registry_path,
                code_ref=code_ref,
                affine_atol=SPATIAL_AFFINE_ATOL,
                fold_metrics_path=fold_metrics_path,
                fold_splits_path=fold_splits_path,
                predictions_path=predictions_path,
                metrics_path=metrics_path,
                spatial_report_path=spatial_compatibility_report_path,
                interpretability_summary_path=interpretability_summary_path,
                interpretability_fold_artifacts_path=interpretability_fold_artifacts_path,
                start_section=start_section,
                end_section=end_section,
                base_artifact_id=base_artifact_id,
                reuse_policy=reuse_policy,
                reuse_completed_artifacts=should_reuse_completed_artifacts,
                build_pipeline_fn=_build_pipeline,
                load_features_from_cache_fn=_load_features_from_cache,
                scores_for_predictions_fn=_scores_for_predictions,
                extract_linear_coefficients_fn=_extract_linear_coefficients,
                compute_interpretability_stability_fn=_compute_interpretability_stability,
                evaluate_permutations_fn=_evaluate_permutations,
            )
        )
    except Exception as exc:
        write_run_status(
            report_dir,
            run_id=resolved_run_id,
            status="failed",
            error=str(exc),
        )
        raise
    artifact_ids = dict(segment_result.artifact_ids)
    metrics = dict(segment_result.metrics or {})
    spatial_compatibility = segment_result.spatial_compatibility
    interpretability_summary = segment_result.interpretability_summary

    spatial_status = str(spatial_compatibility["status"]) if spatial_compatibility else None
    spatial_passed = bool(spatial_compatibility["passed"]) if spatial_compatibility else None
    spatial_groups_checked = (
        int(spatial_compatibility["n_groups_checked"]) if spatial_compatibility else None
    )
    spatial_reference_group = (
        spatial_compatibility["reference_group_id"] if spatial_compatibility else None
    )
    spatial_affine_atol = (
        float(spatial_compatibility["affine_atol"]) if spatial_compatibility else None
    )

    interpretability_enabled = (
        bool(interpretability_summary["enabled"]) if interpretability_summary else None
    )
    interpretability_performed = (
        bool(interpretability_summary["performed"]) if interpretability_summary else None
    )
    interpretability_status = (
        str(interpretability_summary["status"]) if interpretability_summary else None
    )
    interpretability_fold_artifacts = (
        interpretability_summary.get("fold_artifacts_path") if interpretability_summary else None
    )

    config = {
        "run_id": resolved_run_id,
        "timestamp": timestamp,
        "index_csv": str(index_csv.resolve()),
        "data_root": str(data_root.resolve()),
        "cache_dir": str(cache_dir.resolve()),
        "target": target_column,
        "model": model,
        "cv": cv_mode,
        "experiment_mode": cv_mode,
        "subject": str(subject) if cv_mode == "within_subject_loso_session" else None,
        "train_subject": (
            str(train_subject) if cv_mode == "frozen_cross_person_transfer" else None
        ),
        "test_subject": str(test_subject) if cv_mode == "frozen_cross_person_transfer" else None,
        "seed": int(seed),
        "filter_task": filter_task,
        "filter_modality": filter_modality,
        "n_permutations": int(n_permutations),
        "start_section": start_section,
        "end_section": end_section,
        "base_artifact_id": base_artifact_id,
        "reuse_policy": reuse_policy,
        "force": bool(force),
        "resume": bool(resume),
        "reuse_completed_artifacts": bool(should_reuse_completed_artifacts),
        "run_mode": run_mode,
        "planned_sections": segment_result.planned_sections,
        "executed_sections": segment_result.executed_sections,
        "reused_sections": segment_result.reused_sections,
        "fold_splits_path": str(fold_splits_path.resolve()),
        "spatial_compatibility_status": spatial_status,
        "spatial_compatibility_passed": spatial_passed,
        "spatial_compatibility_n_groups_checked": spatial_groups_checked,
        "spatial_compatibility_reference_group_id": spatial_reference_group,
        "spatial_compatibility_affine_atol": spatial_affine_atol,
        "spatial_compatibility_report_path": str(spatial_compatibility_report_path.resolve()),
        "interpretability_enabled": interpretability_enabled,
        "interpretability_performed": interpretability_performed,
        "interpretability_status": interpretability_status,
        "interpretability_fold_artifacts_path": interpretability_fold_artifacts,
        "interpretability_summary_path": str(interpretability_summary_path.resolve()),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "sklearn_version": sklearn.__version__,
        "nibabel_version": nib.__version__,
        "git_commit": code_ref,
    }
    config_path.write_text(f"{json.dumps(config, indent=2)}\n", encoding="utf-8")

    report_upstream_candidates = [
        artifact_ids.get(ARTIFACT_TYPE_METRICS_BUNDLE),
        artifact_ids.get(ARTIFACT_TYPE_INTERPRETABILITY_BUNDLE),
    ]
    report_upstream = [
        artifact_id for artifact_id in report_upstream_candidates if isinstance(artifact_id, str)
    ]
    experiment_report_artifact = register_artifact(
        registry_path=artifact_registry_path,
        artifact_type=ARTIFACT_TYPE_EXPERIMENT_REPORT,
        run_id=resolved_run_id,
        upstream_artifact_ids=report_upstream,
        config_hash=compute_config_hash({"run_id": resolved_run_id, "report_dir": str(report_dir)}),
        code_ref=code_ref,
        path=report_dir,
        status="created",
    )
    artifact_ids[ARTIFACT_TYPE_EXPERIMENT_REPORT] = experiment_report_artifact.artifact_id
    run_status = write_run_status(
        report_dir,
        run_id=resolved_run_id,
        status="completed",
        executed_sections=segment_result.executed_sections,
        reused_sections=segment_result.reused_sections,
    )

    return {
        "run_id": resolved_run_id,
        "report_dir": str(report_dir.resolve()),
        "config_path": str(config_path.resolve()),
        "metrics_path": str(metrics_path.resolve()),
        "fold_metrics_path": str(fold_metrics_path.resolve()),
        "fold_splits_path": str(fold_splits_path.resolve()),
        "predictions_path": str(predictions_path.resolve()),
        "spatial_compatibility_report_path": str(spatial_compatibility_report_path.resolve()),
        "interpretability_summary_path": str(interpretability_summary_path.resolve()),
        "interpretability_fold_artifacts_path": interpretability_fold_artifacts,
        "artifact_registry_path": str(artifact_registry_path.resolve()),
        "planned_sections": segment_result.planned_sections,
        "executed_sections": segment_result.executed_sections,
        "reused_sections": segment_result.reused_sections,
        "artifact_ids": artifact_ids,
        "metrics": metrics,
        "run_status_path": str(run_status.resolve()),
        "run_mode": run_mode,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run leakage-safe grouped-CV fMRI experiments.")
    parser.add_argument("--index-csv", required=True, help="Dataset index CSV.")
    parser.add_argument("--data-root", required=True, help="Root directory for relative paths.")
    parser.add_argument("--cache-dir", required=True, help="Feature cache directory.")
    parser.add_argument(
        "--target", required=True, choices=sorted(_TARGET_ALIASES), help="Target label."
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=[*MODEL_NAMES, "all"],
        help="Model to evaluate.",
    )
    parser.add_argument(
        "--cv",
        required=True,
        choices=list(_CV_MODES),
        help=(
            "Experiment mode. Required. within_subject_loso_session=primary thesis mode; "
            "frozen_cross_person_transfer=secondary transfer mode; "
            "loso_session=auxiliary grouped mode."
        ),
    )
    parser.add_argument(
        "--subject",
        default=None,
        help=(
            "Subject identifier (required for cv=within_subject_loso_session; for example sub-001)."
        ),
    )
    parser.add_argument(
        "--train-subject",
        default=None,
        help=("Training subject identifier (required for cv=frozen_cross_person_transfer)."),
    )
    parser.add_argument(
        "--test-subject",
        default=None,
        help=("Test subject identifier (required for cv=frozen_cross_person_transfer)."),
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--filter-task", default=None, help="Optional task filter.")
    parser.add_argument("--filter-modality", default=None, help="Optional modality filter.")
    parser.add_argument(
        "--n-permutations",
        type=int,
        default=0,
        help="Number of permutation rounds for optional significance testing.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional run identifier. If omitted, timestamp-based ID is used.",
    )
    parser.add_argument(
        "--reports-root",
        default=str(DEFAULT_EXPERIMENT_REPORTS_ROOT),
        help="Root directory for experiment reports.",
    )
    parser.add_argument(
        "--start-section",
        default=None,
        help="Optional first section to execute for segmented runs.",
    )
    parser.add_argument(
        "--end-section",
        default=None,
        help="Optional last section to execute for segmented runs.",
    )
    parser.add_argument(
        "--base-artifact-id",
        default=None,
        help=(
            "Optional artifact ID used to resume segmented runs when start-section "
            "is after feature_cache_build."
        ),
    )
    parser.add_argument(
        "--reuse-policy",
        default=None,
        help="Optional reuse policy for segmented runs (auto, require_explicit_base, disallow).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force rerun: clear existing run output directory before execution.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume a partial run from existing output directory.",
    )
    parser.add_argument(
        "--reuse-completed-artifacts",
        action="store_true",
        help="Allow reusing completed same-run section artifacts when available.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    models = list(MODEL_NAMES) if args.model == "all" else [args.model]
    results: list[dict[str, Any]] = []

    for model_name in models:
        model_run_id = args.run_id
        if args.model == "all":
            base = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
            model_run_id = f"{base}_{model_name}_{args.target}"

        result = run_experiment(
            index_csv=Path(args.index_csv),
            data_root=Path(args.data_root),
            cache_dir=Path(args.cache_dir),
            target=args.target,
            model=model_name,
            cv=args.cv,
            subject=args.subject,
            train_subject=args.train_subject,
            test_subject=args.test_subject,
            seed=args.seed,
            filter_task=args.filter_task,
            filter_modality=args.filter_modality,
            n_permutations=args.n_permutations,
            run_id=model_run_id,
            reports_root=Path(args.reports_root),
            start_section=args.start_section,
            end_section=args.end_section,
            base_artifact_id=args.base_artifact_id,
            reuse_policy=args.reuse_policy,
            force=bool(args.force),
            resume=bool(args.resume),
            reuse_completed_artifacts=bool(args.reuse_completed_artifacts),
        )
        results.append(
            {
                "model": model_name,
                "run_id": result["run_id"],
                "report_dir": result["report_dir"],
                "accuracy": result["metrics"].get("accuracy"),
                "balanced_accuracy": result["metrics"].get("balanced_accuracy"),
                "macro_f1": result["metrics"].get("macro_f1"),
            }
        )

    print(json.dumps({"results": results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
