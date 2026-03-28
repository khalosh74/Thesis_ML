from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

EXPERIMENT_ID = "E01"
PRIMARY_TARGETS = {"coarse_affect", "emotion"}
BENCHMARK_ONLY_TARGET = "binary_valence_like"
REQUIRED_METRIC_KEYS = (
    "target",
    "subject",
    "balanced_accuracy",
    "macro_f1",
    "accuracy",
    "n_samples",
    "n_folds",
    "labels",
    "confusion_matrix",
)


class ReviewE01Error(RuntimeError):
    """Raised when required E01 review inputs are missing or malformed."""


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Review E01 target-lock evidence from completed decision-support runs."
    )
    parser.add_argument(
        "--campaign-root",
        type=Path,
        required=True,
        help="Campaign root containing decision_support_summary.csv and run_log_export.csv.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.05,
        help="Tolerance margin for coarse_affect-vs-primary comparisons (default: 0.05).",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Output directory (default: <campaign-root>/e01_review).",
    )
    return parser


def _require_file(path: Path, *, label: str) -> Path:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def _resolve_column(frame: pd.DataFrame, logical_name: str, candidates: tuple[str, ...]) -> str:
    for name in candidates:
        if name in frame.columns:
            return name
    available = ", ".join(str(col) for col in frame.columns)
    raise ReviewE01Error(
        f"Required column '{logical_name}' is missing. Expected one of {candidates}. "
        f"Available columns: {available}"
    )


def _coerce_float(value: Any, *, key: str, metrics_path: Path) -> float:
    try:
        return float(value)
    except Exception as exc:
        raise ReviewE01Error(
            f"Invalid numeric value for key '{key}' in metrics '{metrics_path}': {value!r}"
        ) from exc


def _resolve_path_from_text(path_text: str, *, campaign_root: Path) -> Path:
    raw = str(path_text).strip()
    if not raw:
        raise ReviewE01Error("Encountered empty path text while resolving metrics path.")
    candidate = Path(raw)
    if candidate.is_absolute():
        return candidate
    campaign_relative = (campaign_root / candidate).resolve()
    if campaign_relative.exists():
        return campaign_relative
    return candidate.resolve()


def _resolve_metrics_path_from_row(
    row: pd.Series,
    *,
    campaign_root: Path,
    metrics_path_column: str | None,
    artifact_path_column: str | None,
) -> Path:
    if metrics_path_column is not None:
        raw_metrics_path = str(row.get(metrics_path_column, "")).strip()
        if raw_metrics_path:
            return _resolve_path_from_text(raw_metrics_path, campaign_root=campaign_root)

    if artifact_path_column is None:
        raise ReviewE01Error(
            "Run log row is missing metrics_path and no artifact-path fallback column is available."
        )

    raw_artifact_path = str(row.get(artifact_path_column, "")).strip()
    if not raw_artifact_path:
        raise ReviewE01Error(
            "Run log row is missing both metrics_path and artifact path; cannot locate metrics JSON."
        )

    artifact_path = _resolve_path_from_text(raw_artifact_path, campaign_root=campaign_root)
    if artifact_path.is_dir():
        candidate = artifact_path / "metrics.json"
        return candidate

    if artifact_path.is_file() and artifact_path.suffix.lower() == ".json":
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
        metrics_from_manifest = str(payload.get("metrics_path", "")).strip()
        if metrics_from_manifest:
            return _resolve_path_from_text(metrics_from_manifest, campaign_root=campaign_root)

    raise ReviewE01Error(
        f"Could not resolve metrics path from run-log row artifact path '{artifact_path}'."
    )


def _extract_required_metrics(metrics_path: Path) -> dict[str, Any]:
    _require_file(metrics_path, label="Metrics JSON")
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ReviewE01Error(f"Metrics payload must be a JSON object: {metrics_path}")
    missing = [key for key in REQUIRED_METRIC_KEYS if key not in payload]
    if missing:
        missing_text = ", ".join(missing)
        raise ReviewE01Error(
            f"Metrics payload missing required keys ({missing_text}) at {metrics_path}"
        )
    return payload


def _collapse_from_confusion_matrix(confusion_matrix: Any, *, subject: str) -> bool:
    if not isinstance(confusion_matrix, list):
        raise ReviewE01Error(f"confusion_matrix must be a list for subject '{subject}'.")

    for row_index, row in enumerate(confusion_matrix):
        if not isinstance(row, list):
            raise ReviewE01Error(
                f"confusion_matrix row {row_index} must be a list for subject '{subject}'."
            )
        if row_index >= len(row):
            raise ReviewE01Error(
                f"confusion_matrix row {row_index} has no diagonal entry for subject '{subject}'."
            )
        try:
            row_total = sum(float(cell) for cell in row)
            diagonal = float(row[row_index])
        except Exception as exc:
            raise ReviewE01Error(
                f"confusion_matrix contains non-numeric values for subject '{subject}'."
            ) from exc
        if row_total > 0.0 and diagonal == 0.0:
            return True
    return False


def _row_sort_key(row: dict[str, Any]) -> tuple[float, float]:
    return (float(row["balanced_accuracy"]), float(row["macro_f1"]))


def _best_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ReviewE01Error("Cannot pick best row from an empty row list.")
    return max(rows, key=_row_sort_key)


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"))


def _build_per_subject_rows(
    run_rows: pd.DataFrame,
    *,
    campaign_root: Path,
) -> list[dict[str, Any]]:
    metrics_path_column = None
    if "metrics_path" in run_rows.columns:
        metrics_path_column = "metrics_path"
    elif "Metrics_Path" in run_rows.columns:
        metrics_path_column = "Metrics_Path"

    artifact_path_column = None
    if "Artifact_Path" in run_rows.columns:
        artifact_path_column = "Artifact_Path"
    elif "artifact_path" in run_rows.columns:
        artifact_path_column = "artifact_path"

    variant_col = None
    for candidate in ("variant_id", "Variant_ID", "Run_ID", "run_id"):
        if candidate in run_rows.columns:
            variant_col = candidate
            break
    if variant_col is None:
        raise ReviewE01Error(
            "Run log is missing variant identity columns; expected one of "
            "('variant_id', 'Variant_ID', 'Run_ID', 'run_id')."
        )

    rows: list[dict[str, Any]] = []
    for _, run_row in run_rows.iterrows():
        metrics_path = _resolve_metrics_path_from_row(
            run_row,
            campaign_root=campaign_root,
            metrics_path_column=metrics_path_column,
            artifact_path_column=artifact_path_column,
        )
        payload = _extract_required_metrics(metrics_path)

        summary_row = {
            "experiment_id": EXPERIMENT_ID,
            "variant_id": str(run_row.get(variant_col, "")).strip(),
            "subject": str(payload["subject"]),
            "target": str(payload["target"]),
            "balanced_accuracy": _coerce_float(
                payload["balanced_accuracy"], key="balanced_accuracy", metrics_path=metrics_path
            ),
            "macro_f1": _coerce_float(payload["macro_f1"], key="macro_f1", metrics_path=metrics_path),
            "accuracy": _coerce_float(payload["accuracy"], key="accuracy", metrics_path=metrics_path),
            "n_samples": int(payload["n_samples"]),
            "n_folds": int(payload["n_folds"]),
            "labels": payload["labels"],
            "confusion_matrix": payload["confusion_matrix"],
            "metrics_path": str(metrics_path.resolve()),
        }
        rows.append(summary_row)
    return rows


def _reduce_to_best_per_subject_target(rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for row in rows:
        grouped.setdefault(row["subject"], {}).setdefault(row["target"], []).append(row)

    reduced: dict[str, dict[str, dict[str, Any]]] = {}
    for subject, target_rows in grouped.items():
        reduced[subject] = {target: _best_row(records) for target, records in target_rows.items()}
    return reduced


def _build_subject_analysis(
    reduced_rows: dict[str, dict[str, dict[str, Any]]],
    *,
    tolerance: float,
) -> dict[str, dict[str, Any]]:
    analysis: dict[str, dict[str, Any]] = {}
    for subject, target_rows in reduced_rows.items():
        primary_candidates = [row for target, row in target_rows.items() if target in PRIMARY_TARGETS]
        best_primary = _best_row(primary_candidates) if primary_candidates else None
        best_primary_target = str(best_primary["target"]) if best_primary else None

        coarse_row = target_rows.get("coarse_affect")
        emotion_row = target_rows.get("emotion")

        coarse_within_tolerance = False
        if coarse_row is not None and best_primary is not None:
            coarse_within_tolerance = bool(
                float(coarse_row["balanced_accuracy"]) >= float(best_primary["balanced_accuracy"]) - tolerance
                and float(coarse_row["macro_f1"]) >= float(best_primary["macro_f1"]) - tolerance
            )

        coarse_collapse = False
        if coarse_row is not None:
            coarse_collapse = _collapse_from_confusion_matrix(
                coarse_row["confusion_matrix"], subject=subject
            )

        emotion_beats_coarse = False
        if coarse_row is not None and emotion_row is not None:
            emotion_beats_coarse = bool(
                float(emotion_row["balanced_accuracy"]) > float(coarse_row["balanced_accuracy"]) + tolerance
                and float(emotion_row["macro_f1"]) > float(coarse_row["macro_f1"]) + tolerance
            )

        analysis[subject] = {
            "best_primary_target": best_primary_target,
            "coarse_exists": coarse_row is not None,
            "emotion_exists": emotion_row is not None,
            "coarse_within_tolerance": coarse_within_tolerance,
            "coarse_collapse": coarse_collapse,
            "emotion_beats_coarse": emotion_beats_coarse,
        }
    return analysis


def _final_recommendation(
    subject_analysis: dict[str, dict[str, Any]],
) -> str:
    subjects = sorted(subject_analysis.keys())
    exactly_two_subjects = len(subjects) == 2
    if not exactly_two_subjects:
        return "INCONCLUSIVE"

    coarse_exists_both = all(bool(subject_analysis[sub]["coarse_exists"]) for sub in subjects)
    coarse_within_both = all(bool(subject_analysis[sub]["coarse_within_tolerance"]) for sub in subjects)
    coarse_no_collapse_both = all(not bool(subject_analysis[sub]["coarse_collapse"]) for sub in subjects)
    if coarse_exists_both and coarse_within_both and coarse_no_collapse_both:
        return "PROVISIONALLY_FAVOR_COARSE_AFFECT"

    emotion_exists_both = all(bool(subject_analysis[sub]["emotion_exists"]) for sub in subjects)
    emotion_beats_both = all(bool(subject_analysis[sub]["emotion_beats_coarse"]) for sub in subjects)
    if emotion_exists_both and coarse_exists_both and emotion_beats_both:
        return "PROVISIONALLY_FAVOR_EMOTION"

    return "INCONCLUSIVE"


def _write_per_subject_summary_csv(
    outdir: Path,
    rows: list[dict[str, Any]],
) -> Path:
    output_rows = []
    for row in rows:
        output_rows.append(
            {
                "experiment_id": row["experiment_id"],
                "variant_id": row["variant_id"],
                "subject": row["subject"],
                "target": row["target"],
                "balanced_accuracy": row["balanced_accuracy"],
                "macro_f1": row["macro_f1"],
                "accuracy": row["accuracy"],
                "n_samples": row["n_samples"],
                "n_folds": row["n_folds"],
                "labels_json": _json_text(row["labels"]),
                "confusion_matrix_json": _json_text(row["confusion_matrix"]),
                "metrics_path": row["metrics_path"],
            }
        )

    output_columns = [
        "experiment_id",
        "variant_id",
        "subject",
        "target",
        "balanced_accuracy",
        "macro_f1",
        "accuracy",
        "n_samples",
        "n_folds",
        "labels_json",
        "confusion_matrix_json",
        "metrics_path",
    ]
    path = outdir / "e01_per_subject_summary.csv"
    pd.DataFrame(output_rows, columns=output_columns).to_csv(path, index=False)
    return path


def _write_memo(
    outdir: Path,
    reduced_rows: dict[str, dict[str, dict[str, Any]]],
    subject_analysis: dict[str, dict[str, Any]],
    *,
    recommendation: str,
) -> Path:
    lines: list[str] = []
    lines.append("# E01 Target Lock Memo")
    lines.append("")
    lines.append(
        "E01 is a method-selection experiment; `binary_valence_like` is treated as a feasibility "
        "benchmark only, and the final D01 target lock is not frozen from E01 alone."
    )
    lines.append("")
    lines.append("| subject | target | balanced_accuracy | macro_f1 | accuracy |")
    lines.append("|---|---|---:|---:|---:|")

    table_rows: list[dict[str, Any]] = []
    for subject in sorted(reduced_rows):
        for target in sorted(reduced_rows[subject]):
            row = reduced_rows[subject][target]
            table_rows.append(row)
            lines.append(
                f"| {row['subject']} | {row['target']} | "
                f"{float(row['balanced_accuracy']):.6f} | "
                f"{float(row['macro_f1']):.6f} | "
                f"{float(row['accuracy']):.6f} |"
            )

    lines.append("")
    for subject in sorted(subject_analysis):
        details = subject_analysis[subject]
        lines.append(f"## Subject {subject}")
        lines.append("")
        lines.append(f"- Best primary target: {details['best_primary_target']}")
        lines.append(f"- Coarse_affect within tolerance: {bool(details['coarse_within_tolerance'])}")
        lines.append(f"- Coarse_affect class collapse: {bool(details['coarse_collapse'])}")
        lines.append("")

    lines.append("## Final")
    lines.append("")
    lines.append(f"Final recommendation: {recommendation}")
    lines.append("D01 freeze status: DO NOT FREEZE FROM E01 ALONE")
    lines.append("Next gate: run E02 and E03 before freezing D01")
    lines.append("")

    memo_path = outdir / "e01_lock_memo.md"
    memo_path.write_text("\n".join(lines), encoding="utf-8")
    return memo_path


def _write_decision_json(
    outdir: Path,
    *,
    recommendation: str,
    tolerance: float,
    subject_analysis: dict[str, dict[str, Any]],
) -> Path:
    subjects = sorted(subject_analysis.keys())
    within_subjects = sorted(
        subject
        for subject, details in subject_analysis.items()
        if bool(details["coarse_within_tolerance"])
    )
    collapse_subjects = sorted(
        subject for subject, details in subject_analysis.items() if bool(details["coarse_collapse"])
    )

    payload = {
        "recommendation": recommendation,
        "tolerance": float(tolerance),
        "subjects_evaluated": subjects,
        "coarse_affect_within_tolerance_subjects": within_subjects,
        "coarse_affect_collapse_subjects": collapse_subjects,
        "notes": [
            "E01 is target-definition method selection, not final target freeze.",
            f"{BENCHMARK_ONLY_TARGET} is feasibility benchmark only.",
            "Run E02 and E03 before freezing D01.",
        ],
    }

    decision_path = outdir / "e01_lock_decision.json"
    decision_path.write_text(f"{json.dumps(payload, indent=2)}\n", encoding="utf-8")
    return decision_path


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    campaign_root = args.campaign_root.resolve()
    tolerance = float(args.tolerance)
    outdir = (args.outdir if args.outdir is not None else campaign_root / "e01_review").resolve()

    summary_path = _require_file(
        campaign_root / "decision_support_summary.csv",
        label="decision_support_summary.csv",
    )
    run_log_path = _require_file(campaign_root / "run_log_export.csv", label="run_log_export.csv")

    summary_df = pd.read_csv(summary_path)
    run_log_df = pd.read_csv(run_log_path)

    summary_experiment_col = _resolve_column(
        summary_df, "experiment_id", ("experiment_id", "Experiment_ID")
    )
    summary_status_col = _resolve_column(summary_df, "status", ("status", "Result_Summary"))
    _ = summary_df[
        (summary_df[summary_experiment_col].astype(str) == EXPERIMENT_ID)
        & (summary_df[summary_status_col].astype(str) == "completed")
    ]

    run_experiment_col = _resolve_column(run_log_df, "experiment_id", ("experiment_id", "Experiment_ID"))
    run_status_col = _resolve_column(run_log_df, "status", ("status", "Result_Summary"))

    completed_e01 = run_log_df[
        (run_log_df[run_experiment_col].astype(str) == EXPERIMENT_ID)
        & (run_log_df[run_status_col].astype(str) == "completed")
    ].copy()

    if completed_e01.empty:
        print("No completed E01 rows found in run_log_export.csv.", file=sys.stderr)
        return 2

    outdir.mkdir(parents=True, exist_ok=True)

    per_subject_rows = _build_per_subject_rows(completed_e01, campaign_root=campaign_root)
    reduced_rows = _reduce_to_best_per_subject_target(per_subject_rows)
    subject_analysis = _build_subject_analysis(reduced_rows, tolerance=tolerance)
    recommendation = _final_recommendation(subject_analysis)

    _write_per_subject_summary_csv(outdir, per_subject_rows)
    _write_memo(outdir, reduced_rows, subject_analysis, recommendation=recommendation)
    _write_decision_json(
        outdir,
        recommendation=recommendation,
        tolerance=tolerance,
        subject_analysis=subject_analysis,
    )

    print(f"Wrote E01 review outputs to: {outdir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
