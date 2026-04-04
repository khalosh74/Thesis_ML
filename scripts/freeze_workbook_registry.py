from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from Thesis_ML.orchestration.workbook_compiler import compile_workbook_file

LEGACY_REVISED_WORKBOOK_PATH = Path("templates/thesis_experiment_program_revised.xlsx")
CANONICAL_STUDY_WORKBOOK_PATH = Path(
    "workbooks/thesis_program_instances/thesis_experiment_program_revised_v1.xlsx"
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Freeze a workbook into an execution-ready decision-support registry JSON."
    )
    parser.add_argument(
        "--workbook",
        type=Path,
        required=True,
        help=(
            "Workbook path to compile. Canonical thesis study instance path: "
            "workbooks/thesis_program_instances/thesis_experiment_program_revised_v1.xlsx"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output registry JSON path.",
    )
    return parser


def _resolve_workbook_path(path: Path, *, cwd: Path) -> Path:
    raw_path = Path(path)
    candidate = (cwd / raw_path).resolve() if not raw_path.is_absolute() else raw_path.resolve()
    if candidate.exists():
        return candidate

    normalized_raw = raw_path.as_posix()
    legacy_tail = LEGACY_REVISED_WORKBOOK_PATH.as_posix()
    if normalized_raw == legacy_tail or normalized_raw.endswith(f"/{legacy_tail}"):
        canonical = (cwd / CANONICAL_STUDY_WORKBOOK_PATH).resolve()
        if canonical.exists():
            print(
                (
                    "[deprecation] templates/thesis_experiment_program_revised.xlsx has moved to "
                    "workbooks/thesis_program_instances/thesis_experiment_program_revised_v1.xlsx."
                ),
                file=sys.stderr,
            )
            return canonical
    return candidate


def _manifest_to_registry_payload(manifest: Any, *, source_workbook: Path) -> dict[str, Any]:
    return {
        "schema_version": str(manifest.schema_version),
        "description": (
            "Frozen execution registry compiled from workbook "
            f"{source_workbook.name}."
        ),
        "experiments": [exp.model_dump(mode="json") for exp in manifest.experiments],
        "search_spaces": [space.model_dump(mode="json") for space in manifest.search_spaces],
        "study_designs": [study.model_dump(mode="json") for study in manifest.study_designs],
        "study_rigor_checklists": [
            checklist.model_dump(mode="json") for checklist in manifest.study_rigor_checklists
        ],
        "analysis_plans": [plan.model_dump(mode="json") for plan in manifest.analysis_plans],
        "study_reviews": [review.model_dump(mode="json") for review in manifest.study_reviews],
        "generated_design_matrix": [
            cell.model_dump(mode="json") for cell in manifest.generated_design_matrix
        ],
        "effect_summaries": [summary.model_dump(mode="json") for summary in manifest.effect_summaries],
        "validation_warnings": list(manifest.validation_warnings),
    }


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    workbook_path = _resolve_workbook_path(Path(args.workbook), cwd=Path.cwd())
    output_path = Path(args.output).resolve()
    manifest = compile_workbook_file(workbook_path)
    payload = _manifest_to_registry_payload(manifest, source_workbook=workbook_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(f"{json.dumps(payload, indent=2)}\n", encoding="utf-8")

    print(json.dumps({"workbook": str(workbook_path), "output": str(output_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
