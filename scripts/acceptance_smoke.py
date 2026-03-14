from __future__ import annotations

import argparse
import csv
import subprocess
import tempfile
from pathlib import Path
from shutil import which

from Thesis_ML.orchestration.workbook_compiler import (
    NoEnabledExecutableRowsError,
    compile_workbook_file,
)
from Thesis_ML.workbook.validation import validate_workbook


def _is_true(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _is_none_or_empty(value: object) -> bool:
    return str(value).strip() in {"", "None", "none", "[]"}


def _run(command: list[str], *, cwd: Path) -> None:
    print("$", " ".join(command))
    subprocess.run(command, cwd=cwd, check=True)


def _require_command(name: str) -> str:
    resolved = which(name)
    if resolved is None:
        raise RuntimeError(
            f"Required CLI command '{name}' was not found on PATH. "
            "Install the package or run this script under `uv run`."
        )
    return resolved


def _validate_shipped_workbook_assets(repo_root: Path) -> None:
    template_path = repo_root / "templates" / "thesis_experiment_program.xlsx"
    if not template_path.exists():
        raise FileNotFoundError(f"Missing shipped workbook template: {template_path}")

    summary = validate_workbook(template_path)
    if not _is_true(summary.get("sheet_order_ok")):
        raise RuntimeError(f"Workbook sheet order validation failed: {summary}")
    if not _is_none_or_empty(summary.get("missing_sheets")):
        raise RuntimeError(f"Workbook missing sheets: {summary.get('missing_sheets')}")
    if not _is_true(summary.get("required_named_lists_present")):
        raise RuntimeError(f"Workbook named-list validation failed: {summary}")
    if not _is_true(summary.get("workbook_schema_supported")):
        raise RuntimeError(f"Workbook schema version is not supported: {summary}")
    if not _is_true(summary.get("schema_metadata_keys_present")):
        raise RuntimeError(f"Workbook schema metadata block is incomplete: {summary}")

    try:
        template_manifest = compile_workbook_file(template_path)
    except NoEnabledExecutableRowsError:
        print("Template compile check: non-runnable template (no enabled rows).")
    else:
        if len(template_manifest.trial_specs) == 0:
            raise RuntimeError("Workbook template compiled with zero trial specs.")
        print(
            f"Template compile check: runnable template with {len(template_manifest.trial_specs)} trial specs."
        )

    workbook_assets = sorted(template_path.parent.glob("*.xlsx"))
    sample_paths = [path for path in workbook_assets if path.resolve() != template_path.resolve()]
    for sample_path in sample_paths:
        sample_manifest = compile_workbook_file(sample_path)
        if len(sample_manifest.trial_specs) == 0:
            raise RuntimeError(f"Workbook asset compiled with zero trial specs: {sample_path}")
        print(
            f"Compiled workbook sample: {sample_path} ({len(sample_manifest.trial_specs)} trial specs)."
        )

    if not sample_paths:
        print("No additional workbook sample assets detected in templates/.")


def _write_minimal_dataset_index(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["subject", "task", "modality"])
        writer.writeheader()
        writer.writerow({"subject": "sub-001", "task": "passive", "modality": "audio"})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Gold acceptance smoke for canonical Thesis_ML operator path "
            "(workbook + shipped registry dry-run)."
        )
    )
    parser.add_argument(
        "--tmp-root",
        default="",
        help="Optional temporary root directory (defaults to a new system temp directory).",
    )
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parent.parent
    _require_command("thesisml-workbook")
    _require_command("thesisml-run-decision-support")

    _validate_shipped_workbook_assets(repo_root)

    temp_root = (
        Path(args.tmp_root)
        if args.tmp_root
        else Path(tempfile.mkdtemp(prefix="thesisml-acceptance-"))
    )
    temp_root.mkdir(parents=True, exist_ok=True)

    generated_workbook = temp_root / "generated_workbook.xlsx"
    _run(["thesisml-workbook", "--output", str(generated_workbook)], cwd=repo_root)
    if not generated_workbook.exists():
        raise RuntimeError(
            f"Workbook generation failed; expected file was not created: {generated_workbook}"
        )

    generated_summary = validate_workbook(generated_workbook)
    if not _is_true(generated_summary.get("sheet_order_ok")):
        raise RuntimeError(f"Generated workbook validation failed: {generated_summary}")

    dataset_index_path = temp_root / "dataset_index.csv"
    _write_minimal_dataset_index(dataset_index_path)

    _run(
        [
            "thesisml-run-decision-support",
            "--registry",
            "configs/decision_support_registry.json",
            "--index-csv",
            str(dataset_index_path),
            "--data-root",
            str(temp_root / "Data"),
            "--cache-dir",
            str(temp_root / "cache"),
            "--output-root",
            str(temp_root / "outputs"),
            "--all",
            "--dry-run",
        ],
        cwd=repo_root,
    )

    print("Acceptance smoke passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
