from __future__ import annotations

from pathlib import Path
from typing import Any

from openpyxl import load_workbook
from openpyxl.workbook.workbook import Workbook

from Thesis_ML.orchestration.compiler import compile_registry_payload
from Thesis_ML.orchestration.contracts import CompiledStudyManifest, ReusePolicy, SectionName

_REQUIRED_SHEETS = {"Master_Experiments", "Experiment_Definitions"}

_MASTER_REQUIRED_COLUMNS = {
    "Experiment_ID",
    "Short_Title",
    "Stage",
}

_EXPERIMENT_DEFINITION_REQUIRED_COLUMNS = [
    "experiment_id",
    "enabled",
    "start_section",
    "end_section",
    "base_artifact_id",
    "target",
    "cv",
    "model",
    "subject",
    "train_subject",
    "test_subject",
    "filter_task",
    "filter_modality",
    "reuse_policy",
]

_SECTION_ORDER: tuple[str, ...] = (
    SectionName.DATASET_SELECTION.value,
    SectionName.FEATURE_CACHE_BUILD.value,
    SectionName.FEATURE_MATRIX_LOAD.value,
    SectionName.SPATIAL_VALIDATION.value,
    SectionName.MODEL_FIT.value,
    SectionName.INTERPRETABILITY.value,
    SectionName.EVALUATION.value,
)
_SECTION_TO_INDEX = {name: idx for idx, name in enumerate(_SECTION_ORDER)}
_FEATURE_MATRIX_LOAD_INDEX = _SECTION_TO_INDEX[SectionName.FEATURE_MATRIX_LOAD.value]


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _header_index_map(ws) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for idx, cell in enumerate(ws[1], start=1):
        header = _normalize_text(cell.value)
        if header:
            mapping[header] = idx
    return mapping


def _read_cell(row: tuple[Any, ...], header_map: dict[str, int], column_name: str) -> Any:
    col_idx = header_map.get(column_name)
    if col_idx is None or col_idx <= 0 or col_idx > len(row):
        return None
    return row[col_idx - 1]


def _parse_enabled(value: Any, *, row_index: int) -> bool:
    text = _normalize_text(value).lower()
    if text in {"", "no", "n", "false", "0"}:
        return False
    if text in {"yes", "y", "true", "1"}:
        return True
    raise ValueError(
        "Invalid enabled value in Experiment_Definitions row "
        f"{row_index}: '{value}'. Use Yes/No."
    )


def _validated_section(value: Any, *, field_name: str, row_index: int, default: str) -> str:
    text = _normalize_text(value) or default
    if text not in _SECTION_TO_INDEX:
        allowed = ", ".join(_SECTION_ORDER)
        raise ValueError(
            f"Invalid {field_name} in Experiment_Definitions row {row_index}: '{text}'. "
            f"Allowed values: {allowed}"
        )
    return text


def _validated_reuse_policy(value: Any, *, row_index: int) -> str:
    text = _normalize_text(value) or ReusePolicy.AUTO.value
    try:
        return ReusePolicy(text).value
    except ValueError as exc:
        allowed = ", ".join(policy.value for policy in ReusePolicy)
        raise ValueError(
            f"Invalid reuse_policy in Experiment_Definitions row {row_index}: '{text}'. "
            f"Allowed values: {allowed}"
        ) from exc


def _assert_base_artifact_usage(
    *,
    row_index: int,
    start_section: str,
    base_artifact_id: str,
    reuse_policy: str,
) -> None:
    start_idx = _SECTION_TO_INDEX[start_section]
    requires_base = start_idx >= _FEATURE_MATRIX_LOAD_INDEX

    if base_artifact_id and not requires_base:
        raise ValueError(
            "Invalid base artifact usage in Experiment_Definitions row "
            f"{row_index}: base_artifact_id is not allowed when start_section="
            f"'{start_section}'."
        )

    if reuse_policy == ReusePolicy.REQUIRE_EXPLICIT_BASE.value and requires_base and not base_artifact_id:
        raise ValueError(
            "Invalid base artifact usage in Experiment_Definitions row "
            f"{row_index}: reuse_policy='require_explicit_base' requires base_artifact_id "
            "when start_section is feature_matrix_load or later."
        )

    if reuse_policy == ReusePolicy.DISALLOW.value and requires_base:
        raise ValueError(
            "Invalid base artifact usage in Experiment_Definitions row "
            f"{row_index}: reuse_policy='disallow' is incompatible with start_section "
            "feature_matrix_load or later."
        )


def _parse_master_experiment_rows(ws) -> dict[str, dict[str, Any]]:
    header_map = _header_index_map(ws)
    missing_master = [name for name in _MASTER_REQUIRED_COLUMNS if name not in header_map]
    if missing_master:
        raise ValueError(
            "Master_Experiments is missing required columns: " + ", ".join(sorted(missing_master))
        )

    result: dict[str, dict[str, Any]] = {}
    for row in ws.iter_rows(min_row=2, values_only=True):
        experiment_id = _normalize_text(_read_cell(row, header_map, "Experiment_ID"))
        if not experiment_id:
            continue
        result[experiment_id] = {
            "experiment_id": experiment_id,
            "title": _normalize_text(_read_cell(row, header_map, "Short_Title")) or experiment_id,
            "stage": _normalize_text(_read_cell(row, header_map, "Stage")) or "Unspecified stage",
            "manipulated_factor": _normalize_text(
                _read_cell(row, header_map, "Manipulated_Factor")
            )
            or None,
            "primary_metric": _normalize_text(_read_cell(row, header_map, "Primary_Metric"))
            or "balanced_accuracy",
        }
    return result


def _parse_experiment_definitions_rows(ws) -> list[dict[str, Any]]:
    header_map = _header_index_map(ws)
    missing_columns = [
        name for name in _EXPERIMENT_DEFINITION_REQUIRED_COLUMNS if name not in header_map
    ]
    if missing_columns:
        raise ValueError(
            "Experiment_Definitions is missing required columns: "
            + ", ".join(missing_columns)
        )

    compiled_trials: list[dict[str, Any]] = []
    for row_index, row in enumerate(ws.iter_rows(min_row=2, values_only=True), start=2):
        experiment_id = _normalize_text(_read_cell(row, header_map, "experiment_id"))
        if not experiment_id:
            continue
        enabled = _parse_enabled(_read_cell(row, header_map, "enabled"), row_index=row_index)
        if not enabled:
            continue

        target = _normalize_text(_read_cell(row, header_map, "target"))
        cv = _normalize_text(_read_cell(row, header_map, "cv"))
        model = _normalize_text(_read_cell(row, header_map, "model"))
        missing_required = [name for name, value in (("target", target), ("cv", cv), ("model", model)) if not value]
        if missing_required:
            raise ValueError(
                "Experiment_Definitions row "
                f"{row_index} is missing required values: {', '.join(missing_required)}"
            )

        start_section = _validated_section(
            _read_cell(row, header_map, "start_section"),
            field_name="start_section",
            row_index=row_index,
            default=SectionName.DATASET_SELECTION.value,
        )
        end_section = _validated_section(
            _read_cell(row, header_map, "end_section"),
            field_name="end_section",
            row_index=row_index,
            default=SectionName.EVALUATION.value,
        )
        if _SECTION_TO_INDEX[start_section] > _SECTION_TO_INDEX[end_section]:
            raise ValueError(
                "Invalid section range in Experiment_Definitions row "
                f"{row_index}: start_section='{start_section}' is after end_section='{end_section}'."
            )

        base_artifact_id = _normalize_text(_read_cell(row, header_map, "base_artifact_id")) or None
        reuse_policy = _validated_reuse_policy(
            _read_cell(row, header_map, "reuse_policy"),
            row_index=row_index,
        )
        _assert_base_artifact_usage(
            row_index=row_index,
            start_section=start_section,
            base_artifact_id=base_artifact_id or "",
            reuse_policy=reuse_policy,
        )

        params: dict[str, Any] = {
            "target": target,
            "cv": cv,
            "model": model,
        }
        for key in (
            "subject",
            "train_subject",
            "test_subject",
            "filter_task",
            "filter_modality",
        ):
            value = _normalize_text(_read_cell(row, header_map, key))
            if value:
                params[key] = value

        compiled_trials.append(
            {
                "experiment_id": experiment_id,
                "template_id": f"wb_row_{row_index:03d}",
                "supported": True,
                "params": params,
                "expand": {},
                "sections": list(_SECTION_ORDER),
                "artifacts": [],
                "start_section": start_section,
                "end_section": end_section,
                "base_artifact_id": base_artifact_id,
                "reuse_policy": reuse_policy,
            }
        )
    return compiled_trials


def compile_workbook_workbook(
    workbook: Workbook,
    *,
    source_workbook_path: Path | None = None,
) -> CompiledStudyManifest:
    sheet_names = set(workbook.sheetnames)
    missing_sheets = sorted(_REQUIRED_SHEETS - sheet_names)
    if missing_sheets:
        raise ValueError("Workbook missing required sheet(s): " + ", ".join(missing_sheets))

    master_map = _parse_master_experiment_rows(workbook["Master_Experiments"])
    trial_specs = _parse_experiment_definitions_rows(workbook["Experiment_Definitions"])
    if not trial_specs:
        raise ValueError("No enabled executable rows were found in Experiment_Definitions.")

    experiment_ids = sorted({trial["experiment_id"] for trial in trial_specs})
    unknown_experiments = [experiment_id for experiment_id in experiment_ids if experiment_id not in master_map]
    if unknown_experiments:
        raise ValueError(
            "Experiment_Definitions references unknown experiment_id values: "
            + ", ".join(unknown_experiments)
        )

    experiments_payload: list[dict[str, Any]] = []
    for experiment_id in experiment_ids:
        master_row = master_map[experiment_id]
        variants = [trial for trial in trial_specs if trial["experiment_id"] == experiment_id]
        experiments_payload.append(
            {
                "experiment_id": experiment_id,
                "title": master_row["title"],
                "stage": master_row["stage"],
                "manipulated_factor": master_row["manipulated_factor"],
                "primary_metric": master_row["primary_metric"],
                "variant_templates": variants,
            }
        )

    payload = {
        "schema_version": "workbook-v1",
        "description": "Compiled from thesis_experiment_program.xlsx",
        "experiments": experiments_payload,
    }
    return compile_registry_payload(payload, source_registry_path=source_workbook_path)


def compile_workbook_file(path: Path) -> CompiledStudyManifest:
    workbook_path = Path(path)
    workbook = load_workbook(workbook_path, data_only=False)
    return compile_workbook_workbook(workbook, source_workbook_path=workbook_path)

