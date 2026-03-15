from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from Thesis_ML.comparisons.compiler import compile_comparison
from Thesis_ML.comparisons.loader import load_comparison_spec
from Thesis_ML.comparisons.models import ComparisonStatus
from Thesis_ML.comparisons.runner import compile_and_run_comparison
from Thesis_ML.data.index_dataset import build_dataset_index


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _comparison_spec_path() -> Path:
    return _repo_root() / "configs" / "comparisons" / "model_family_comparison_v1.json"


def _write_nifti(path: Path, data: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = nib.Nifti1Image(data.astype(np.float32), affine=np.eye(4, dtype=np.float64))
    nib.save(image, str(path))


def _create_glm_session(
    glm_dir: Path,
    labels: list[str],
    *,
    class_signal: bool,
    shape: tuple[int, int, int] = (3, 3, 3),
) -> None:
    glm_dir.mkdir(parents=True, exist_ok=True)
    mask = np.zeros(shape, dtype=np.float32)
    mask[1:, 1:, 1:] = 1.0
    _write_nifti(glm_dir / "mask.nii", mask)
    pd.Series(labels).to_csv(glm_dir / "regressor_labels.csv", index=False, header=False)

    for idx, label in enumerate(labels, start=1):
        beta = np.full(shape, fill_value=float(idx), dtype=np.float32)
        if class_signal:
            if "_anger_" in label:
                beta[1:, 1:, 1:] += 5.0
            if "_happiness_" in label:
                beta[1:, 1:, 1:] -= 5.0
        _write_nifti(glm_dir / f"beta_{idx:04d}.nii", beta)


@pytest.fixture
def comparison_dataset(tmp_path: Path) -> dict[str, Path]:
    data_root = tmp_path / "Data"
    labels = [
        "run-1_passive_anger_audio",
        "run-1_passive_happiness_audio",
    ]
    for subject in ("sub-001", "sub-002"):
        for session in ("ses-01", "ses-02"):
            _create_glm_session(
                glm_dir=data_root / subject / session / "BAS2",
                labels=labels,
                class_signal=True,
            )

    index_csv = tmp_path / "dataset_index.csv"
    build_dataset_index(data_root=data_root, out_csv=index_csv)
    return {
        "index_csv": index_csv,
        "data_root": data_root,
        "cache_dir": tmp_path / "cache",
        "reports_root": tmp_path / "reports" / "comparisons",
    }


def test_load_comparison_spec_validates() -> None:
    spec = load_comparison_spec(_comparison_spec_path())
    assert spec.comparison_schema_version == "comparison-spec-v1"
    assert spec.framework_mode == "locked_comparison"
    assert spec.scientific_contract.target == "coarse_affect"
    assert spec.scientific_contract.split_mode == "within_subject_loso_session"
    assert {variant.variant_id for variant in spec.allowed_variants} == {
        "ridge",
        "logreg",
        "linearsvc",
    }


def test_compile_comparison_expands_variants_and_subjects(
    comparison_dataset: dict[str, Path],
) -> None:
    spec = load_comparison_spec(_comparison_spec_path())
    manifest = compile_comparison(spec, index_csv=comparison_dataset["index_csv"])
    assert manifest.framework_mode == "locked_comparison"
    assert set(manifest.variant_ids) == {"ridge", "logreg", "linearsvc"}
    assert len(manifest.runs) == 6
    assert all(run.framework_mode == "locked_comparison" for run in manifest.runs)
    assert all(run.canonical_run is False for run in manifest.runs)
    assert all(run.subject in {"sub-001", "sub-002"} for run in manifest.runs)


def test_compile_comparison_rejects_unknown_variant(
    comparison_dataset: dict[str, Path],
) -> None:
    spec = load_comparison_spec(_comparison_spec_path())
    with pytest.raises(ValueError, match="Unknown comparison variant"):
        compile_comparison(
            spec,
            index_csv=comparison_dataset["index_csv"],
            variant_ids=["not_registered_variant"],
        )


def test_comparison_runner_dry_run_emits_artifacts(
    comparison_dataset: dict[str, Path],
) -> None:
    spec = load_comparison_spec(_comparison_spec_path())
    result = compile_and_run_comparison(
        comparison=spec,
        index_csv=comparison_dataset["index_csv"],
        data_root=comparison_dataset["data_root"],
        cache_dir=comparison_dataset["cache_dir"],
        reports_root=comparison_dataset["reports_root"],
        variant_ids=["ridge"],
        dry_run=True,
    )
    assert result["n_failed"] == 0
    assert result["n_completed"] == 0
    assert result["n_planned"] == 2
    for artifact_path in result["artifact_paths"].values():
        assert Path(artifact_path).exists()

    status_payload = json.loads(
        Path(result["artifact_paths"]["execution_status"]).read_text(encoding="utf-8")
    )
    assert status_payload["framework_mode"] == "locked_comparison"
    assert status_payload["dry_run"] is True
    assert all(run["status"] == "planned" for run in status_payload["runs"])


def test_comparison_runner_real_run_stamps_metadata(
    comparison_dataset: dict[str, Path],
) -> None:
    spec = load_comparison_spec(_comparison_spec_path())
    result = compile_and_run_comparison(
        comparison=spec,
        index_csv=comparison_dataset["index_csv"],
        data_root=comparison_dataset["data_root"],
        cache_dir=comparison_dataset["cache_dir"],
        reports_root=comparison_dataset["reports_root"],
        variant_ids=["ridge"],
        dry_run=False,
    )
    assert result["n_failed"] == 0
    completed = [row for row in result["run_results"] if row["status"] == "completed"]
    assert completed

    config_path = Path(str(completed[0]["config_path"]))
    metrics_path = Path(str(completed[0]["metrics_path"]))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    assert config["framework_mode"] == "locked_comparison"
    assert config["canonical_run"] is False
    assert config["comparison_id"] == spec.comparison_id
    assert config["comparison_version"] == spec.comparison_version
    assert config["comparison_variant_id"] == "ridge"

    assert metrics["framework_mode"] == "locked_comparison"
    assert metrics["canonical_run"] is False
    assert metrics["comparison_id"] == spec.comparison_id
    assert metrics["comparison_version"] == spec.comparison_version
    assert metrics["comparison_variant_id"] == "ridge"


def test_comparison_runner_rejects_draft_or_retired_status(
    comparison_dataset: dict[str, Path],
) -> None:
    spec = load_comparison_spec(_comparison_spec_path()).model_copy(deep=True)

    spec.status = ComparisonStatus.DRAFT
    with pytest.raises(ValueError, match="status is 'draft'"):
        compile_and_run_comparison(
            comparison=spec,
            index_csv=comparison_dataset["index_csv"],
            data_root=comparison_dataset["data_root"],
            cache_dir=comparison_dataset["cache_dir"],
            reports_root=comparison_dataset["reports_root"],
            dry_run=True,
        )

    spec.status = ComparisonStatus.RETIRED
    with pytest.raises(ValueError, match="status is 'retired'"):
        compile_and_run_comparison(
            comparison=spec,
            index_csv=comparison_dataset["index_csv"],
            data_root=comparison_dataset["data_root"],
            cache_dir=comparison_dataset["cache_dir"],
            reports_root=comparison_dataset["reports_root"],
            dry_run=True,
        )

