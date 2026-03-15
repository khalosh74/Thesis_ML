from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from Thesis_ML.data.index_dataset import build_dataset_index
from Thesis_ML.protocols.compiler import compile_protocol
from Thesis_ML.protocols.loader import load_protocol
from Thesis_ML.protocols.runner import compile_and_run_protocol


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _canonical_protocol_path() -> Path:
    return _repo_root() / "configs" / "protocols" / "thesis_canonical_v1.json"


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
def protocol_dataset(tmp_path: Path) -> dict[str, Path]:
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
        "reports_root": tmp_path / "reports" / "experiments",
    }


def test_load_canonical_protocol_validates() -> None:
    protocol = load_protocol(_canonical_protocol_path())
    assert protocol.protocol_schema_version == "thesis-protocol-v1"
    assert protocol.framework_mode == "confirmatory"
    assert protocol.scientific_contract.target == "coarse_affect"
    assert protocol.scientific_contract.primary_metric == "balanced_accuracy"
    assert {suite.suite_id for suite in protocol.official_run_suites} == {
        "primary_within_subject",
        "secondary_cross_person_transfer",
        "primary_controls",
    }


def test_load_protocol_rejects_invalid_schema_version(tmp_path: Path) -> None:
    payload = json.loads(_canonical_protocol_path().read_text(encoding="utf-8"))
    payload["protocol_schema_version"] = "thesis-protocol-v999"
    protocol_path = tmp_path / "invalid_schema_protocol.json"
    protocol_path.write_text(f"{json.dumps(payload, indent=2)}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported protocol_schema_version"):
        load_protocol(protocol_path)


def test_protocol_validation_rejects_permutation_metric_conflict_without_justification(
    tmp_path: Path,
) -> None:
    payload = json.loads(_canonical_protocol_path().read_text(encoding="utf-8"))
    payload["control_policy"]["permutation"]["metric"] = "accuracy"
    payload["control_policy"]["permutation"]["metric_conflict_justification"] = None
    protocol_path = tmp_path / "invalid_metric_conflict_protocol.json"
    protocol_path.write_text(f"{json.dumps(payload, indent=2)}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="metric conflicts with scientific_contract.primary_metric"):
        load_protocol(protocol_path)


def test_protocol_compiler_expands_primary_and_transfer_suites(
    protocol_dataset: dict[str, Path],
) -> None:
    protocol = load_protocol(_canonical_protocol_path())
    manifest = compile_protocol(
        protocol,
        index_csv=protocol_dataset["index_csv"],
        suite_ids=["primary_within_subject", "secondary_cross_person_transfer"],
    )

    within_runs = [run for run in manifest.runs if run.cv_mode == "within_subject_loso_session"]
    transfer_runs = [run for run in manifest.runs if run.cv_mode == "frozen_cross_person_transfer"]

    assert len(within_runs) == 2
    assert len(transfer_runs) == 2
    assert all(run.subject is not None for run in within_runs)
    assert all(run.train_subject is not None and run.test_subject is not None for run in transfer_runs)


def test_control_suite_expands_dummy_and_permutation_controls(
    protocol_dataset: dict[str, Path],
) -> None:
    protocol = load_protocol(_canonical_protocol_path())
    manifest = compile_protocol(
        protocol,
        index_csv=protocol_dataset["index_csv"],
        suite_ids=["primary_controls"],
    )

    models = {run.model for run in manifest.runs}
    assert models == {"ridge", "dummy"}
    assert all(run.controls.permutation_enabled for run in manifest.runs)
    assert all(run.controls.permutation_metric == "balanced_accuracy" for run in manifest.runs)


def test_protocol_runner_dry_run_emits_protocol_artifacts(
    protocol_dataset: dict[str, Path],
) -> None:
    protocol = load_protocol(_canonical_protocol_path())
    result = compile_and_run_protocol(
        protocol=protocol,
        index_csv=protocol_dataset["index_csv"],
        data_root=protocol_dataset["data_root"],
        cache_dir=protocol_dataset["cache_dir"],
        reports_root=protocol_dataset["reports_root"],
        suite_ids=["primary_within_subject"],
        dry_run=True,
    )

    assert result["n_failed"] == 0
    assert result["n_completed"] == 0
    assert result["n_planned"] > 0
    for artifact_path in result["artifact_paths"].values():
        assert Path(artifact_path).exists()

    execution_status = json.loads(
        Path(result["artifact_paths"]["execution_status"]).read_text(encoding="utf-8")
    )
    assert execution_status["framework_mode"] == "confirmatory"
    assert execution_status["dry_run"] is True
    assert all(row["status"] == "planned" for row in execution_status["runs"])


def test_protocol_run_records_metadata_in_run_artifacts(
    protocol_dataset: dict[str, Path],
) -> None:
    protocol = load_protocol(_canonical_protocol_path())
    result = compile_and_run_protocol(
        protocol=protocol,
        index_csv=protocol_dataset["index_csv"],
        data_root=protocol_dataset["data_root"],
        cache_dir=protocol_dataset["cache_dir"],
        reports_root=protocol_dataset["reports_root"],
        suite_ids=["primary_within_subject"],
        dry_run=False,
    )

    assert result["n_failed"] == 0
    completed = [row for row in result["run_results"] if row["status"] == "completed"]
    assert completed

    config_path = Path(str(completed[0]["config_path"]))
    metrics_path = Path(str(completed[0]["metrics_path"]))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    assert config["canonical_run"] is True
    assert config["framework_mode"] == "confirmatory"
    assert config["protocol_id"] == protocol.protocol_id
    assert config["protocol_version"] == protocol.protocol_version
    assert config["suite_id"] == "primary_within_subject"
    assert isinstance(config["claim_ids"], list) and config["claim_ids"]

    assert metrics["canonical_run"] is True
    assert metrics["framework_mode"] == "confirmatory"
    assert metrics["protocol_id"] == protocol.protocol_id
    assert metrics["protocol_version"] == protocol.protocol_version
    assert metrics["suite_id"] == "primary_within_subject"
    assert isinstance(metrics["claim_ids"], list) and metrics["claim_ids"]


def test_permutation_control_uses_primary_metric(
    protocol_dataset: dict[str, Path],
) -> None:
    protocol = load_protocol(_canonical_protocol_path()).model_copy(deep=True)
    protocol.control_policy.permutation.n_permutations = 2

    result = compile_and_run_protocol(
        protocol=protocol,
        index_csv=protocol_dataset["index_csv"],
        data_root=protocol_dataset["data_root"],
        cache_dir=protocol_dataset["cache_dir"],
        reports_root=protocol_dataset["reports_root"],
        suite_ids=["primary_controls"],
        dry_run=False,
    )

    assert result["n_failed"] == 0
    completed = [row for row in result["run_results"] if row["status"] == "completed"]
    assert completed

    for run_row in completed:
        metrics_path = Path(str(run_row["metrics_path"]))
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        permutation = metrics["permutation_test"]
        assert permutation["metric_name"] == "balanced_accuracy"
        assert permutation["observed_metric"] == metrics["primary_metric_value"]
        assert "permutation_metric_mean" in permutation
        assert "permutation_metric_std" in permutation


def test_protocol_level_report_index_is_emitted_for_completed_runs(
    protocol_dataset: dict[str, Path],
) -> None:
    protocol = load_protocol(_canonical_protocol_path())
    result = compile_and_run_protocol(
        protocol=protocol,
        index_csv=protocol_dataset["index_csv"],
        data_root=protocol_dataset["data_root"],
        cache_dir=protocol_dataset["cache_dir"],
        reports_root=protocol_dataset["reports_root"],
        suite_ids=["secondary_cross_person_transfer"],
        dry_run=False,
    )
    assert result["n_failed"] == 0

    report_index_path = Path(result["artifact_paths"]["report_index"])
    report_index = pd.read_csv(report_index_path)
    assert len(report_index) == 2
    assert set(report_index["suite_id"].astype(str)) == {"secondary_cross_person_transfer"}
    assert set(report_index["status"].astype(str)) == {"completed"}
