from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from Thesis_ML.data.index_dataset import build_dataset_index
from Thesis_ML.experiments.errors import OfficialContractValidationError
from Thesis_ML.protocols.compiler import compile_protocol
from Thesis_ML.protocols.loader import load_protocol
from Thesis_ML.protocols.runner import compile_and_run_protocol


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _canonical_protocol_path() -> Path:
    return _repo_root() / "configs" / "protocols" / "thesis_canonical_v1.json"


def _nested_protocol_path() -> Path:
    return _repo_root() / "configs" / "protocols" / "thesis_canonical_nested_v1.json"


def _confirmatory_freeze_protocol_path() -> Path:
    return _repo_root() / "configs" / "protocols" / "thesis_confirmatory_v1.json"


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
    assert protocol.methodology_policy.policy_name.value == "fixed_baselines_only"
    assert protocol.metric_policy.primary_metric == "balanced_accuracy"
    assert protocol.subgroup_reporting_policy.enabled is True
    assert {suite.suite_id for suite in protocol.official_run_suites} == {
        "primary_within_subject",
        "secondary_cross_person_transfer",
        "primary_controls",
    }


def test_load_confirmatory_freeze_protocol_validates_and_adapts() -> None:
    protocol = load_protocol(_confirmatory_freeze_protocol_path())
    assert protocol.protocol_schema_version == "thesis-protocol-v1"
    assert protocol.framework_mode == "confirmatory"
    assert protocol.protocol_id == "thesis_confirmatory_v1"
    assert protocol.status.value == "locked"
    assert protocol.scientific_contract.target == "coarse_affect"
    assert protocol.scientific_contract.primary_metric == "balanced_accuracy"
    assert protocol.control_policy.dummy_baseline.enabled is True
    assert protocol.control_policy.permutation.enabled is True


def test_confirmatory_freeze_preflight_rejects_invalid_schema_payload(
    tmp_path: Path,
) -> None:
    payload = json.loads(_confirmatory_freeze_protocol_path().read_text(encoding="utf-8"))
    payload["target"]["mapping_hash"] = "NOT_A_SHA"
    protocol_path = tmp_path / "invalid_confirmatory_freeze.json"
    protocol_path.write_text(f"{json.dumps(payload, indent=2)}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="preflight schema validation failed"):
        load_protocol(protocol_path)


def test_confirmatory_freeze_preflight_rejects_mapping_hash_mismatch(
    tmp_path: Path,
) -> None:
    payload = json.loads(_confirmatory_freeze_protocol_path().read_text(encoding="utf-8"))
    payload["target"]["mapping_hash"] = "0" * 64
    protocol_path = tmp_path / "invalid_confirmatory_freeze_hash.json"
    protocol_path.write_text(f"{json.dumps(payload, indent=2)}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="target.mapping_hash mismatch"):
        load_protocol(protocol_path)


def test_confirmatory_freeze_protocol_dry_run_executes_with_locked_gates(
    protocol_dataset: dict[str, Path],
) -> None:
    protocol = load_protocol(_confirmatory_freeze_protocol_path())
    suite_id = protocol.official_run_suites[0].suite_id
    result = compile_and_run_protocol(
        protocol=protocol,
        index_csv=protocol_dataset["index_csv"],
        data_root=protocol_dataset["data_root"],
        cache_dir=protocol_dataset["cache_dir"],
        reports_root=protocol_dataset["reports_root"],
        suite_ids=[suite_id],
        dry_run=True,
    )

    assert result["n_failed"] == 0
    assert result["n_planned"] > 0
    deviation_log = json.loads(
        Path(result["artifact_paths"]["deviation_log"]).read_text(encoding="utf-8")
    )
    suite_summary = json.loads(
        Path(result["artifact_paths"]["suite_summary"]).read_text(encoding="utf-8")
    )
    assert deviation_log["confirmatory_status"] == "confirmatory"
    assert deviation_log["science_critical_deviation_detected"] is False
    assert deviation_log["deviations"][0]["status"] == "no_deviation"
    contract = suite_summary["confirmatory_reporting_contract"]
    assert contract["protocol_id"] == "thesis_confirmatory_v1"
    assert contract["target_mapping_version"] == "affect_mapping_v1"
    assert isinstance(contract["target_mapping_hash"], str) and contract["target_mapping_hash"]
    assert contract["primary_split"] == "within_subject_loso_session"
    assert contract["primary_metric"] == "balanced_accuracy"
    assert contract["model_family"] == "ridge"
    assert isinstance(contract["controls_status"], dict)
    assert isinstance(contract["multiplicity_policy"], dict)
    assert contract["multiplicity_policy"]["primary_hypotheses"] == 1
    assert contract["multiplicity_policy"]["primary_alpha"] == 0.05
    assert contract["multiplicity_policy"]["secondary_policy"] == "descriptive_only"
    assert contract["multiplicity_policy"]["exploratory_claims_allowed"] is False
    assert isinstance(contract["interpretation_limits"], dict)
    assert contract["subgroup_evidence_policy"]["evidence_role"] == "descriptive_only"
    assert (
        contract["subgroup_evidence_policy"]["primary_evidence_substitution_allowed"]
        is False
    )
    assert contract["deviations_from_protocol"]["controls_valid_for_confirmatory"] is True


def test_confirmatory_freeze_execution_rejects_unlocked_analysis_status(
    protocol_dataset: dict[str, Path],
) -> None:
    protocol = load_protocol(_confirmatory_freeze_protocol_path()).model_copy(deep=True)
    assert isinstance(protocol.confirmatory_lock, dict)
    protocol.confirmatory_lock["analysis_status"] = "draft"

    with pytest.raises(ValueError, match="analysis_status='locked'"):
        compile_and_run_protocol(
            protocol=protocol,
            index_csv=protocol_dataset["index_csv"],
            data_root=protocol_dataset["data_root"],
            cache_dir=protocol_dataset["cache_dir"],
            reports_root=protocol_dataset["reports_root"],
            suite_ids=[protocol.official_run_suites[0].suite_id],
            dry_run=True,
        )


def test_confirmatory_freeze_execution_rejects_missing_dummy_controls(
    protocol_dataset: dict[str, Path],
) -> None:
    protocol = load_protocol(_confirmatory_freeze_protocol_path()).model_copy(deep=True)
    protocol.control_policy.dummy_baseline.enabled = False
    protocol.control_policy.dummy_baseline.suites = []

    with pytest.raises(ValueError, match="dummy baseline is required"):
        compile_and_run_protocol(
            protocol=protocol,
            index_csv=protocol_dataset["index_csv"],
            data_root=protocol_dataset["data_root"],
            cache_dir=protocol_dataset["cache_dir"],
            reports_root=protocol_dataset["reports_root"],
            suite_ids=[protocol.official_run_suites[0].suite_id],
            dry_run=True,
        )


def test_confirmatory_freeze_execution_rejects_permutation_minimum_drift(
    protocol_dataset: dict[str, Path],
) -> None:
    protocol = load_protocol(_confirmatory_freeze_protocol_path()).model_copy(deep=True)
    protocol.control_policy.permutation.n_permutations = 10

    with pytest.raises(ValueError, match="n_permutations >= 1000"):
        compile_and_run_protocol(
            protocol=protocol,
            index_csv=protocol_dataset["index_csv"],
            data_root=protocol_dataset["data_root"],
            cache_dir=protocol_dataset["cache_dir"],
            reports_root=protocol_dataset["reports_root"],
            suite_ids=[protocol.official_run_suites[0].suite_id],
            dry_run=True,
        )


def test_confirmatory_freeze_execution_records_downgrade_on_science_critical_deviation(
    protocol_dataset: dict[str, Path],
) -> None:
    protocol = load_protocol(_confirmatory_freeze_protocol_path()).model_copy(deep=True)
    assert isinstance(protocol.confirmatory_lock, dict)
    protocol.confirmatory_lock["target_source_column"] = "emotion_missing"

    with pytest.raises(ValueError, match="Protocol artifact verification failed"):
        compile_and_run_protocol(
            protocol=protocol,
            index_csv=protocol_dataset["index_csv"],
            data_root=protocol_dataset["data_root"],
            cache_dir=protocol_dataset["cache_dir"],
            reports_root=protocol_dataset["reports_root"],
            suite_ids=[protocol.official_run_suites[0].suite_id],
            dry_run=False,
        )

    output_dir = (
        protocol_dataset["reports_root"]
        / "protocol_runs"
        / f"{protocol.protocol_id}__{protocol.protocol_version}"
    )

    deviation_log = json.loads(
        (output_dir / "deviation_log.json").read_text(encoding="utf-8")
    )
    execution_status = json.loads(
        (output_dir / "execution_status.json").read_text(encoding="utf-8")
    )
    suite_summary = json.loads((output_dir / "suite_summary.json").read_text(encoding="utf-8"))
    assert deviation_log["science_critical_deviation_detected"] is True
    assert deviation_log["confirmatory_status"] == "downgraded"
    assert any(
        bool(row.get("science_critical", False)) for row in deviation_log["deviations"]
    )
    assert execution_status["confirmatory_status"] == "downgraded"
    assert (
        suite_summary["confirmatory_reporting_contract"]["deviations_from_protocol"][
            "controls_valid_for_confirmatory"
        ]
        is False
    )


def test_confirmatory_freeze_subgroup_rows_mark_insufficient_data(
    protocol_dataset: dict[str, Path],
) -> None:
    protocol = load_protocol(_confirmatory_freeze_protocol_path()).model_copy(deep=True)
    assert isinstance(protocol.confirmatory_lock, dict)
    protocol.control_policy.permutation.n_permutations = 1
    protocol.confirmatory_lock["minimum_permutations"] = 1

    result = compile_and_run_protocol(
        protocol=protocol,
        index_csv=protocol_dataset["index_csv"],
        data_root=protocol_dataset["data_root"],
        cache_dir=protocol_dataset["cache_dir"],
        reports_root=protocol_dataset["reports_root"],
        suite_ids=[protocol.official_run_suites[0].suite_id],
        dry_run=False,
    )
    assert result["n_failed"] == 0
    completed = [row for row in result["run_results"] if row["status"] == "completed"]
    assert completed
    config = json.loads(Path(str(completed[0]["config_path"])).read_text(encoding="utf-8"))
    subgroup_payload = json.loads(
        Path(str(config["subgroup_metrics_json_path"])).read_text(encoding="utf-8")
    )
    assert subgroup_payload["confirmatory_guardrails_enabled"] is True
    assert subgroup_payload["evidence_role"] == "descriptive_only"
    assert subgroup_payload["primary_evidence_substitution_allowed"] is False
    assert subgroup_payload["n_rows"] > 0
    assert all(row["status"] == "insufficient_data" for row in subgroup_payload["rows"])
    assert all(row["interpretable"] is False for row in subgroup_payload["rows"])
    assert all(row["balanced_accuracy"] is None for row in subgroup_payload["rows"])


def test_load_protocol_rejects_invalid_schema_version(tmp_path: Path) -> None:
    payload = json.loads(_canonical_protocol_path().read_text(encoding="utf-8"))
    payload["protocol_schema_version"] = "thesis-protocol-v999"
    protocol_path = tmp_path / "invalid_schema_protocol.json"
    protocol_path.write_text(f"{json.dumps(payload, indent=2)}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported protocol_schema_version"):
        load_protocol(protocol_path)


def test_protocol_requires_explicit_methodology_policy(tmp_path: Path) -> None:
    payload = json.loads(_canonical_protocol_path().read_text(encoding="utf-8"))
    payload.pop("methodology_policy", None)
    protocol_path = tmp_path / "missing_methodology_protocol.json"
    protocol_path.write_text(f"{json.dumps(payload, indent=2)}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="methodology_policy"):
        load_protocol(protocol_path)


def test_protocol_validation_rejects_permutation_metric_conflict_without_justification(
    tmp_path: Path,
) -> None:
    payload = json.loads(_canonical_protocol_path().read_text(encoding="utf-8"))
    payload["control_policy"]["permutation"]["metric"] = "accuracy"
    protocol_path = tmp_path / "invalid_metric_conflict_protocol.json"
    protocol_path.write_text(f"{json.dumps(payload, indent=2)}\n", encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="control_policy.permutation.metric must match metric_policy.primary_metric",
    ):
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
    assert "repeated_run_metrics" in result["artifact_paths"]
    assert "repeated_run_summary" in result["artifact_paths"]
    assert "confidence_intervals" in result["artifact_paths"]
    assert "metric_intervals" in result["artifact_paths"]

    execution_status = json.loads(
        Path(result["artifact_paths"]["execution_status"]).read_text(encoding="utf-8")
    )
    suite_summary = json.loads(
        Path(result["artifact_paths"]["suite_summary"]).read_text(encoding="utf-8")
    )
    manifest_payload = json.loads(
        Path(result["artifact_paths"]["compiled_protocol_manifest"]).read_text(encoding="utf-8")
    )
    assert execution_status["framework_mode"] == "confirmatory"
    assert execution_status["dry_run"] is True
    assert all(row["status"] == "planned" for row in execution_status["runs"])
    assert suite_summary["metric_policy_effective"]["primary_metric"] == "balanced_accuracy"
    assert suite_summary["metric_policy_effective"]["decision_metric"] == "balanced_accuracy"
    assert isinstance(suite_summary["required_evidence_status"], dict)
    assert manifest_payload["metric_policy_effective"]["primary_metric"] == "balanced_accuracy"
    assert manifest_payload["metric_policy_effective"]["decision_metric"] == "balanced_accuracy"


def test_protocol_runner_surfaces_structured_failure_metadata(
    protocol_dataset: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = load_protocol(_canonical_protocol_path())

    def _failing_run_experiment(**_: object) -> dict[str, object]:
        raise OfficialContractValidationError(
            "synthetic preflight contract failure",
            details={"reason": "unit_test"},
        )

    monkeypatch.setattr("Thesis_ML.protocols.runner.run_experiment", _failing_run_experiment)
    result = compile_and_run_protocol(
        protocol=protocol,
        index_csv=protocol_dataset["index_csv"],
        data_root=protocol_dataset["data_root"],
        cache_dir=protocol_dataset["cache_dir"],
        reports_root=protocol_dataset["reports_root"],
        suite_ids=["primary_within_subject"],
        dry_run=False,
    )

    assert result["n_failed"] > 0
    failed = [row for row in result["run_results"] if row["status"] == "failed"]
    assert failed
    first = failed[0]
    assert first["error"] == "synthetic preflight contract failure"
    assert first["error_code"] == "official_contract_validation_error"
    assert first["error_type"] == "OfficialContractValidationError"
    assert first["failure_stage"] == "preflight_validation"
    assert first["error_details"] == {"reason": "unit_test"}


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
    assert config["methodology_policy_name"] == "fixed_baselines_only"
    assert config["subgroup_reporting_enabled"] is True
    assert config["metric_policy_effective"]["primary_metric"] == "balanced_accuracy"
    assert config["metric_policy_effective"]["decision_metric"] == "balanced_accuracy"
    assert config["metric_policy_effective"]["tuning_metric"] == "balanced_accuracy"
    assert config["metric_policy_effective"]["permutation_metric"] == "balanced_accuracy"
    assert config["metric_policy_effective"]["higher_is_better"] is True
    assert config["evidence_run_role"] == "primary"
    assert config["repeat_id"] == 1
    assert config["repeat_count"] == 1
    assert config["base_run_id"]
    assert isinstance(config["evidence_policy_effective"], dict)
    assert config["evidence_policy_effective"]["confidence_intervals"]["method"] == (
        "grouped_bootstrap_percentile"
    )
    assert isinstance(config["claim_ids"], list) and config["claim_ids"]

    assert metrics["canonical_run"] is True
    assert metrics["framework_mode"] == "confirmatory"
    assert metrics["protocol_id"] == protocol.protocol_id
    assert metrics["protocol_version"] == protocol.protocol_version
    assert metrics["suite_id"] == "primary_within_subject"
    assert metrics["methodology_policy_name"] == "fixed_baselines_only"
    assert "subgroup_reporting" in metrics
    assert metrics["decision_metric_name"] == "balanced_accuracy"
    assert metrics["tuning_metric_name"] == "balanced_accuracy"
    assert metrics["permutation_metric_name"] == "balanced_accuracy"
    assert metrics["metric_policy_effective"]["primary_metric"] == "balanced_accuracy"
    assert metrics["metric_policy_effective"]["decision_metric"] == "balanced_accuracy"
    assert metrics["metric_policy_effective"]["tuning_metric"] == "balanced_accuracy"
    assert metrics["metric_policy_effective"]["permutation_metric"] == "balanced_accuracy"
    assert metrics["metric_policy_effective"]["higher_is_better"] is True
    assert metrics["evidence_run_role"] == "primary"
    assert metrics["repeat_id"] == 1
    assert metrics["repeat_count"] == 1
    assert metrics["base_run_id"]
    assert isinstance(metrics["evidence_policy_effective"], dict)
    assert metrics["calibration"]["status"] in {"performed", "not_applicable", "failed"}
    assert isinstance(metrics["claim_ids"], list) and metrics["claim_ids"]
    assert Path(str(config["subgroup_metrics_json_path"])).exists()
    assert Path(str(config["subgroup_metrics_csv_path"])).exists()
    assert Path(str(config["tuning_summary_path"])).exists()
    assert Path(str(config["tuning_best_params_path"])).exists()
    assert Path(str(config["calibration_summary_path"])).exists()
    assert Path(str(config["calibration_table_path"])).exists()


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
        assert "observed_score" in permutation
        assert "p_value" in permutation
        assert permutation["minimum_required"] >= 0
        assert isinstance(permutation["meets_minimum"], bool)
        assert isinstance(permutation["passes_threshold"], bool)
        assert permutation["interpretation_status"] in {
            "passes_threshold",
            "fails_threshold",
        }


def test_protocol_primary_metric_change_propagates_to_official_metric_policy(
    protocol_dataset: dict[str, Path],
) -> None:
    protocol = load_protocol(_canonical_protocol_path()).model_copy(deep=True)
    protocol.scientific_contract.primary_metric = "macro_f1"
    protocol.scientific_contract.secondary_metrics = ["balanced_accuracy", "accuracy"]
    protocol.metric_policy.primary_metric = "macro_f1"
    protocol.metric_policy.secondary_metrics = ["balanced_accuracy", "accuracy"]
    protocol.control_policy.permutation.metric = "macro_f1"
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

    metrics_path = Path(str(completed[0]["metrics_path"]))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["primary_metric_name"] == "macro_f1"
    assert metrics["decision_metric_name"] == "macro_f1"
    assert metrics["tuning_metric_name"] == "macro_f1"
    assert metrics["permutation_metric_name"] == "macro_f1"
    assert metrics["metric_policy_effective"]["primary_metric"] == "macro_f1"
    assert metrics["metric_policy_effective"]["decision_metric"] == "macro_f1"
    assert metrics["metric_policy_effective"]["tuning_metric"] == "macro_f1"
    assert metrics["metric_policy_effective"]["permutation_metric"] == "macro_f1"
    assert metrics["permutation_test"]["metric_name"] == "macro_f1"


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


def test_nested_protocol_supports_grouped_nested_methodology(
    protocol_dataset: dict[str, Path],
) -> None:
    protocol = load_protocol(_nested_protocol_path())
    assert protocol.methodology_policy.policy_name.value == "grouped_nested_tuning"
    assert protocol.model_policy.selection_strategy.value == "nested_tuned"
    assert protocol.model_policy.tuning_enabled is True

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
    assert result["n_planned"] > 0


def test_nested_protocol_compiler_expands_repeats_and_untuned_ablation(
    protocol_dataset: dict[str, Path],
) -> None:
    protocol = load_protocol(_nested_protocol_path()).model_copy(deep=True)
    protocol.evidence_policy.repeat_evaluation.repeat_count = 2
    protocol.evidence_policy.repeat_evaluation.seed_stride = 23
    manifest = compile_protocol(
        protocol,
        index_csv=protocol_dataset["index_csv"],
        suite_ids=["primary_within_subject"],
    )

    primary_runs = [run for run in manifest.runs if run.evidence_run_role.value == "primary"]
    untuned_runs = [
        run for run in manifest.runs if run.evidence_run_role.value == "untuned_baseline"
    ]
    assert len(primary_runs) == 12
    assert len(untuned_runs) == 12
    assert {run.repeat_id for run in primary_runs} == {1, 2}
    assert {run.repeat_count for run in primary_runs} == {2}
    assert all(run.tuning_enabled is True for run in primary_runs)
    assert all(run.tuning_enabled is False for run in untuned_runs)
    assert all(run.run_id.endswith("__untuned") for run in untuned_runs)
    for primary in primary_runs:
        expected_untuned = f"{primary.run_id}__untuned"
        assert any(run.run_id == expected_untuned for run in untuned_runs)
