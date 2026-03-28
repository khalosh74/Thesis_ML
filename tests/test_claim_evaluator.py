import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from Thesis_ML.protocols.claim_evaluator import evaluate_claim_outcomes
from Thesis_ML.protocols.loader import load_protocol
from Thesis_ML.protocols.models import ClaimCategory, ClaimRole, ProtocolRunResult

_PRIMARY_METRIC = "balanced_accuracy"
_GROUPED_NESTED = "grouped_nested_tuning"
_SEARCH_SPACE_ID = "official-linear-grouped-nested-v2"
_SEARCH_SPACE_VERSION = "2.0.0"


@dataclass
class _RunControlsStub:
    dummy_baseline_run: bool = False


@dataclass
class _RunSpecStub:
    run_id: str
    suite_id: str
    model: str
    cv_mode: str = "within_subject_loso_session"
    subject: str | None = "sub-001"
    train_subject: str | None = None
    test_subject: str | None = None
    repeat_id: int = 1
    repeat_count: int = 1
    evidence_run_role: str = "primary"
    primary_metric: str = _PRIMARY_METRIC
    methodology_policy_name: str = _GROUPED_NESTED
    tuning_enabled: bool = True
    tuning_search_space_id: str | None = _SEARCH_SPACE_ID
    tuning_search_space_version: str | None = _SEARCH_SPACE_VERSION
    tuning_inner_cv_scheme: str | None = "grouped_leave_one_group_out"
    tuning_inner_group_field: str | None = "session"
    controls: _RunControlsStub = field(default_factory=_RunControlsStub)


@dataclass
class _ClaimManifestStub:
    claim_to_run_map: dict[str, list[str]]
    runs: list[_RunSpecStub]


def _write_json(path: Path, payload: dict[str, Any]) -> str:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def _reporting_contract(
    *,
    confirmatory_valid: bool = True,
    controls_valid: bool = True,
    science_critical: bool = False,
) -> dict[str, object]:
    return {
        "confirmatory_valid": bool(confirmatory_valid),
        "controls_status": {"controls_valid_for_confirmatory": bool(controls_valid)},
        "deviations_from_protocol": {
            "science_critical_deviation_detected": bool(science_critical)
        },
    }


def _load_protocol_and_claims() -> tuple[object, object, object]:
    protocol = load_protocol(Path("configs/protocols/thesis_canonical_nested_v2.json"))
    primary_claim = next(
        claim
        for claim in protocol.claims
        if claim.claim_id == protocol.success_criteria.primary_claim_id
    )
    control_claim = next(
        claim
        for claim in protocol.claims
        if (
            claim.role == ClaimRole.SUPPORTING
            and claim.category == ClaimCategory.CONTROL_EVIDENCE
            and claim.estimand_scope == primary_claim.estimand_scope
        )
    )
    return protocol, primary_claim, control_claim


def _write_run_payloads(
    *,
    tmp_path: Path,
    stem: str,
    primary_metric_value: float,
    primary_metric_name: str = _PRIMARY_METRIC,
    cv_mode: str = "within_subject_loso_session",
    methodology_policy_name: str = _GROUPED_NESTED,
    tuning_enabled: bool = True,
    include_tuning_metadata: bool = True,
    permutation_test: dict[str, Any] | None = None,
) -> tuple[str, str]:
    config_payload: dict[str, Any] = {
        "primary_metric_name": primary_metric_name,
        "metric_policy_effective": {"primary_metric": primary_metric_name},
        "cv": cv_mode,
        "methodology_policy_name": methodology_policy_name,
        "tuning_enabled": bool(tuning_enabled),
    }
    metrics_payload: dict[str, Any] = {
        "primary_metric_name": primary_metric_name,
        "metric_policy_effective": {"primary_metric": primary_metric_name},
        "methodology_policy_name": methodology_policy_name,
        "tuning_enabled": bool(tuning_enabled),
        "primary_metric_value": float(primary_metric_value),
    }
    if include_tuning_metadata:
        tuning_payload = {
            "tuning_search_space_id": _SEARCH_SPACE_ID,
            "tuning_search_space_version": _SEARCH_SPACE_VERSION,
            "tuning_inner_cv_scheme": "grouped_leave_one_group_out",
            "tuning_inner_group_field": "session",
        }
        config_payload.update(tuning_payload)
        metrics_payload.update(tuning_payload)
    if permutation_test is not None:
        metrics_payload["permutation_test"] = dict(permutation_test)
    config_path = _write_json(tmp_path / f"{stem}_config.json", config_payload)
    metrics_path = _write_json(tmp_path / f"{stem}_metrics.json", metrics_payload)
    return config_path, metrics_path


def _run_result(
    run_id: str,
    suite_id: str,
    *,
    config_path: str,
    metrics_path: str,
) -> ProtocolRunResult:
    return ProtocolRunResult.model_validate(
        {
            "run_id": run_id,
            "suite_id": suite_id,
            "status": "success",
            "config_path": config_path,
            "metrics_path": metrics_path,
        }
    )


def _primary_entry(payload: dict[str, Any]) -> dict[str, Any]:
    return next(entry for entry in payload["claims"] if entry["claim_id"] == payload["primary_claim_id"])


def _manifest(
    protocol: object,
    primary_claim: object,
    control_claim: object,
    runs: list[_RunSpecStub],
    *,
    extra_claim_map: dict[str, list[str]] | None = None,
) -> _ClaimManifestStub:
    primary_run_ids = [run.run_id for run in runs if run.suite_id == primary_claim.suite_ids[0]]
    control_run_ids = [run.run_id for run in runs if run.suite_id == control_claim.suite_ids[0]]
    claim_to_run_map = {
        primary_claim.claim_id: primary_run_ids,
        control_claim.claim_id: control_run_ids,
    }
    for claim in protocol.claims:
        claim_to_run_map.setdefault(claim.claim_id, [])
    if extra_claim_map:
        for claim_id, mapped_runs in extra_claim_map.items():
            claim_to_run_map[claim_id] = list(mapped_runs)
    return _ClaimManifestStub(claim_to_run_map=claim_to_run_map, runs=runs)


def _passing_permutation() -> dict[str, Any]:
    return {"p_value": 0.01, "passes_threshold": True}


def _failing_permutation() -> dict[str, Any]:
    return {"p_value": 0.5, "passes_threshold": False}


def test_supported_when_all_required_conditions_pass(tmp_path: Path):
    protocol, primary_claim, control_claim = _load_protocol_and_claims()
    primary_run_id = "primary_sub001_r1"
    control_model_run_id = "control_ridge_sub001_r1"
    control_dummy_run_id = "control_dummy_sub001_r1"
    runs = [
        _RunSpecStub(run_id=primary_run_id, suite_id=primary_claim.suite_ids[0], model="ridge"),
        _RunSpecStub(
            run_id=control_model_run_id,
            suite_id=control_claim.suite_ids[0],
            model="ridge",
        ),
        _RunSpecStub(
            run_id=control_dummy_run_id,
            suite_id=control_claim.suite_ids[0],
            model="dummy",
            controls=_RunControlsStub(dummy_baseline_run=True),
            tuning_enabled=False,
            tuning_search_space_id=None,
            tuning_search_space_version=None,
            tuning_inner_cv_scheme=None,
            tuning_inner_group_field=None,
        ),
    ]
    manifest = _manifest(protocol, primary_claim, control_claim, runs)
    primary_config_path, primary_metrics_path = _write_run_payloads(
        tmp_path=tmp_path,
        stem="primary_pass",
        primary_metric_value=0.72,
    )
    control_config_path, control_metrics_path = _write_run_payloads(
        tmp_path=tmp_path,
        stem="control_model_pass",
        primary_metric_value=0.72,
        permutation_test=_passing_permutation(),
    )
    dummy_config_path, dummy_metrics_path = _write_run_payloads(
        tmp_path=tmp_path,
        stem="control_dummy_pass",
        primary_metric_value=0.41,
        methodology_policy_name="fixed_baselines_only",
        tuning_enabled=False,
        include_tuning_metadata=False,
    )

    payload = evaluate_claim_outcomes(
        protocol=protocol,
        compiled_manifest=manifest,
        run_results=[
            _run_result(
                primary_run_id,
                primary_claim.suite_ids[0],
                config_path=primary_config_path,
                metrics_path=primary_metrics_path,
            ),
            _run_result(
                control_model_run_id,
                control_claim.suite_ids[0],
                config_path=control_config_path,
                metrics_path=control_metrics_path,
            ),
            _run_result(
                control_dummy_run_id,
                control_claim.suite_ids[0],
                config_path=dummy_config_path,
                metrics_path=dummy_metrics_path,
            ),
        ],
        reporting_contract=_reporting_contract(),
    )

    entry = _primary_entry(payload)
    assert payload["primary_claim_verdict"] == "supported"
    assert entry["all_required_conditions_passed"] is True
    assert entry["strict_gate_summary"]["failed_condition_names"] == []


def test_invalid_when_primary_metric_mismatches_locked_protocol(tmp_path: Path):
    protocol, primary_claim, control_claim = _load_protocol_and_claims()
    primary_run_id = "primary_metric_mismatch"
    control_model_run_id = "control_metric_mismatch"
    control_dummy_run_id = "dummy_metric_mismatch"
    runs = [
        _RunSpecStub(run_id=primary_run_id, suite_id=primary_claim.suite_ids[0], model="ridge"),
        _RunSpecStub(run_id=control_model_run_id, suite_id=control_claim.suite_ids[0], model="ridge"),
        _RunSpecStub(
            run_id=control_dummy_run_id,
            suite_id=control_claim.suite_ids[0],
            model="dummy",
            controls=_RunControlsStub(dummy_baseline_run=True),
        ),
    ]
    manifest = _manifest(protocol, primary_claim, control_claim, runs)
    primary_config_path, primary_metrics_path = _write_run_payloads(
        tmp_path=tmp_path,
        stem="primary_metric_mismatch",
        primary_metric_value=0.70,
        primary_metric_name="accuracy",
    )
    control_config_path, control_metrics_path = _write_run_payloads(
        tmp_path=tmp_path,
        stem="control_metric_mismatch",
        primary_metric_value=0.70,
        permutation_test=_passing_permutation(),
    )
    dummy_config_path, dummy_metrics_path = _write_run_payloads(
        tmp_path=tmp_path,
        stem="dummy_metric_mismatch",
        primary_metric_value=0.20,
        methodology_policy_name="fixed_baselines_only",
        tuning_enabled=False,
        include_tuning_metadata=False,
    )

    payload = evaluate_claim_outcomes(
        protocol=protocol,
        compiled_manifest=manifest,
        run_results=[
            _run_result(
                primary_run_id,
                primary_claim.suite_ids[0],
                config_path=primary_config_path,
                metrics_path=primary_metrics_path,
            ),
            _run_result(
                control_model_run_id,
                control_claim.suite_ids[0],
                config_path=control_config_path,
                metrics_path=control_metrics_path,
            ),
            _run_result(
                control_dummy_run_id,
                control_claim.suite_ids[0],
                config_path=dummy_config_path,
                metrics_path=dummy_metrics_path,
            ),
        ],
        reporting_contract=_reporting_contract(),
    )

    entry = _primary_entry(payload)
    assert payload["primary_claim_verdict"] == "invalid"
    assert entry["reason"] == "primary_metric_mismatch"
    assert entry["strict_gate_summary"]["metric_mismatch_run_ids"] == [primary_run_id]


def test_invalid_when_cv_or_methodology_mismatches_locked_protocol(tmp_path: Path):
    protocol, primary_claim, control_claim = _load_protocol_and_claims()
    primary_run_id = "primary_method_mismatch"
    control_model_run_id = "control_method_mismatch"
    control_dummy_run_id = "dummy_method_mismatch"
    runs = [
        _RunSpecStub(run_id=primary_run_id, suite_id=primary_claim.suite_ids[0], model="ridge"),
        _RunSpecStub(run_id=control_model_run_id, suite_id=control_claim.suite_ids[0], model="ridge"),
        _RunSpecStub(
            run_id=control_dummy_run_id,
            suite_id=control_claim.suite_ids[0],
            model="dummy",
            controls=_RunControlsStub(dummy_baseline_run=True),
        ),
    ]
    manifest = _manifest(protocol, primary_claim, control_claim, runs)
    primary_config_path, primary_metrics_path = _write_run_payloads(
        tmp_path=tmp_path,
        stem="primary_method_mismatch",
        primary_metric_value=0.70,
        methodology_policy_name="fixed_baselines_only",
    )
    control_config_path, control_metrics_path = _write_run_payloads(
        tmp_path=tmp_path,
        stem="control_method_mismatch",
        primary_metric_value=0.70,
        permutation_test=_passing_permutation(),
    )
    dummy_config_path, dummy_metrics_path = _write_run_payloads(
        tmp_path=tmp_path,
        stem="dummy_method_mismatch",
        primary_metric_value=0.20,
        methodology_policy_name="fixed_baselines_only",
        tuning_enabled=False,
        include_tuning_metadata=False,
    )

    payload = evaluate_claim_outcomes(
        protocol=protocol,
        compiled_manifest=manifest,
        run_results=[
            _run_result(
                primary_run_id,
                primary_claim.suite_ids[0],
                config_path=primary_config_path,
                metrics_path=primary_metrics_path,
            ),
            _run_result(
                control_model_run_id,
                control_claim.suite_ids[0],
                config_path=control_config_path,
                metrics_path=control_metrics_path,
            ),
            _run_result(
                control_dummy_run_id,
                control_claim.suite_ids[0],
                config_path=dummy_config_path,
                metrics_path=dummy_metrics_path,
            ),
        ],
        reporting_contract=_reporting_contract(),
    )

    entry = _primary_entry(payload)
    assert payload["primary_claim_verdict"] == "invalid"
    assert entry["reason"] == "methodology_or_cv_mismatch"
    assert entry["strict_gate_summary"]["cv_mismatch_run_ids"] == [primary_run_id]


def test_invalid_when_required_primary_evidence_is_incomplete(tmp_path: Path):
    protocol, primary_claim, control_claim = _load_protocol_and_claims()
    primary_run_id_1 = "primary_complete_1"
    primary_run_id_2 = "primary_complete_2"
    control_model_run_id = "control_complete"
    control_dummy_run_id = "dummy_complete"
    runs = [
        _RunSpecStub(run_id=primary_run_id_1, suite_id=primary_claim.suite_ids[0], model="ridge"),
        _RunSpecStub(run_id=primary_run_id_2, suite_id=primary_claim.suite_ids[0], model="ridge", repeat_id=2, repeat_count=2),
        _RunSpecStub(run_id=control_model_run_id, suite_id=control_claim.suite_ids[0], model="ridge"),
        _RunSpecStub(
            run_id=control_dummy_run_id,
            suite_id=control_claim.suite_ids[0],
            model="dummy",
            controls=_RunControlsStub(dummy_baseline_run=True),
        ),
    ]
    manifest = _manifest(protocol, primary_claim, control_claim, runs)
    primary_config_path, primary_metrics_path = _write_run_payloads(
        tmp_path=tmp_path,
        stem="primary_complete_1",
        primary_metric_value=0.70,
    )
    control_config_path, control_metrics_path = _write_run_payloads(
        tmp_path=tmp_path,
        stem="control_complete",
        primary_metric_value=0.70,
        permutation_test=_passing_permutation(),
    )
    dummy_config_path, dummy_metrics_path = _write_run_payloads(
        tmp_path=tmp_path,
        stem="dummy_complete",
        primary_metric_value=0.20,
        methodology_policy_name="fixed_baselines_only",
        tuning_enabled=False,
        include_tuning_metadata=False,
    )

    payload = evaluate_claim_outcomes(
        protocol=protocol,
        compiled_manifest=manifest,
        run_results=[
            _run_result(
                primary_run_id_1,
                primary_claim.suite_ids[0],
                config_path=primary_config_path,
                metrics_path=primary_metrics_path,
            ),
            _run_result(
                control_model_run_id,
                control_claim.suite_ids[0],
                config_path=control_config_path,
                metrics_path=control_metrics_path,
            ),
            _run_result(
                control_dummy_run_id,
                control_claim.suite_ids[0],
                config_path=dummy_config_path,
                metrics_path=dummy_metrics_path,
            ),
        ],
        reporting_contract=_reporting_contract(),
    )

    entry = _primary_entry(payload)
    assert payload["primary_claim_verdict"] == "invalid"
    assert entry["reason"] == "incomplete_primary_evidence"
    assert entry["strict_gate_summary"]["missing_primary_run_ids"] == [primary_run_id_2]


def test_invalid_when_confirmatory_valid_is_false(tmp_path: Path):
    protocol, primary_claim, control_claim = _load_protocol_and_claims()
    primary_run_id = "primary_confirmatory_invalid"
    control_model_run_id = "control_confirmatory_invalid"
    control_dummy_run_id = "dummy_confirmatory_invalid"
    runs = [
        _RunSpecStub(run_id=primary_run_id, suite_id=primary_claim.suite_ids[0], model="ridge"),
        _RunSpecStub(run_id=control_model_run_id, suite_id=control_claim.suite_ids[0], model="ridge"),
        _RunSpecStub(
            run_id=control_dummy_run_id,
            suite_id=control_claim.suite_ids[0],
            model="dummy",
            controls=_RunControlsStub(dummy_baseline_run=True),
        ),
    ]
    manifest = _manifest(protocol, primary_claim, control_claim, runs)
    primary_config_path, primary_metrics_path = _write_run_payloads(
        tmp_path=tmp_path,
        stem="primary_confirmatory_invalid",
        primary_metric_value=0.75,
    )
    control_config_path, control_metrics_path = _write_run_payloads(
        tmp_path=tmp_path,
        stem="control_confirmatory_invalid",
        primary_metric_value=0.75,
        permutation_test=_passing_permutation(),
    )
    dummy_config_path, dummy_metrics_path = _write_run_payloads(
        tmp_path=tmp_path,
        stem="dummy_confirmatory_invalid",
        primary_metric_value=0.10,
        methodology_policy_name="fixed_baselines_only",
        tuning_enabled=False,
        include_tuning_metadata=False,
    )

    payload = evaluate_claim_outcomes(
        protocol=protocol,
        compiled_manifest=manifest,
        run_results=[
            _run_result(
                primary_run_id,
                primary_claim.suite_ids[0],
                config_path=primary_config_path,
                metrics_path=primary_metrics_path,
            ),
            _run_result(
                control_model_run_id,
                control_claim.suite_ids[0],
                config_path=control_config_path,
                metrics_path=control_metrics_path,
            ),
            _run_result(
                control_dummy_run_id,
                control_claim.suite_ids[0],
                config_path=dummy_config_path,
                metrics_path=dummy_metrics_path,
            ),
        ],
        reporting_contract=_reporting_contract(confirmatory_valid=False),
    )

    entry = _primary_entry(payload)
    assert payload["primary_claim_verdict"] == "invalid"
    assert entry["reason"] == "confirmatory_valid_false"


def test_invalid_when_science_critical_deviation_detected(tmp_path: Path):
    protocol, primary_claim, control_claim = _load_protocol_and_claims()
    primary_run_id = "primary_science_critical"
    control_model_run_id = "control_science_critical"
    control_dummy_run_id = "dummy_science_critical"
    runs = [
        _RunSpecStub(run_id=primary_run_id, suite_id=primary_claim.suite_ids[0], model="ridge"),
        _RunSpecStub(run_id=control_model_run_id, suite_id=control_claim.suite_ids[0], model="ridge"),
        _RunSpecStub(
            run_id=control_dummy_run_id,
            suite_id=control_claim.suite_ids[0],
            model="dummy",
            controls=_RunControlsStub(dummy_baseline_run=True),
        ),
    ]
    manifest = _manifest(protocol, primary_claim, control_claim, runs)
    primary_config_path, primary_metrics_path = _write_run_payloads(
        tmp_path=tmp_path,
        stem="primary_science_critical",
        primary_metric_value=0.75,
    )
    control_config_path, control_metrics_path = _write_run_payloads(
        tmp_path=tmp_path,
        stem="control_science_critical",
        primary_metric_value=0.75,
        permutation_test=_passing_permutation(),
    )
    dummy_config_path, dummy_metrics_path = _write_run_payloads(
        tmp_path=tmp_path,
        stem="dummy_science_critical",
        primary_metric_value=0.10,
        methodology_policy_name="fixed_baselines_only",
        tuning_enabled=False,
        include_tuning_metadata=False,
    )

    payload = evaluate_claim_outcomes(
        protocol=protocol,
        compiled_manifest=manifest,
        run_results=[
            _run_result(
                primary_run_id,
                primary_claim.suite_ids[0],
                config_path=primary_config_path,
                metrics_path=primary_metrics_path,
            ),
            _run_result(
                control_model_run_id,
                control_claim.suite_ids[0],
                config_path=control_config_path,
                metrics_path=control_metrics_path,
            ),
            _run_result(
                control_dummy_run_id,
                control_claim.suite_ids[0],
                config_path=dummy_config_path,
                metrics_path=dummy_metrics_path,
            ),
        ],
        reporting_contract=_reporting_contract(science_critical=True),
    )

    entry = _primary_entry(payload)
    assert payload["primary_claim_verdict"] == "invalid"
    assert entry["reason"] == "science_critical_deviation_detected"


def test_not_supported_when_dummy_is_not_beaten(tmp_path: Path):
    protocol, primary_claim, control_claim = _load_protocol_and_claims()
    primary_run_id = "primary_dummy_fail"
    control_model_run_id = "control_dummy_fail"
    control_dummy_run_id = "dummy_dummy_fail"
    runs = [
        _RunSpecStub(run_id=primary_run_id, suite_id=primary_claim.suite_ids[0], model="ridge"),
        _RunSpecStub(run_id=control_model_run_id, suite_id=control_claim.suite_ids[0], model="ridge"),
        _RunSpecStub(
            run_id=control_dummy_run_id,
            suite_id=control_claim.suite_ids[0],
            model="dummy",
            controls=_RunControlsStub(dummy_baseline_run=True),
        ),
    ]
    manifest = _manifest(protocol, primary_claim, control_claim, runs)
    primary_config_path, primary_metrics_path = _write_run_payloads(
        tmp_path=tmp_path,
        stem="primary_dummy_fail",
        primary_metric_value=0.61,
    )
    control_config_path, control_metrics_path = _write_run_payloads(
        tmp_path=tmp_path,
        stem="control_dummy_fail",
        primary_metric_value=0.61,
        permutation_test=_passing_permutation(),
    )
    dummy_config_path, dummy_metrics_path = _write_run_payloads(
        tmp_path=tmp_path,
        stem="dummy_dummy_fail",
        primary_metric_value=0.61,
        methodology_policy_name="fixed_baselines_only",
        tuning_enabled=False,
        include_tuning_metadata=False,
    )
    payload = evaluate_claim_outcomes(
        protocol=protocol,
        compiled_manifest=manifest,
        run_results=[
            _run_result(
                primary_run_id,
                primary_claim.suite_ids[0],
                config_path=primary_config_path,
                metrics_path=primary_metrics_path,
            ),
            _run_result(
                control_model_run_id,
                control_claim.suite_ids[0],
                config_path=control_config_path,
                metrics_path=control_metrics_path,
            ),
            _run_result(
                control_dummy_run_id,
                control_claim.suite_ids[0],
                config_path=dummy_config_path,
                metrics_path=dummy_metrics_path,
            ),
        ],
        reporting_contract=_reporting_contract(),
    )

    entry = _primary_entry(payload)
    assert payload["primary_claim_verdict"] == "not_supported"
    assert entry["reason"] == "dummy_baseline_not_outperformed"
    assert entry["strict_gate_summary"]["baseline_fail_run_ids"] == [primary_run_id]


def test_not_supported_when_permutation_fails(tmp_path: Path):
    protocol, primary_claim, control_claim = _load_protocol_and_claims()
    primary_run_id = "primary_perm_fail"
    control_model_run_id = "control_perm_fail"
    control_dummy_run_id = "dummy_perm_fail"
    runs = [
        _RunSpecStub(run_id=primary_run_id, suite_id=primary_claim.suite_ids[0], model="ridge"),
        _RunSpecStub(run_id=control_model_run_id, suite_id=control_claim.suite_ids[0], model="ridge"),
        _RunSpecStub(
            run_id=control_dummy_run_id,
            suite_id=control_claim.suite_ids[0],
            model="dummy",
            controls=_RunControlsStub(dummy_baseline_run=True),
        ),
    ]
    manifest = _manifest(protocol, primary_claim, control_claim, runs)
    primary_config_path, primary_metrics_path = _write_run_payloads(
        tmp_path=tmp_path,
        stem="primary_perm_fail",
        primary_metric_value=0.71,
    )
    control_config_path, control_metrics_path = _write_run_payloads(
        tmp_path=tmp_path,
        stem="control_perm_fail",
        primary_metric_value=0.71,
        permutation_test=_failing_permutation(),
    )
    dummy_config_path, dummy_metrics_path = _write_run_payloads(
        tmp_path=tmp_path,
        stem="dummy_perm_fail",
        primary_metric_value=0.30,
        methodology_policy_name="fixed_baselines_only",
        tuning_enabled=False,
        include_tuning_metadata=False,
    )
    payload = evaluate_claim_outcomes(
        protocol=protocol,
        compiled_manifest=manifest,
        run_results=[
            _run_result(
                primary_run_id,
                primary_claim.suite_ids[0],
                config_path=primary_config_path,
                metrics_path=primary_metrics_path,
            ),
            _run_result(
                control_model_run_id,
                control_claim.suite_ids[0],
                config_path=control_config_path,
                metrics_path=control_metrics_path,
            ),
            _run_result(
                control_dummy_run_id,
                control_claim.suite_ids[0],
                config_path=dummy_config_path,
                metrics_path=dummy_metrics_path,
            ),
        ],
        reporting_contract=_reporting_contract(),
    )

    entry = _primary_entry(payload)
    assert payload["primary_claim_verdict"] == "not_supported"
    assert entry["reason"] == "permutation_requirement_not_met"
    assert entry["strict_gate_summary"]["permutation_fail_run_ids"] == [primary_run_id]


def test_regression_local_primary_permutation_alone_is_not_enough(tmp_path: Path):
    protocol, primary_claim, control_claim = _load_protocol_and_claims()
    primary_run_id = "primary_local_permutation_only"
    control_dummy_run_id = "dummy_local_permutation_only"
    runs = [
        _RunSpecStub(run_id=primary_run_id, suite_id=primary_claim.suite_ids[0], model="ridge"),
        _RunSpecStub(
            run_id=control_dummy_run_id,
            suite_id=control_claim.suite_ids[0],
            model="dummy",
            controls=_RunControlsStub(dummy_baseline_run=True),
        ),
    ]
    manifest = _manifest(protocol, primary_claim, control_claim, runs)
    primary_config_path, primary_metrics_path = _write_run_payloads(
        tmp_path=tmp_path,
        stem="primary_local_permutation_only",
        primary_metric_value=0.80,
        permutation_test=_passing_permutation(),
    )
    dummy_config_path, dummy_metrics_path = _write_run_payloads(
        tmp_path=tmp_path,
        stem="dummy_local_permutation_only",
        primary_metric_value=0.20,
        methodology_policy_name="fixed_baselines_only",
        tuning_enabled=False,
        include_tuning_metadata=False,
    )
    payload = evaluate_claim_outcomes(
        protocol=protocol,
        compiled_manifest=manifest,
        run_results=[
            _run_result(
                primary_run_id,
                primary_claim.suite_ids[0],
                config_path=primary_config_path,
                metrics_path=primary_metrics_path,
            ),
            _run_result(
                control_dummy_run_id,
                control_claim.suite_ids[0],
                config_path=dummy_config_path,
                metrics_path=dummy_metrics_path,
            ),
        ],
        reporting_contract=_reporting_contract(),
    )

    entry = _primary_entry(payload)
    assert payload["primary_claim_verdict"] == "invalid"
    assert entry["reason"] == "missing_or_inconsistent_supporting_control_evidence"
    assert entry["strict_gate_summary"]["missing_control_run_ids"] == [primary_run_id]


def test_regression_secondary_or_supporting_cannot_override_primary_gate(tmp_path: Path):
    protocol, primary_claim, control_claim = _load_protocol_and_claims()
    extra_claim = next(
        claim
        for claim in protocol.claims
        if claim.claim_id not in {primary_claim.claim_id, control_claim.claim_id}
    )
    primary_run_id = "primary_override_fail"
    control_model_run_id = "control_override_fail"
    control_dummy_run_id = "dummy_override_fail"
    extra_run_id = "extra_supporting_positive"
    runs = [
        _RunSpecStub(run_id=primary_run_id, suite_id=primary_claim.suite_ids[0], model="ridge"),
        _RunSpecStub(run_id=control_model_run_id, suite_id=control_claim.suite_ids[0], model="ridge"),
        _RunSpecStub(
            run_id=control_dummy_run_id,
            suite_id=control_claim.suite_ids[0],
            model="dummy",
            controls=_RunControlsStub(dummy_baseline_run=True),
        ),
        _RunSpecStub(
            run_id=extra_run_id,
            suite_id=extra_claim.suite_ids[0],
            model="ridge",
            cv_mode="frozen_cross_person_transfer",
            subject=None,
            train_subject="sub-001",
            test_subject="sub-002",
            methodology_policy_name="fixed_baselines_only",
            tuning_enabled=False,
            tuning_search_space_id=None,
            tuning_search_space_version=None,
            tuning_inner_cv_scheme=None,
            tuning_inner_group_field=None,
        ),
    ]
    manifest = _manifest(
        protocol,
        primary_claim,
        control_claim,
        runs,
        extra_claim_map={extra_claim.claim_id: [extra_run_id]},
    )
    primary_config_path, primary_metrics_path = _write_run_payloads(
        tmp_path=tmp_path,
        stem="primary_override_fail",
        primary_metric_value=0.80,
        primary_metric_name="accuracy",
    )
    control_config_path, control_metrics_path = _write_run_payloads(
        tmp_path=tmp_path,
        stem="control_override_fail",
        primary_metric_value=0.80,
        permutation_test=_passing_permutation(),
    )
    dummy_config_path, dummy_metrics_path = _write_run_payloads(
        tmp_path=tmp_path,
        stem="dummy_override_fail",
        primary_metric_value=0.10,
        methodology_policy_name="fixed_baselines_only",
        tuning_enabled=False,
        include_tuning_metadata=False,
    )
    extra_config_path, extra_metrics_path = _write_run_payloads(
        tmp_path=tmp_path,
        stem="extra_supporting_positive",
        primary_metric_value=0.99,
        cv_mode="frozen_cross_person_transfer",
        methodology_policy_name="fixed_baselines_only",
        tuning_enabled=False,
        include_tuning_metadata=False,
        permutation_test=_passing_permutation(),
    )
    payload = evaluate_claim_outcomes(
        protocol=protocol,
        compiled_manifest=manifest,
        run_results=[
            _run_result(
                primary_run_id,
                primary_claim.suite_ids[0],
                config_path=primary_config_path,
                metrics_path=primary_metrics_path,
            ),
            _run_result(
                control_model_run_id,
                control_claim.suite_ids[0],
                config_path=control_config_path,
                metrics_path=control_metrics_path,
            ),
            _run_result(
                control_dummy_run_id,
                control_claim.suite_ids[0],
                config_path=dummy_config_path,
                metrics_path=dummy_metrics_path,
            ),
            _run_result(
                extra_run_id,
                extra_claim.suite_ids[0],
                config_path=extra_config_path,
                metrics_path=extra_metrics_path,
            ),
        ],
        reporting_contract=_reporting_contract(),
    )

    entry = _primary_entry(payload)
    assert payload["primary_claim_verdict"] == "invalid"
    assert entry["reason"] == "primary_metric_mismatch"
