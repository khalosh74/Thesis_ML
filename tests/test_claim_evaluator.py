import json
from dataclasses import dataclass, field
from pathlib import Path

from Thesis_ML.protocols.claim_evaluator import evaluate_claim_outcomes
from Thesis_ML.protocols.loader import load_protocol
from Thesis_ML.protocols.models import ClaimCategory, ClaimRole, ProtocolRunResult


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
    controls: _RunControlsStub = field(default_factory=_RunControlsStub)


@dataclass
class _ClaimManifestStub:
    claim_to_run_map: dict[str, list[str]]
    runs: list[_RunSpecStub]


def _write_json(path: Path, payload: dict) -> str:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def _reporting_contract() -> dict[str, object]:
    return {
        "controls_status": {
            "controls_valid_for_confirmatory": True,
        },
        "deviations_from_protocol": {
            "science_critical_deviation_detected": False,
        },
    }


def _load_protocol_and_claims() -> tuple[object, object, object]:
    protocol = load_protocol(Path("configs/protocols/thesis_canonical_nested_v2.json"))
    primary_claim_id = protocol.success_criteria.primary_claim_id
    primary_claim = next(claim for claim in protocol.claims if claim.claim_id == primary_claim_id)
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


def _run_result(run_id: str, suite_id: str, metrics_path: str) -> ProtocolRunResult:
    return ProtocolRunResult.model_validate(
        {
            "run_id": run_id,
            "suite_id": suite_id,
            "status": "success",
            "metrics_path": metrics_path,
        }
    )


def _primary_claim_entry(payload: dict) -> dict:
    claim_id = payload["primary_claim_id"]
    return next(entry for entry in payload["claims"] if entry["claim_id"] == claim_id)


def test_supported_when_control_permutation_passes_and_non_dummy_beats_dummy(tmp_path: Path):
    protocol, primary_claim, control_claim = _load_protocol_and_claims()
    primary_run_id = "run_primary_sub001_r001"
    control_run_id = "run_control_sub001_r001"
    dummy_run_id = "run_dummy_sub001_r001"

    manifest = _ClaimManifestStub(
        claim_to_run_map={
            primary_claim.claim_id: [primary_run_id],
            control_claim.claim_id: [control_run_id, dummy_run_id],
        },
        runs=[
            _RunSpecStub(
                run_id=primary_run_id,
                suite_id=primary_claim.suite_ids[0],
                model="ridge",
            ),
            _RunSpecStub(
                run_id=control_run_id,
                suite_id=control_claim.suite_ids[0],
                model="ridge",
            ),
            _RunSpecStub(
                run_id=dummy_run_id,
                suite_id=control_claim.suite_ids[0],
                model="dummy",
                controls=_RunControlsStub(dummy_baseline_run=True),
            ),
        ],
    )

    run_results = [
        _run_result(
            primary_run_id,
            primary_claim.suite_ids[0],
            _write_json(
                tmp_path / "primary_metrics.json",
                {
                    "primary_metric_value": 0.72,
                    "permutation_test": {"passes_threshold": False},
                },
            ),
        ),
        _run_result(
            control_run_id,
            control_claim.suite_ids[0],
            _write_json(
                tmp_path / "control_metrics.json",
                {
                    "primary_metric_value": 0.72,
                    "permutation_test": {
                        "p_value": 0.01,
                        "passes_threshold": True,
                    },
                },
            ),
        ),
        _run_result(
            dummy_run_id,
            control_claim.suite_ids[0],
            _write_json(
                tmp_path / "dummy_metrics.json",
                {
                    "primary_metric_value": 0.50,
                },
            ),
        ),
    ]

    payload = evaluate_claim_outcomes(
        protocol=protocol,
        compiled_manifest=manifest,
        run_results=run_results,
        reporting_contract=_reporting_contract(),
    )

    assert payload["primary_claim_verdict"] == "supported"
    primary_entry = _primary_claim_entry(payload)
    summary = primary_entry["control_evidence_summary"]
    assert summary["matched_primary_run_count"] == 1
    assert summary["matched_control_run_count"] == 1
    assert summary["matched_dummy_run_count"] == 1


def test_not_supported_when_permutation_fails(tmp_path: Path):
    protocol, primary_claim, control_claim = _load_protocol_and_claims()
    primary_run_id = "run_primary_perm_fail"
    control_run_id = "run_control_perm_fail"
    dummy_run_id = "run_dummy_perm_fail"

    manifest = _ClaimManifestStub(
        claim_to_run_map={
            primary_claim.claim_id: [primary_run_id],
            control_claim.claim_id: [control_run_id, dummy_run_id],
        },
        runs=[
            _RunSpecStub(
                run_id=primary_run_id,
                suite_id=primary_claim.suite_ids[0],
                model="ridge",
            ),
            _RunSpecStub(
                run_id=control_run_id,
                suite_id=control_claim.suite_ids[0],
                model="ridge",
            ),
            _RunSpecStub(
                run_id=dummy_run_id,
                suite_id=control_claim.suite_ids[0],
                model="dummy",
                controls=_RunControlsStub(dummy_baseline_run=True),
            ),
        ],
    )

    payload = evaluate_claim_outcomes(
        protocol=protocol,
        compiled_manifest=manifest,
        run_results=[
            _run_result(
                primary_run_id,
                primary_claim.suite_ids[0],
                _write_json(tmp_path / "primary_perm_fail.json", {"primary_metric_value": 0.68}),
            ),
            _run_result(
                control_run_id,
                control_claim.suite_ids[0],
                _write_json(
                    tmp_path / "control_perm_fail.json",
                    {
                        "primary_metric_value": 0.68,
                        "permutation_test": {"p_value": 0.5, "passes_threshold": False},
                    },
                ),
            ),
            _run_result(
                dummy_run_id,
                control_claim.suite_ids[0],
                _write_json(tmp_path / "dummy_perm_fail.json", {"primary_metric_value": 0.40}),
            ),
        ],
        reporting_contract=_reporting_contract(),
    )

    assert payload["primary_claim_verdict"] == "not_supported"
    primary_entry = _primary_claim_entry(payload)
    assert primary_entry["control_evidence_summary"]["permutation_fail_run_ids"] == [primary_run_id]


def test_not_supported_when_dummy_is_not_beaten(tmp_path: Path):
    protocol, primary_claim, control_claim = _load_protocol_and_claims()
    primary_run_id = "run_primary_dummy_fail"
    control_run_id = "run_control_dummy_fail"
    dummy_run_id = "run_dummy_dummy_fail"

    manifest = _ClaimManifestStub(
        claim_to_run_map={
            primary_claim.claim_id: [primary_run_id],
            control_claim.claim_id: [control_run_id, dummy_run_id],
        },
        runs=[
            _RunSpecStub(
                run_id=primary_run_id,
                suite_id=primary_claim.suite_ids[0],
                model="ridge",
            ),
            _RunSpecStub(
                run_id=control_run_id,
                suite_id=control_claim.suite_ids[0],
                model="ridge",
            ),
            _RunSpecStub(
                run_id=dummy_run_id,
                suite_id=control_claim.suite_ids[0],
                model="dummy",
                controls=_RunControlsStub(dummy_baseline_run=True),
            ),
        ],
    )

    payload = evaluate_claim_outcomes(
        protocol=protocol,
        compiled_manifest=manifest,
        run_results=[
            _run_result(
                primary_run_id,
                primary_claim.suite_ids[0],
                _write_json(tmp_path / "primary_dummy_fail.json", {"primary_metric_value": 0.60}),
            ),
            _run_result(
                control_run_id,
                control_claim.suite_ids[0],
                _write_json(
                    tmp_path / "control_dummy_fail.json",
                    {
                        "primary_metric_value": 0.60,
                        "permutation_test": {"p_value": 0.01, "passes_threshold": True},
                    },
                ),
            ),
            _run_result(
                dummy_run_id,
                control_claim.suite_ids[0],
                _write_json(tmp_path / "dummy_dummy_fail.json", {"primary_metric_value": 0.60}),
            ),
        ],
        reporting_contract=_reporting_contract(),
    )

    assert payload["primary_claim_verdict"] == "not_supported"
    primary_entry = _primary_claim_entry(payload)
    assert primary_entry["control_evidence_summary"]["baseline_fail_run_ids"] == [primary_run_id]


def test_invalid_when_matched_control_evidence_is_missing(tmp_path: Path):
    protocol, primary_claim, control_claim = _load_protocol_and_claims()
    primary_run_id = "run_primary_missing_control"
    dummy_run_id = "run_dummy_missing_control"

    manifest = _ClaimManifestStub(
        claim_to_run_map={
            primary_claim.claim_id: [primary_run_id],
            control_claim.claim_id: [dummy_run_id],
        },
        runs=[
            _RunSpecStub(
                run_id=primary_run_id,
                suite_id=primary_claim.suite_ids[0],
                model="ridge",
            ),
            _RunSpecStub(
                run_id=dummy_run_id,
                suite_id=control_claim.suite_ids[0],
                model="dummy",
                controls=_RunControlsStub(dummy_baseline_run=True),
            ),
        ],
    )

    payload = evaluate_claim_outcomes(
        protocol=protocol,
        compiled_manifest=manifest,
        run_results=[
            _run_result(
                primary_run_id,
                primary_claim.suite_ids[0],
                _write_json(tmp_path / "primary_missing_control.json", {"primary_metric_value": 0.70}),
            ),
            _run_result(
                dummy_run_id,
                control_claim.suite_ids[0],
                _write_json(tmp_path / "dummy_missing_control.json", {"primary_metric_value": 0.30}),
            ),
        ],
        reporting_contract=_reporting_contract(),
    )

    assert payload["primary_claim_verdict"] == "invalid"
    primary_entry = _primary_claim_entry(payload)
    assert primary_run_id in primary_entry["control_evidence_summary"]["missing_control_runs"]


def test_invalid_when_matched_dummy_evidence_is_missing(tmp_path: Path):
    protocol, primary_claim, control_claim = _load_protocol_and_claims()
    primary_run_id = "run_primary_missing_dummy"
    control_run_id = "run_control_missing_dummy"

    manifest = _ClaimManifestStub(
        claim_to_run_map={
            primary_claim.claim_id: [primary_run_id],
            control_claim.claim_id: [control_run_id],
        },
        runs=[
            _RunSpecStub(
                run_id=primary_run_id,
                suite_id=primary_claim.suite_ids[0],
                model="ridge",
            ),
            _RunSpecStub(
                run_id=control_run_id,
                suite_id=control_claim.suite_ids[0],
                model="ridge",
            ),
        ],
    )

    payload = evaluate_claim_outcomes(
        protocol=protocol,
        compiled_manifest=manifest,
        run_results=[
            _run_result(
                primary_run_id,
                primary_claim.suite_ids[0],
                _write_json(tmp_path / "primary_missing_dummy.json", {"primary_metric_value": 0.70}),
            ),
            _run_result(
                control_run_id,
                control_claim.suite_ids[0],
                _write_json(
                    tmp_path / "control_missing_dummy.json",
                    {
                        "primary_metric_value": 0.70,
                        "permutation_test": {"p_value": 0.01, "passes_threshold": True},
                    },
                ),
            ),
        ],
        reporting_contract=_reporting_contract(),
    )

    assert payload["primary_claim_verdict"] == "invalid"
    primary_entry = _primary_claim_entry(payload)
    assert primary_run_id in primary_entry["control_evidence_summary"]["missing_dummy_runs"]


def test_invalid_when_primary_and_control_metrics_disagree(tmp_path: Path):
    protocol, primary_claim, control_claim = _load_protocol_and_claims()
    primary_run_id = "run_primary_metric_mismatch"
    control_run_id = "run_control_metric_mismatch"
    dummy_run_id = "run_dummy_metric_mismatch"

    manifest = _ClaimManifestStub(
        claim_to_run_map={
            primary_claim.claim_id: [primary_run_id],
            control_claim.claim_id: [control_run_id, dummy_run_id],
        },
        runs=[
            _RunSpecStub(
                run_id=primary_run_id,
                suite_id=primary_claim.suite_ids[0],
                model="ridge",
            ),
            _RunSpecStub(
                run_id=control_run_id,
                suite_id=control_claim.suite_ids[0],
                model="ridge",
            ),
            _RunSpecStub(
                run_id=dummy_run_id,
                suite_id=control_claim.suite_ids[0],
                model="dummy",
                controls=_RunControlsStub(dummy_baseline_run=True),
            ),
        ],
    )

    payload = evaluate_claim_outcomes(
        protocol=protocol,
        compiled_manifest=manifest,
        run_results=[
            _run_result(
                primary_run_id,
                primary_claim.suite_ids[0],
                _write_json(
                    tmp_path / "primary_metric_mismatch.json",
                    {"primary_metric_value": 0.80},
                ),
            ),
            _run_result(
                control_run_id,
                control_claim.suite_ids[0],
                _write_json(
                    tmp_path / "control_metric_mismatch.json",
                    {
                        "primary_metric_value": 0.60,
                        "permutation_test": {"p_value": 0.01, "passes_threshold": True},
                    },
                ),
            ),
            _run_result(
                dummy_run_id,
                control_claim.suite_ids[0],
                _write_json(tmp_path / "dummy_metric_mismatch.json", {"primary_metric_value": 0.30}),
            ),
        ],
        reporting_contract=_reporting_contract(),
    )

    assert payload["primary_claim_verdict"] == "invalid"
    primary_entry = _primary_claim_entry(payload)
    assert primary_run_id in primary_entry["control_evidence_summary"]["metric_mismatch_run_ids"]


def test_invalid_when_completeness_required_and_primary_evidence_incomplete(tmp_path: Path):
    protocol, primary_claim, control_claim = _load_protocol_and_claims()
    primary_run_id_1 = "run_primary_complete_1"
    primary_run_id_2 = "run_primary_complete_2"
    control_run_id = "run_control_complete_1"
    dummy_run_id = "run_dummy_complete_1"

    manifest = _ClaimManifestStub(
        claim_to_run_map={
            primary_claim.claim_id: [primary_run_id_1, primary_run_id_2],
            control_claim.claim_id: [control_run_id, dummy_run_id],
        },
        runs=[
            _RunSpecStub(
                run_id=primary_run_id_1,
                suite_id=primary_claim.suite_ids[0],
                model="ridge",
                repeat_id=1,
                repeat_count=2,
            ),
            _RunSpecStub(
                run_id=primary_run_id_2,
                suite_id=primary_claim.suite_ids[0],
                model="ridge",
                repeat_id=2,
                repeat_count=2,
            ),
            _RunSpecStub(
                run_id=control_run_id,
                suite_id=control_claim.suite_ids[0],
                model="ridge",
                repeat_id=1,
                repeat_count=2,
            ),
            _RunSpecStub(
                run_id=dummy_run_id,
                suite_id=control_claim.suite_ids[0],
                model="dummy",
                repeat_id=1,
                repeat_count=2,
                controls=_RunControlsStub(dummy_baseline_run=True),
            ),
        ],
    )

    payload = evaluate_claim_outcomes(
        protocol=protocol,
        compiled_manifest=manifest,
        run_results=[
            _run_result(
                primary_run_id_1,
                primary_claim.suite_ids[0],
                _write_json(tmp_path / "primary_complete_1.json", {"primary_metric_value": 0.70}),
            ),
            _run_result(
                control_run_id,
                control_claim.suite_ids[0],
                _write_json(
                    tmp_path / "control_complete_1.json",
                    {
                        "primary_metric_value": 0.70,
                        "permutation_test": {"p_value": 0.01, "passes_threshold": True},
                    },
                ),
            ),
            _run_result(
                dummy_run_id,
                control_claim.suite_ids[0],
                _write_json(tmp_path / "dummy_complete_1.json", {"primary_metric_value": 0.50}),
            ),
        ],
        reporting_contract=_reporting_contract(),
    )

    assert payload["primary_claim_verdict"] == "invalid"
    primary_entry = _primary_claim_entry(payload)
    assert primary_run_id_2 in primary_entry["control_evidence_summary"]["missing_primary_runs"]


def test_regression_primary_local_permutation_is_not_enough_without_supporting_controls(
    tmp_path: Path,
):
    protocol, primary_claim, control_claim = _load_protocol_and_claims()
    primary_run_id = "run_primary_local_permutation_only"

    manifest = _ClaimManifestStub(
        claim_to_run_map={
            primary_claim.claim_id: [primary_run_id],
            control_claim.claim_id: [],
        },
        runs=[
            _RunSpecStub(
                run_id=primary_run_id,
                suite_id=primary_claim.suite_ids[0],
                model="ridge",
            )
        ],
    )

    payload = evaluate_claim_outcomes(
        protocol=protocol,
        compiled_manifest=manifest,
        run_results=[
            _run_result(
                primary_run_id,
                primary_claim.suite_ids[0],
                _write_json(
                    tmp_path / "primary_local_permutation_only.json",
                    {
                        "primary_metric_value": 0.75,
                        "permutation_test": {"p_value": 0.001, "passes_threshold": True},
                    },
                ),
            )
        ],
        reporting_contract=_reporting_contract(),
    )

    assert payload["primary_claim_verdict"] == "invalid"


def test_regression_untuned_baseline_runs_do_not_count_as_primary_claim_evidence(tmp_path: Path):
    protocol, primary_claim, control_claim = _load_protocol_and_claims()
    untuned_run_id = "run_primary_untuned_only"

    manifest = _ClaimManifestStub(
        claim_to_run_map={
            primary_claim.claim_id: [untuned_run_id],
            control_claim.claim_id: [],
        },
        runs=[
            _RunSpecStub(
                run_id=untuned_run_id,
                suite_id=primary_claim.suite_ids[0],
                model="ridge",
                evidence_run_role="untuned_baseline",
            )
        ],
    )

    payload = evaluate_claim_outcomes(
        protocol=protocol,
        compiled_manifest=manifest,
        run_results=[
            _run_result(
                untuned_run_id,
                primary_claim.suite_ids[0],
                _write_json(tmp_path / "untuned_only.json", {"primary_metric_value": 0.90}),
            )
        ],
        reporting_contract=_reporting_contract(),
    )

    assert payload["primary_claim_verdict"] == "inconclusive"
