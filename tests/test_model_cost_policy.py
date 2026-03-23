from __future__ import annotations

import json
from pathlib import Path

import pytest

from Thesis_ML.comparisons.compiler import compile_comparison
from Thesis_ML.comparisons.loader import load_comparison_spec
from Thesis_ML.experiments.model_catalog import projected_runtime_seconds
from Thesis_ML.protocols.compiler import compile_protocol
from Thesis_ML.protocols.loader import load_protocol
from Thesis_ML.verification.model_cost_policy import verify_model_cost_policy_precheck


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _demo_index_csv() -> Path:
    return _repo_root() / "demo_data" / "synthetic_v1" / "dataset_index.csv"


def _canonical_protocol_path() -> Path:
    return _repo_root() / "configs" / "protocols" / "thesis_canonical_v1.json"


def _comparison_path() -> Path:
    return _repo_root() / "configs" / "comparisons" / "model_family_comparison_v1.json"


def _nested_comparison_path() -> Path:
    return (
        _repo_root() / "configs" / "comparisons" / "model_family_grouped_nested_comparison_v1.json"
    )


def test_confirmatory_protocol_rejects_disallowed_cost_tier(tmp_path: Path) -> None:
    payload = json.loads(_canonical_protocol_path().read_text(encoding="utf-8"))
    payload["model_policy"]["models"] = ["logreg"]
    invalid_protocol_path = tmp_path / "invalid_confirmatory_cost_policy.json"
    invalid_protocol_path.write_text(f"{json.dumps(payload, indent=2)}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="model cost policy rejected model 'logreg'"):
        load_protocol(invalid_protocol_path)


def test_comparison_rejects_expensive_model_without_explicit_allowance(tmp_path: Path) -> None:
    payload = json.loads(_comparison_path().read_text(encoding="utf-8"))
    payload["cost_policy"]["explicit_benchmark_expensive_models"] = []
    invalid_comparison_path = tmp_path / "invalid_comparison_cost_policy.json"
    invalid_comparison_path.write_text(f"{json.dumps(payload, indent=2)}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="explicit_benchmark_expensive_models"):
        load_comparison_spec(invalid_comparison_path)


def test_compiled_protocol_and_comparison_manifest_stamp_cost_metadata() -> None:
    protocol = load_protocol(_canonical_protocol_path())
    protocol_manifest = compile_protocol(protocol, index_csv=_demo_index_csv())
    assert protocol_manifest.runs
    first_protocol_run = protocol_manifest.runs[0]
    assert first_protocol_run.model_cost_tier.value in {
        "official_fast",
        "official_allowed",
        "benchmark_expensive",
        "exploratory_only",
    }
    assert int(first_protocol_run.projected_runtime_seconds) > 0

    comparison = load_comparison_spec(_comparison_path())
    comparison_manifest = compile_comparison(comparison, index_csv=_demo_index_csv())
    assert comparison_manifest.runs
    first_comparison_run = comparison_manifest.runs[0]
    assert first_comparison_run.model_cost_tier.value in {
        "official_fast",
        "official_allowed",
        "benchmark_expensive",
        "exploratory_only",
    }
    assert int(first_comparison_run.projected_runtime_seconds) > 0


def test_projected_runtime_values_are_positive_and_stable() -> None:
    for model_name in ("ridge", "linearsvc", "logreg", "xgboost", "dummy"):
        for mode in ("exploratory", "locked_comparison", "confirmatory"):
            first = projected_runtime_seconds(model_name=model_name, framework_mode=mode)
            second = projected_runtime_seconds(model_name=model_name, framework_mode=mode)
            assert int(first) > 0
            assert int(first) == int(second)


def test_model_cost_precheck_passes_for_shipped_official_configs() -> None:
    summary = verify_model_cost_policy_precheck(
        index_csv=_demo_index_csv(),
        confirmatory_protocol=_canonical_protocol_path(),
        comparison_specs=[_comparison_path(), _nested_comparison_path()],
    )
    assert summary["passed"] is True
    assert int(summary["aggregate"]["n_runs"]) > 0
    assert not summary["issues"]


def test_model_cost_precheck_fails_for_invalid_synthetic_spec(tmp_path: Path) -> None:
    payload = json.loads(_comparison_path().read_text(encoding="utf-8"))
    payload["cost_policy"]["explicit_benchmark_expensive_models"] = []
    invalid_comparison_path = tmp_path / "invalid_comparison_cost_policy.json"
    invalid_comparison_path.write_text(f"{json.dumps(payload, indent=2)}\n", encoding="utf-8")

    summary = verify_model_cost_policy_precheck(
        index_csv=_demo_index_csv(),
        confirmatory_protocol=_canonical_protocol_path(),
        comparison_specs=[invalid_comparison_path],
    )
    assert summary["passed"] is False
    assert any(str(issue.get("code")) == "comparison_compile_failed" for issue in summary["issues"])
