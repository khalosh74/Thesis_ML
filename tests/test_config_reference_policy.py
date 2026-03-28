from __future__ import annotations

from pathlib import Path


_MIGRATED_TEST_FILES = (
    "tests/acceptance/test_canonical_nested_v2_protocol_acceptance.py",
    "tests/acceptance/test_execution_status_payload_keys.py",
    "tests/acceptance/test_grouped_nested_comparison_dry_run_acceptance.py",
    "tests/acceptance/test_canonical_protocol_acceptance.py",
    "tests/acceptance/test_comparison_dry_run_acceptance.py",
    "tests/test_claim_evaluator.py",
    "tests/test_official_verification.py",
    "tests/test_protocols.py",
    "tests/test_comparisons.py",
    "tests/test_model_layer_architecture.py",
    "tests/test_model_cost_policy.py",
    "tests/test_campaign_runtime_profile.py",
    "tests/test_official_gpu_admission.py",
    "tests/test_official_max_both_admission.py",
    "tests/test_official_protocol_gpu_admission.py",
)

_BANNED_FILENAME_LITERALS = (
    "thesis_canonical_v1.json",
    "thesis_canonical_nested_v1.json",
    "thesis_canonical_nested_v2.json",
    "thesis_confirmatory_v1.json",
    "model_family_comparison_v1.json",
    "model_family_grouped_nested_comparison_v1.json",
    "model_family_grouped_nested_comparison_v2.json",
)

_BANNED_PATH_PATTERNS = (
    '_repo_root() / "configs"',
    'Path("configs/',
)


def test_migrated_tests_do_not_hardcode_config_json_paths() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    failures: list[str] = []
    for relative_path in _MIGRATED_TEST_FILES:
        file_path = repo_root / relative_path
        text = file_path.read_text(encoding="utf-8")
        for literal in _BANNED_FILENAME_LITERALS:
            if literal in text:
                failures.append(f"{relative_path}: contains banned filename literal '{literal}'")
        for pattern in _BANNED_PATH_PATTERNS:
            if pattern in text:
                failures.append(f"{relative_path}: contains banned path pattern '{pattern}'")
    assert not failures, "\n".join(failures)
