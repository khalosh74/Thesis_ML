"""Verification package exports.

Avoid eager imports so model/comparison/protocol import paths do not pull in
heavy runtime dependencies (for example nibabel) unless those verification
entrypoints are actually used.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_LAZY_EXPORT_MODULES: dict[str, str] = {
    "BaselineBundle": "Thesis_ML.verification.performance_baseline",
    "BenchmarkCaseResult": "Thesis_ML.verification.performance_baseline",
    "BenchmarkCaseSpec": "Thesis_ML.verification.performance_baseline",
    "BenchmarkFingerprint": "Thesis_ML.verification.performance_baseline",
    "EnvironmentSnapshot": "Thesis_ML.verification.performance_baseline",
    "build_reproducibility_manifest": "Thesis_ML.verification.repro_manifest",
    "compare_baseline_bundles": "Thesis_ML.verification.performance_baseline",
    "compare_official_outputs": "Thesis_ML.verification.reproducibility",
    "run_baseline_suite": "Thesis_ML.verification.performance_baseline",
    "verify_campaign_runtime_profile": "Thesis_ML.verification.campaign_runtime_profile",
    "verify_confirmatory_ready": "Thesis_ML.verification.confirmatory_ready",
    "verify_official_artifacts": "Thesis_ML.verification.official_artifacts",
    "write_baseline_bundle": "Thesis_ML.verification.performance_baseline",
    "write_reproducibility_manifest": "Thesis_ML.verification.repro_manifest",
}


def __getattr__(name: str) -> Any:
    module_name = _LAZY_EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
    module = import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(__all__))


__all__ = [
    "BaselineBundle",
    "BenchmarkCaseResult",
    "BenchmarkCaseSpec",
    "BenchmarkFingerprint",
    "EnvironmentSnapshot",
    "build_reproducibility_manifest",
    "compare_baseline_bundles",
    "compare_official_outputs",
    "run_baseline_suite",
    "verify_campaign_runtime_profile",
    "verify_confirmatory_ready",
    "verify_official_artifacts",
    "write_baseline_bundle",
    "write_reproducibility_manifest",
]
