from Thesis_ML.verification.confirmatory_ready import verify_confirmatory_ready
from Thesis_ML.verification.official_artifacts import verify_official_artifacts
from Thesis_ML.verification.performance_baseline import (
    BaselineBundle,
    BenchmarkCaseResult,
    BenchmarkCaseSpec,
    BenchmarkFingerprint,
    EnvironmentSnapshot,
    compare_baseline_bundles,
    run_baseline_suite,
    write_baseline_bundle,
)
from Thesis_ML.verification.repro_manifest import (
    build_reproducibility_manifest,
    write_reproducibility_manifest,
)
from Thesis_ML.verification.reproducibility import compare_official_outputs


def verify_campaign_runtime_profile(*args, **kwargs):
    from Thesis_ML.verification.campaign_runtime_profile import (
        verify_campaign_runtime_profile as _verify_campaign_runtime_profile,
    )

    return _verify_campaign_runtime_profile(*args, **kwargs)


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
