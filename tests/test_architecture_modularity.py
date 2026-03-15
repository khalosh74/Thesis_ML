from __future__ import annotations

from Thesis_ML.comparisons.artifacts import build_comparison_decision as artifacts_decision
from Thesis_ML.comparisons.decision import build_comparison_decision as decision_module_decision
from Thesis_ML.config.framework_mode import FrameworkMode
from Thesis_ML.experiments.runtime_policies import resolve_framework_context
from Thesis_ML.orchestration.study_review import append_warning


def test_comparison_decision_reexport_uses_dedicated_module() -> None:
    assert artifacts_decision is decision_module_decision


def test_runtime_policies_resolve_exploratory_context() -> None:
    mode, canonical, protocol_context, comparison_context = resolve_framework_context(
        FrameworkMode.EXPLORATORY.value,
        protocol_context=None,
        comparison_context=None,
    )
    assert mode == FrameworkMode.EXPLORATORY
    assert canonical is False
    assert protocol_context == {}
    assert comparison_context == {}


def test_study_review_append_warning_keeps_non_empty_messages() -> None:
    warnings: list[str] = []
    append_warning(warnings, "  warn me  ")
    append_warning(warnings, "   ")
    assert warnings == ["warn me"]
