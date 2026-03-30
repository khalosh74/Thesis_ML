from __future__ import annotations

import argparse
from typing import Any

from Thesis_ML.orchestration.campaign_cli import (
    build_parser as _cli_build_parser,
)
from Thesis_ML.orchestration.campaign_cli import (
    main as _cli_main,
)
from Thesis_ML.orchestration.campaign_cli import (
    print_registry_status as _cli_print_registry_status,
)
from Thesis_ML.orchestration.campaign_cli import (
    print_stage1_commands as _cli_print_stage1_commands,
)
from Thesis_ML.orchestration.campaign_engine import (
    run_decision_support_campaign as _engine_run_decision_support_campaign,
)
from Thesis_ML.orchestration.campaign_engine import (
    run_workbook_decision_support_campaign as _engine_run_workbook_decision_support_campaign,
)
from Thesis_ML.orchestration.contracts import CompiledStudyManifest
from Thesis_ML.orchestration.experiment_selection import (
    collect_dataset_scope as _collect_dataset_scope,
)
from Thesis_ML.orchestration.experiment_selection import (
    select_experiments as _select_experiments,
)
from Thesis_ML.orchestration.reporting import status_snapshot as _status_snapshot
from Thesis_ML.orchestration.study_loading import (
    read_registry_manifest as _read_registry,
)
from Thesis_ML.orchestration.variant_expansion import (
    expand_experiment_variants as _expand_experiment_variants,
)
from Thesis_ML.orchestration.variant_expansion import (
    expand_template_variants as _expand_template_variants,
)


def run_experiment(**kwargs: Any) -> dict[str, Any]:
    from Thesis_ML.experiments.run_experiment import run_experiment as _run_experiment

    return _run_experiment(**kwargs)


def run_decision_support_campaign(
    *,
    run_experiment_fn: Any | None = None,
    runtime_profile_summary: Any | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    if runtime_profile_summary is not None:
        kwargs["runtime_profile_summary"] = runtime_profile_summary
    return _engine_run_decision_support_campaign(
        run_experiment_fn=run_experiment_fn or run_experiment,
        **kwargs,
    )


def run_workbook_decision_support_campaign(
    *,
    run_experiment_fn: Any | None = None,
    runtime_profile_summary: Any | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    if runtime_profile_summary is not None:
        kwargs["runtime_profile_summary"] = runtime_profile_summary
    return _engine_run_workbook_decision_support_campaign(
        run_experiment_fn=run_experiment_fn or run_experiment,
        **kwargs,
    )


def _build_parser() -> argparse.ArgumentParser:
    return _cli_build_parser()


def _print_registry_status(registry: CompiledStudyManifest) -> None:
    _cli_print_registry_status(registry)


def _print_stage1_commands(args: argparse.Namespace) -> None:
    _cli_print_stage1_commands(args)


def main(argv: list[str] | None = None) -> int:
    return _cli_main(
        argv,
        run_decision_support_campaign_fn=run_decision_support_campaign,
        run_workbook_decision_support_campaign_fn=run_workbook_decision_support_campaign,
        read_registry_manifest_fn=_read_registry,
    )


__all__ = [
    "run_experiment",
    "_collect_dataset_scope",
    "_expand_template_variants",
    "_expand_experiment_variants",
    "_select_experiments",
    "_status_snapshot",
    "run_decision_support_campaign",
    "run_workbook_decision_support_campaign",
    "main",
]
