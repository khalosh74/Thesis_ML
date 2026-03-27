from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from hashlib import sha1
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid

from Thesis_ML.comparisons.compiler import compile_comparison
from Thesis_ML.comparisons.loader import load_comparison_spec
from Thesis_ML.comparisons.models import CompiledComparisonRunSpec
from Thesis_ML.config.framework_mode import FrameworkMode
from Thesis_ML.experiments.cache_loading import load_features_from_cache
from Thesis_ML.experiments.progress import ProgressCallback, emit_progress
from Thesis_ML.experiments.run_experiment import run_experiment
from Thesis_ML.experiments.section_models import DatasetSelectionInput
from Thesis_ML.experiments.sections import dataset_selection
from Thesis_ML.experiments.tuning_search_spaces import get_search_space
from Thesis_ML.protocols.compiler import compile_protocol
from Thesis_ML.protocols.loader import load_protocol
from Thesis_ML.protocols.models import CompiledRunSpec


@dataclass(frozen=True)
class _PlannedProfileRun:
    phase: str
    source_id: str
    source_version: str
    run: CompiledRunSpec | CompiledComparisonRunSpec
    evidence_policy: dict[str, Any]
    data_policy: dict[str, Any]


@dataclass(frozen=True)
class _ProfileRunValidity:
    can_profile_measured: bool
    expected_outer_folds: int
    estimated_train_inner_groups_after_outer_fold: int | None
    reason: str | None
    profiling_subset_description: str


@dataclass
class _ProfilingFeatureMatrixMemoizer:
    _store: dict[str, tuple[np.ndarray, pd.DataFrame, dict[str, Any]]] = field(default_factory=dict)
    hits: int = 0
    misses: int = 0
    last_lookup_hit: bool = False
    last_lookup_key: str | None = None
    last_lookup_sample_signature: str | None = None

    @staticmethod
    def _sample_signature(index_df: pd.DataFrame) -> str:
        if "sample_id" in index_df.columns:
            sample_values = index_df["sample_id"].astype(str).tolist()
        else:
            sample_values = index_df.index.astype(str).tolist()
        signature_source = "\x1f".join(sample_values)
        return sha1(signature_source.encode("utf-8")).hexdigest()

    def _cache_key(
        self,
        *,
        index_df: pd.DataFrame,
        cache_manifest_path: Path,
        affine_atol: float,
    ) -> tuple[str, str]:
        sample_signature = self._sample_signature(index_df)
        resolved_manifest = Path(cache_manifest_path).resolve()
        key_payload = {
            "cache_manifest_path": str(resolved_manifest),
            "affine_atol": float(affine_atol),
            "n_rows": int(len(index_df)),
            "sample_signature": sample_signature,
        }
        cache_key = json.dumps(key_payload, sort_keys=True, separators=(",", ":"))
        return cache_key, sample_signature

    def load(
        self,
        *,
        index_df: pd.DataFrame,
        cache_manifest_path: Path,
        spatial_report_path: Path | None = None,
        affine_atol: float,
    ) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any]]:
        cache_key, sample_signature = self._cache_key(
            index_df=index_df,
            cache_manifest_path=cache_manifest_path,
            affine_atol=affine_atol,
        )
        self.last_lookup_key = cache_key
        self.last_lookup_sample_signature = sample_signature
        cached = self._store.get(cache_key)
        if cached is not None:
            self.hits += 1
            self.last_lookup_hit = True
            x_matrix, metadata_df, spatial_payload = cached
            return x_matrix, metadata_df.copy(deep=False), dict(spatial_payload)

        self.misses += 1
        self.last_lookup_hit = False
        x_matrix, metadata_df, spatial_payload = load_features_from_cache(
            index_df=index_df,
            cache_manifest_path=cache_manifest_path,
            spatial_report_path=spatial_report_path,
            affine_atol=affine_atol,
        )
        self._store[cache_key] = (x_matrix, metadata_df.copy(deep=True), dict(spatial_payload))
        return x_matrix, metadata_df.copy(deep=False), dict(spatial_payload)

    def summary_payload(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "hits": int(self.hits),
            "misses": int(self.misses),
            "unique_entries": int(len(self._store)),
            "lookups": int(self.hits + self.misses),
            "reuse_happened": bool(self.hits > 0),
        }


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _humanize_seconds(total_seconds: float) -> str:
    seconds = max(0, int(round(float(total_seconds))))
    days, remainder = divmod(seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, secs = divmod(remainder, 60)
    parts: list[str] = []
    if days:
        parts.append(f"{days}d")
    if hours or days:
        parts.append(f"{hours}h")
    if minutes or hours or days:
        parts.append(f"{minutes}m")
    parts.append(f"{secs}s")
    return " ".join(parts)


def _issue(code: str, message: str, *, details: dict[str, Any] | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {"code": str(code), "message": str(message)}
    if details:
        payload["details"] = dict(details)
    return payload


def _coerce_enum(value: Any) -> str:
    if hasattr(value, "value"):
        return str(value.value)
    return str(value)


def _cohort_descriptor(record: _PlannedProfileRun) -> dict[str, Any]:
    run = record.run
    descriptor: dict[str, Any] = {
        "phase": str(record.phase),
        "framework_mode_source": str(run.framework_mode),
        "cv_mode": str(run.cv_mode),
        "methodology_policy_name": _coerce_enum(run.methodology_policy_name),
        "tuning_enabled": bool(run.tuning_enabled),
        "model": str(run.model),
        "class_weight_policy": _coerce_enum(run.class_weight_policy),
        "evidence_run_role": _coerce_enum(run.evidence_run_role),
        "controls_permutation_enabled": bool(run.controls.permutation_enabled),
        "controls_n_permutations": int(run.controls.n_permutations),
        "target": str(run.target),
        "filter_task": str(run.filter_task) if run.filter_task is not None else None,
        "filter_modality": str(run.filter_modality) if run.filter_modality is not None else None,
    }
    if isinstance(run, CompiledRunSpec):
        descriptor["cohort_source_id"] = str(run.suite_id)
    elif isinstance(run, CompiledComparisonRunSpec):
        descriptor["cohort_source_id"] = str(run.variant_id)
    return descriptor


def _cohort_key(descriptor: dict[str, Any]) -> str:
    return json.dumps(descriptor, sort_keys=True, separators=(",", ":"))


def _expected_outer_folds(index_csv: Path, run: CompiledRunSpec | CompiledComparisonRunSpec) -> int:
    selected = dataset_selection(
        DatasetSelectionInput(
            index_csv=index_csv,
            target_column=str(run.target),
            cv_mode=str(run.cv_mode),
            subject=str(run.subject) if run.subject is not None else None,
            train_subject=str(run.train_subject) if run.train_subject is not None else None,
            test_subject=str(run.test_subject) if run.test_subject is not None else None,
            filter_task=str(run.filter_task) if run.filter_task is not None else None,
            filter_modality=(str(run.filter_modality) if run.filter_modality is not None else None),
        )
    ).selected_index_df
    cv_mode = str(run.cv_mode)
    if cv_mode == "frozen_cross_person_transfer":
        return 1
    if cv_mode == "within_subject_loso_session":
        folds = int(selected["session"].astype(str).nunique(dropna=False))
        if folds <= 0:
            raise ValueError(
                f"Expected at least one outer fold for run '{run.run_id}', resolved {folds}."
            )
        return folds
    groups = (selected["subject"].astype(str) + "_" + selected["session"].astype(str)).nunique(
        dropna=False
    )
    folds = int(groups)
    if folds <= 0:
        raise ValueError(
            f"Expected at least one outer fold for run '{run.run_id}', resolved {folds}."
        )
    return folds


def _resolve_selected_index(
    index_csv: Path, run: CompiledRunSpec | CompiledComparisonRunSpec
) -> pd.DataFrame:
    return dataset_selection(
        DatasetSelectionInput(
            index_csv=index_csv,
            target_column=str(run.target),
            cv_mode=str(run.cv_mode),
            subject=str(run.subject) if run.subject is not None else None,
            train_subject=str(run.train_subject) if run.train_subject is not None else None,
            test_subject=str(run.test_subject) if run.test_subject is not None else None,
            filter_task=str(run.filter_task) if run.filter_task is not None else None,
            filter_modality=(str(run.filter_modality) if run.filter_modality is not None else None),
        )
    ).selected_index_df


def _profiling_validity_for_run(
    *,
    index_csv: Path,
    run: CompiledRunSpec | CompiledComparisonRunSpec,
) -> _ProfileRunValidity:
    selected = _resolve_selected_index(index_csv, run)
    cv_mode = str(run.cv_mode)
    expected_outer_folds = _expected_outer_folds(index_csv, run)
    methodology = _coerce_enum(run.methodology_policy_name)
    tuning_enabled = bool(run.tuning_enabled)
    requires_nested_inner_groups = bool(
        methodology == "grouped_nested_tuning"
        and tuning_enabled
        and str(run.model).strip().lower() != "dummy"
    )

    if cv_mode == "frozen_cross_person_transfer":
        train_subject = str(run.train_subject)
        train_rows = selected[selected["subject"].astype(str) == train_subject].copy()
        train_group_count = int(train_rows["session"].astype(str).nunique(dropna=False))
        if requires_nested_inner_groups and train_group_count < 2:
            return _ProfileRunValidity(
                can_profile_measured=False,
                expected_outer_folds=int(expected_outer_folds),
                estimated_train_inner_groups_after_outer_fold=int(train_group_count),
                reason=(
                    "Grouped nested tuning requires at least two inner groups in the training slice "
                    f"for transfer profiling (train_subject='{train_subject}')."
                ),
                profiling_subset_description=(
                    "one train/test pair; one repeat; one outer fold (transfer mode)"
                ),
            )
        return _ProfileRunValidity(
            can_profile_measured=True,
            expected_outer_folds=int(expected_outer_folds),
            estimated_train_inner_groups_after_outer_fold=int(train_group_count),
            reason=None,
            profiling_subset_description=(
                "one train/test pair; one repeat; one outer fold (transfer mode)"
            ),
        )

    if cv_mode == "within_subject_loso_session":
        session_count = int(selected["session"].astype(str).nunique(dropna=False))
        train_groups_after_one_outer_fold = int(max(session_count - 1, 0))
        if requires_nested_inner_groups and train_groups_after_one_outer_fold < 2:
            return _ProfileRunValidity(
                can_profile_measured=False,
                expected_outer_folds=int(expected_outer_folds),
                estimated_train_inner_groups_after_outer_fold=int(
                    train_groups_after_one_outer_fold
                ),
                reason=(
                    "Grouped nested tuning requires at least two inner groups in the training slice "
                    "after holding out one outer fold; this run has insufficient session groups."
                ),
                profiling_subset_description=(
                    "one subject; one repeat; one outer fold (within-subject LOSO)"
                ),
            )
        return _ProfileRunValidity(
            can_profile_measured=True,
            expected_outer_folds=int(expected_outer_folds),
            estimated_train_inner_groups_after_outer_fold=int(train_groups_after_one_outer_fold),
            reason=None,
            profiling_subset_description=(
                "one subject; one repeat; one outer fold (within-subject LOSO)"
            ),
        )

    return _ProfileRunValidity(
        can_profile_measured=True,
        expected_outer_folds=int(expected_outer_folds),
        estimated_train_inner_groups_after_outer_fold=None,
        reason=None,
        profiling_subset_description="one representative run; one repeat; one outer fold",
    )


def _minimal_valid_profile_subset(
    *,
    records: list[_PlannedProfileRun],
    index_csv: Path,
) -> tuple[_PlannedProfileRun, _ProfileRunValidity]:
    scored_candidates: list[tuple[int, int, str, str, _PlannedProfileRun, _ProfileRunValidity]] = []
    fallback_candidates: list[tuple[str, str, _PlannedProfileRun, _ProfileRunValidity]] = []
    for record in records:
        validity = _profiling_validity_for_run(index_csv=index_csv, run=record.run)
        if validity.can_profile_measured:
            scored_candidates.append(
                (
                    int(validity.expected_outer_folds),
                    int(record.run.repeat_id),
                    str(record.run.base_run_id),
                    str(record.run.run_id),
                    record,
                    validity,
                )
            )
        else:
            fallback_candidates.append(
                (str(record.run.base_run_id), str(record.run.run_id), record, validity)
            )
    if scored_candidates:
        scored_candidates.sort(key=lambda item: item[:4])
        _, _, _, _, chosen_record, chosen_validity = scored_candidates[0]
        return chosen_record, chosen_validity

    if fallback_candidates:
        fallback_candidates.sort(key=lambda item: item[:2])
        _, _, chosen_record, chosen_validity = fallback_candidates[0]
        return chosen_record, chosen_validity

    raise ValueError("No runs available to select a profiling subset.")


def _profile_run_id(*, phase: str, cohort_id: str) -> str:
    return f"profile_{phase}_{cohort_id}"


def _build_planned_runs(
    *,
    index_csv: Path,
    confirmatory_protocol: Path,
    comparison_specs: list[Path],
) -> tuple[list[_PlannedProfileRun], list[dict[str, Any]]]:
    issues: list[dict[str, Any]] = []
    planned: list[_PlannedProfileRun] = []

    try:
        protocol = load_protocol(confirmatory_protocol)
        protocol_manifest = compile_protocol(protocol, index_csv=index_csv)
        evidence_policy_payload = protocol_manifest.evidence_policy.model_dump(mode="json")
        data_policy_payload = protocol_manifest.data_policy.model_dump(mode="json")
        for run in protocol_manifest.runs:
            planned.append(
                _PlannedProfileRun(
                    phase="confirmatory",
                    source_id=protocol_manifest.protocol_id,
                    source_version=protocol_manifest.protocol_version,
                    run=run,
                    evidence_policy=dict(evidence_policy_payload),
                    data_policy=dict(data_policy_payload),
                )
            )
    except Exception as exc:
        issues.append(
            _issue(
                "confirmatory_compile_failed",
                "Failed to compile confirmatory protocol for runtime profiling precheck.",
                details={
                    "confirmatory_protocol": str(confirmatory_protocol),
                    "error": str(exc),
                },
            )
        )

    for comparison_path in comparison_specs:
        try:
            comparison = load_comparison_spec(comparison_path)
            comparison_manifest = compile_comparison(comparison, index_csv=index_csv)
            evidence_policy_payload = comparison_manifest.evidence_policy.model_dump(mode="json")
            data_policy_payload = comparison_manifest.data_policy.model_dump(mode="json")
            for run in comparison_manifest.runs:
                planned.append(
                    _PlannedProfileRun(
                        phase="comparison",
                        source_id=comparison_manifest.comparison_id,
                        source_version=comparison_manifest.comparison_version,
                        run=run,
                        evidence_policy=dict(evidence_policy_payload),
                        data_policy=dict(data_policy_payload),
                    )
                )
        except Exception as exc:
            issues.append(
                _issue(
                    "comparison_compile_failed",
                    "Failed to compile comparison spec for runtime profiling precheck.",
                    details={
                        "comparison_spec": str(comparison_path),
                        "error": str(exc),
                    },
                )
            )

    return planned, issues


def _aggregate_warnings_and_recommendations(
    *,
    estimated_total_seconds: float,
    phase_totals: dict[str, float],
    model_totals: dict[str, float],
    profiling_runs_executed: int,
    fallback_estimates_used: int,
    cohort_count: int,
    issues: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    warnings: list[dict[str, Any]] = []
    recommendations: list[dict[str, Any]] = []
    if issues:
        warnings.append(
            _issue(
                "profiling_issues_present",
                "Runtime profiling encountered one or more failures.",
                details={"n_issues": int(len(issues))},
            )
        )
    if profiling_runs_executed < cohort_count:
        warnings.append(
            _issue(
                "profiling_incomplete",
                "Not all runtime cohorts were successfully profiled.",
                details={
                    "profiling_runs_executed": int(profiling_runs_executed),
                    "cohort_count": int(cohort_count),
                },
            )
        )
    if fallback_estimates_used > 0:
        warnings.append(
            _issue(
                "profiling_fallback_used",
                "One or more cohorts used conservative fallback runtime estimates.",
                details={"fallback_estimates_used": int(fallback_estimates_used)},
            )
        )
    if estimated_total_seconds > (8 * 3600):
        warnings.append(
            _issue(
                "estimated_runtime_exceeds_8h",
                "Estimated total campaign wall time exceeds 8 hours.",
                details={
                    "estimated_total_wall_time_seconds": float(estimated_total_seconds),
                    "estimated_total_wall_time_human": _humanize_seconds(estimated_total_seconds),
                },
            )
        )
    if estimated_total_seconds > (24 * 3600):
        warnings.append(
            _issue(
                "estimated_runtime_exceeds_24h",
                "Estimated total campaign wall time exceeds 24 hours.",
                details={
                    "estimated_total_wall_time_seconds": float(estimated_total_seconds),
                    "estimated_total_wall_time_human": _humanize_seconds(estimated_total_seconds),
                },
            )
        )
        recommendations.append(
            _issue(
                "split_campaign_phases",
                "Run confirmatory and comparison phases separately to avoid long single-session execution.",
            )
        )

    confirmatory_total = float(phase_totals.get("confirmatory", 0.0))
    comparison_total = float(phase_totals.get("comparison", 0.0))
    if comparison_total > 0.0 and comparison_total >= (2.0 * max(confirmatory_total, 1.0)):
        recommendations.append(
            _issue(
                "run_confirmatory_before_comparison",
                "Comparison runtime dominates estimated wall time; run confirmatory phase first for earlier evidence completion.",
                details={
                    "comparison_estimated_seconds": comparison_total,
                    "confirmatory_estimated_seconds": confirmatory_total,
                },
            )
        )

    if model_totals and estimated_total_seconds > 0.0:
        model_name, model_seconds = max(model_totals.items(), key=lambda item: float(item[1]))
        share = float(model_seconds) / float(estimated_total_seconds)
        if share >= 0.5:
            recommendations.append(
                _issue(
                    "dominant_model_runtime",
                    "A single model dominates estimated runtime; consider isolating it in a separate comparison pass.",
                    details={
                        "model": str(model_name),
                        "estimated_seconds": float(model_seconds),
                        "share_of_total": float(round(share, 4)),
                    },
                )
            )

    return warnings, recommendations


def verify_campaign_runtime_profile(
    *,
    index_csv: Path | str,
    data_root: Path | str,
    cache_dir: Path | str,
    confirmatory_protocol: Path | str,
    comparison_specs: list[Path | str],
    profile_root: Path | str,
    hardware_mode: str = "cpu_only",
    gpu_device_id: int | None = None,
    deterministic_compute: bool = False,
    allow_backend_fallback: bool = False,
    profile_permutations: int | None = None,
    profile_inner_folds: int | None = None,
    profile_tuning_candidates: int | None = None,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    index_csv_path = Path(index_csv).resolve()
    data_root_path = Path(data_root).resolve()
    cache_dir_path = Path(cache_dir).resolve()
    confirmatory_protocol_path = Path(confirmatory_protocol).resolve()
    comparison_spec_paths = [Path(path).resolve() for path in comparison_specs]
    profile_root_path = Path(profile_root).resolve()
    profile_root_path.mkdir(parents=True, exist_ok=True)
    resolved_profile_permutations: int | None = (
        int(profile_permutations) if profile_permutations is not None else None
    )
    resolved_profile_inner_folds: int | None = (
        int(profile_inner_folds) if profile_inner_folds is not None else None
    )
    resolved_profile_tuning_candidates: int | None = (
        int(profile_tuning_candidates) if profile_tuning_candidates is not None else None
    )
    if resolved_profile_permutations is not None and resolved_profile_permutations < 0:
        raise ValueError("profile_permutations must be >= 0 when provided.")
    if resolved_profile_inner_folds is not None and resolved_profile_inner_folds < 0:
        raise ValueError("profile_inner_folds must be >= 0 when provided.")
    if resolved_profile_tuning_candidates is not None and resolved_profile_tuning_candidates < 0:
        raise ValueError("profile_tuning_candidates must be >= 0 when provided.")
    feature_matrix_memoizer = _ProfilingFeatureMatrixMemoizer()

    planned_runs, compile_issues = _build_planned_runs(
        index_csv=index_csv_path,
        confirmatory_protocol=confirmatory_protocol_path,
        comparison_specs=comparison_spec_paths,
    )
    issues: list[dict[str, Any]] = list(compile_issues)

    cohorts: dict[str, list[_PlannedProfileRun]] = {}
    cohort_descriptors: dict[str, dict[str, Any]] = {}
    for record in planned_runs:
        descriptor = _cohort_descriptor(record)
        key = _cohort_key(descriptor)
        cohorts.setdefault(key, []).append(record)
        cohort_descriptors[key] = descriptor

    total_cohorts = int(len(cohorts))
    completed_cohorts = 0

    emit_progress(
        progress_callback,
        stage="campaign",
        message="compiled campaign runtime profile plan",
        completed_units=0.0,
        total_units=float(total_cohorts),
        metadata={
            "n_planned_runs": int(len(planned_runs)),
            "n_cohorts": int(total_cohorts),
        },
    )

    cohort_estimates: list[dict[str, Any]] = []
    phase_totals: dict[str, float] = {"confirmatory": 0.0, "comparison": 0.0}
    phase_runs: dict[str, int] = {"confirmatory": 0, "comparison": 0}
    model_totals: dict[str, float] = {}
    profile_artifact_paths: list[str] = []
    profiling_runs_executed = 0
    fallback_estimates_used = 0

    for cohort_index, cohort_key in enumerate(sorted(cohorts.keys()), start=1):
        cohort_runs = cohorts[cohort_key]
        representative, profile_validity = _minimal_valid_profile_subset(
            records=cohort_runs,
            index_csv=index_csv_path,
        )
        run = representative.run
        descriptor = cohort_descriptors[cohort_key]
        digest = sha1(cohort_key.encode("utf-8")).hexdigest()[:10]
        cohort_id = f"cohort_{cohort_index:03d}_{digest}"
        run_id = _profile_run_id(phase=representative.phase, cohort_id=cohort_id)
        cohort_reports_root = profile_root_path / representative.phase / cohort_id
        cohort_reports_root.mkdir(parents=True, exist_ok=True)

        emit_progress(
            progress_callback,
            stage="campaign",
            message=f"starting cohort {cohort_index}/{total_cohorts}",
            completed_units=float(completed_cohorts),
            total_units=float(total_cohorts),
            metadata={
                "phase": str(representative.phase),
                "cohort_id": str(cohort_id),
                "cohort_index": int(cohort_index),
                "n_cohorts": int(total_cohorts),
                "model": str(run.model),
                "target": str(run.target),
                "source_run_id": str(run.run_id),
            },
        )
        expected_outer_folds = 0
        measured_outer_folds = 0
        elapsed_seconds = 0.0
        run_result: dict[str, Any] | None = None
        run_error: str | None = None
        configured_n_permutations = (
            int(run.controls.n_permutations) if bool(run.controls.permutation_enabled) else 0
        )
        methodology_name = _coerce_enum(run.methodology_policy_name)
        grouped_nested_tuning_active = bool(
            methodology_name == "grouped_nested_tuning"
            and bool(run.tuning_enabled)
            and str(run.model).strip().lower() != "dummy"
        )
        configured_inner_fold_count: int | None = (
            int(profile_validity.estimated_train_inner_groups_after_outer_fold)
            if grouped_nested_tuning_active
            and profile_validity.estimated_train_inner_groups_after_outer_fold is not None
            else None
        )
        configured_candidate_count: int | None = None
        if grouped_nested_tuning_active:
            if not run.tuning_search_space_id:
                raise ValueError(
                    "Grouped nested runtime profiling requires tuning_search_space_id for tuned runs."
                )
            _, runtime_profile_param_grid = get_search_space(
                str(run.tuning_search_space_id),
                str(run.model),
            )
            configured_candidate_count = int(len(list(ParameterGrid(runtime_profile_param_grid))))
        profiled_inner_fold_count: int | None = configured_inner_fold_count
        if configured_inner_fold_count is not None and resolved_profile_inner_folds is not None:
            if resolved_profile_inner_folds <= 0:
                raise ValueError(
                    "profile_inner_folds must be > 0 when grouped nested tuning is active "
                    "for profiled cohorts."
                )
            profiled_inner_fold_count = int(
                min(configured_inner_fold_count, resolved_profile_inner_folds)
            )
        profiled_candidate_count: int | None = configured_candidate_count
        if (
            configured_candidate_count is not None
            and resolved_profile_tuning_candidates is not None
        ):
            if resolved_profile_tuning_candidates <= 0:
                raise ValueError(
                    "profile_tuning_candidates must be > 0 when grouped nested tuning is active "
                    "for profiled cohorts."
                )
            profiled_candidate_count = int(
                min(configured_candidate_count, resolved_profile_tuning_candidates)
            )

        expected_outer_folds = int(profile_validity.expected_outer_folds)
        if not profile_validity.can_profile_measured:
            fallback_estimates_used += 1
            estimated_seconds_per_full_run = float(int(run.projected_runtime_seconds))
            estimated_total_seconds = float(estimated_seconds_per_full_run) * float(
                len(cohort_runs)
            )
            phase_totals[str(representative.phase)] += float(estimated_total_seconds)
            phase_runs[str(representative.phase)] += int(len(cohort_runs))
            model_name = str(run.model)
            model_totals[model_name] = float(model_totals.get(model_name, 0.0)) + float(
                estimated_total_seconds
            )
            cohort_estimates.append(
                {
                    "status": "passed",
                    "cohort_id": cohort_id,
                    "cohort_key": cohort_key,
                    "cohort": descriptor,
                    "phase": representative.phase,
                    "source_run_id": run.run_id,
                    "source_framework_mode": str(run.framework_mode),
                    "n_planned_runs": int(len(cohort_runs)),
                    "estimate_source": "conservative_fallback",
                    "estimate_confidence": "low",
                    "profiling_subset_description": profile_validity.profiling_subset_description,
                    "expected_outer_folds_per_run": int(expected_outer_folds),
                    "measured_outer_folds": None,
                    "configured_n_permutations": int(configured_n_permutations),
                    "profiled_n_permutations": None,
                    "permutation_loop_measured_seconds": None,
                    "estimated_full_permutation_seconds": None,
                    "permutation_extrapolation_applied": False,
                    "configured_inner_fold_count": (
                        int(configured_inner_fold_count)
                        if configured_inner_fold_count is not None
                        else None
                    ),
                    "profiled_inner_fold_count": (
                        int(profiled_inner_fold_count)
                        if profiled_inner_fold_count is not None
                        else None
                    ),
                    "configured_candidate_count": (
                        int(configured_candidate_count)
                        if configured_candidate_count is not None
                        else None
                    ),
                    "profiled_candidate_count": (
                        int(profiled_candidate_count)
                        if profiled_candidate_count is not None
                        else None
                    ),
                    "measured_inner_tuning_seconds": None,
                    "estimated_full_tuning_seconds": None,
                    "tuning_extrapolation_applied": False,
                    "specialized_linearsvc_tuning_used": None,
                    "specialized_logreg_tuning_used": None,
                    "profile_elapsed_seconds": None,
                    "fold_scale_factor": None,
                    "estimated_seconds_per_full_run": float(estimated_seconds_per_full_run),
                    "estimated_total_seconds": float(estimated_total_seconds),
                    "estimated_total_human": _humanize_seconds(estimated_total_seconds),
                    "projected_runtime_seconds_per_run": int(run.projected_runtime_seconds),
                    "feature_matrix_cache_hit": False,
                    "feature_matrix_cache_key": None,
                    "feature_matrix_cache_sample_signature": None,
                    "fallback_reason": str(profile_validity.reason or "profiling_subset_invalid"),
                    "estimated_train_inner_groups_after_outer_fold": (
                        int(profile_validity.estimated_train_inner_groups_after_outer_fold)
                        if profile_validity.estimated_train_inner_groups_after_outer_fold
                        is not None
                        else None
                    ),
                }
            )
            completed_cohorts += 1
            emit_progress(
                progress_callback,
                stage="campaign",
                message=f"used fallback estimate for cohort {cohort_index}/{total_cohorts}",
                completed_units=float(completed_cohorts),
                total_units=float(total_cohorts),
                metadata={
                    "phase": str(representative.phase),
                    "cohort_id": str(cohort_id),
                    "cohort_index": int(cohort_index),
                    "n_cohorts": int(total_cohorts),
                    "model": str(run.model),
                    "target": str(run.target),
                    "source_run_id": str(run.run_id),
                    "estimate_source": "conservative_fallback",
                    "fallback_reason": str(profile_validity.reason or "profiling_subset_invalid"),
                },
            )
            continue

        profiled_n_permutations = int(configured_n_permutations)
        if configured_n_permutations > 0 and resolved_profile_permutations is not None:
            if resolved_profile_permutations <= 0:
                raise ValueError(
                    "profile_permutations must be > 0 when permutation controls are enabled "
                    "for profiled cohorts."
                )
            profiled_n_permutations = int(
                min(configured_n_permutations, resolved_profile_permutations)
            )
        permutation_loop_measured_seconds: float | None = None
        estimated_full_permutation_seconds: float | None = None
        permutation_extrapolation_applied = False
        measured_inner_tuning_seconds: float | None = None
        estimated_full_tuning_seconds: float | None = None
        tuning_extrapolation_applied = False
        specialized_linearsvc_tuning_used: bool | None = None
        specialized_logreg_tuning_used: bool | None = None
        feature_matrix_cache_hit = False
        feature_matrix_cache_key: str | None = None
        feature_matrix_cache_sample_signature: str | None = None

        try:
            previous_cache_hits = int(feature_matrix_memoizer.hits)
            started = perf_counter()
            run_result = run_experiment(
                index_csv=index_csv_path,
                data_root=data_root_path,
                cache_dir=cache_dir_path,
                target=str(run.target),
                model=str(run.model),
                cv=str(run.cv_mode),
                subject=str(run.subject) if run.subject is not None else None,
                train_subject=str(run.train_subject) if run.train_subject is not None else None,
                test_subject=str(run.test_subject) if run.test_subject is not None else None,
                seed=int(run.seed),
                filter_task=str(run.filter_task) if run.filter_task is not None else None,
                filter_modality=str(run.filter_modality)
                if run.filter_modality is not None
                else None,
                n_permutations=int(profiled_n_permutations),
                primary_metric_name=str(run.primary_metric),
                permutation_metric_name=(
                    str(run.controls.permutation_metric)
                    if run.controls.permutation_metric is not None
                    else str(run.primary_metric)
                ),
                repeat_id=1,
                repeat_count=1,
                base_run_id=str(run.base_run_id),
                evidence_run_role=_coerce_enum(run.evidence_run_role),
                evidence_policy=dict(representative.evidence_policy),
                data_policy=dict(representative.data_policy),
                methodology_policy_name=_coerce_enum(run.methodology_policy_name),
                class_weight_policy=_coerce_enum(run.class_weight_policy),
                tuning_enabled=bool(run.tuning_enabled),
                tuning_search_space_id=run.tuning_search_space_id,
                tuning_search_space_version=run.tuning_search_space_version,
                tuning_inner_cv_scheme=run.tuning_inner_cv_scheme,
                tuning_inner_group_field=run.tuning_inner_group_field,
                interpretability_enabled_override=bool(run.interpretability_enabled),
                framework_mode=FrameworkMode.EXPLORATORY,
                run_id=run_id,
                reports_root=cohort_reports_root,
                force=True,
                resume=False,
                reuse_completed_artifacts=False,
                model_cost_tier=_coerce_enum(run.model_cost_tier),
                projected_runtime_seconds=int(run.projected_runtime_seconds),
                profiling_context={
                    "source": "campaign_runtime_profile_precheck",
                    "profiling_only": True,
                    "precheck_only": True,
                    "max_outer_folds": 1,
                    "source_framework_mode": str(run.framework_mode),
                    "source_phase": str(representative.phase),
                    "source_run_id": str(run.run_id),
                    "configured_n_permutations": int(configured_n_permutations),
                    "profiled_n_permutations": int(profiled_n_permutations),
                    "configured_inner_fold_count": (
                        int(configured_inner_fold_count)
                        if configured_inner_fold_count is not None
                        else None
                    ),
                    "profiled_inner_fold_count": (
                        int(profiled_inner_fold_count)
                        if profiled_inner_fold_count is not None
                        else None
                    ),
                    "configured_candidate_count": (
                        int(configured_candidate_count)
                        if configured_candidate_count is not None
                        else None
                    ),
                    "profiled_candidate_count": (
                        int(profiled_candidate_count)
                        if profiled_candidate_count is not None
                        else None
                    ),
                    "profile_inner_folds": (
                        int(profiled_inner_fold_count)
                        if profiled_inner_fold_count is not None
                        else None
                    ),
                    "profile_tuning_candidates": (
                        int(profiled_candidate_count)
                        if profiled_candidate_count is not None
                        else None
                    ),
                },
                hardware_mode=str(hardware_mode),
                gpu_device_id=(int(gpu_device_id) if gpu_device_id is not None else None),
                deterministic_compute=bool(deterministic_compute),
                allow_backend_fallback=bool(allow_backend_fallback),
                progress_callback=progress_callback,
                load_features_from_cache_fn_override=feature_matrix_memoizer.load,
            )
            elapsed_seconds = float(perf_counter() - started)
            feature_matrix_cache_hit = int(feature_matrix_memoizer.hits) > int(previous_cache_hits)
            feature_matrix_cache_key = feature_matrix_memoizer.last_lookup_key
            feature_matrix_cache_sample_signature = (
                feature_matrix_memoizer.last_lookup_sample_signature
            )

            profiling_runs_executed += 1
            if isinstance(run_result, dict):
                stage_timings = run_result.get("stage_timings_seconds")
                if isinstance(stage_timings, dict) and "total" in stage_timings:
                    elapsed_seconds = float(stage_timings["total"])
                metrics_payload = run_result.get("metrics")
                if isinstance(metrics_payload, dict):
                    measured_outer_folds = int(metrics_payload.get("n_folds", 0))
                    permutation_payload = metrics_payload.get("permutation_test")
                    if isinstance(permutation_payload, dict):
                        measured_loop = permutation_payload.get("permutation_loop_seconds")
                        if isinstance(measured_loop, (int, float)):
                            permutation_loop_measured_seconds = float(measured_loop)
                report_dir = run_result.get("report_dir")
                if isinstance(report_dir, str) and report_dir:
                    profile_artifact_paths.append(report_dir)
                run_status_path = run_result.get("run_status_path")
                if isinstance(run_status_path, str) and run_status_path:
                    profile_artifact_paths.append(run_status_path)
                tuning_summary_path = run_result.get("tuning_summary_path")
                if isinstance(tuning_summary_path, str) and tuning_summary_path:
                    tuning_summary_candidate = Path(tuning_summary_path)
                    if tuning_summary_candidate.exists():
                        tuning_summary_payload = json.loads(
                            tuning_summary_candidate.read_text(encoding="utf-8")
                        )
                        timing_totals_payload = tuning_summary_payload.get("timing_totals_seconds")
                        if isinstance(timing_totals_payload, dict):
                            measured_inner_value = timing_totals_payload.get(
                                "measured_inner_tuning_total"
                            )
                            if isinstance(measured_inner_value, (int, float)):
                                measured_inner_tuning_seconds = float(measured_inner_value)
                            estimated_tuning_value = timing_totals_payload.get(
                                "estimated_full_tuned_search_total"
                            )
                            if isinstance(estimated_tuning_value, (int, float)):
                                estimated_full_tuning_seconds = float(estimated_tuning_value)
                        specialized_flag = tuning_summary_payload.get(
                            "specialized_linearsvc_tuning_used"
                        )
                        if isinstance(specialized_flag, bool):
                            specialized_linearsvc_tuning_used = bool(specialized_flag)
                        specialized_logreg_flag = tuning_summary_payload.get(
                            "specialized_logreg_tuning_used"
                        )
                        if isinstance(specialized_logreg_flag, bool):
                            specialized_logreg_tuning_used = bool(specialized_logreg_flag)
                        tuning_extrapolation_flag = tuning_summary_payload.get(
                            "tuning_extrapolation_applied"
                        )
                        if isinstance(tuning_extrapolation_flag, bool):
                            tuning_extrapolation_applied = bool(tuning_extrapolation_flag)
            if measured_outer_folds <= 0:
                measured_outer_folds = 1
        except Exception as exc:
            run_error = str(exc)
            issues.append(
                _issue(
                    "profiling_run_failed",
                    "Runtime profiling failed for cohort representative run.",
                    details={
                        "cohort_id": cohort_id,
                        "phase": representative.phase,
                        "source_run_id": run.run_id,
                        "error": str(exc),
                    },
                )
            )

        if run_error is not None:
            cohort_estimates.append(
                {
                    "status": "failed",
                    "cohort_id": cohort_id,
                    "cohort_key": cohort_key,
                    "cohort": descriptor,
                    "phase": representative.phase,
                    "source_run_id": run.run_id,
                    "n_planned_runs": int(len(cohort_runs)),
                    "error": run_error,
                }
            )
            continue

        if expected_outer_folds <= 0:
            issues.append(
                _issue(
                    "invalid_expected_outer_folds",
                    "Resolved expected outer fold count must be positive.",
                    details={
                        "cohort_id": cohort_id,
                        "source_run_id": run.run_id,
                        "expected_outer_folds": expected_outer_folds,
                    },
                )
            )
            cohort_estimates.append(
                {
                    "status": "failed",
                    "cohort_id": cohort_id,
                    "cohort_key": cohort_key,
                    "cohort": descriptor,
                    "phase": representative.phase,
                    "source_run_id": run.run_id,
                    "n_planned_runs": int(len(cohort_runs)),
                    "error": "invalid_expected_outer_folds",
                }
            )
            continue

        adjusted_elapsed_seconds = float(elapsed_seconds)
        if (
            configured_n_permutations > 0
            and profiled_n_permutations > 0
            and permutation_loop_measured_seconds is not None
        ):
            estimated_full_permutation_seconds = (
                float(permutation_loop_measured_seconds)
                * float(configured_n_permutations)
                / float(profiled_n_permutations)
            )
            permutation_extrapolation_applied = int(profiled_n_permutations) != int(
                configured_n_permutations
            )
            adjusted_elapsed_seconds = max(
                0.0,
                float(elapsed_seconds)
                - float(permutation_loop_measured_seconds)
                + float(estimated_full_permutation_seconds),
            )
        if (
            measured_inner_tuning_seconds is not None
            and estimated_full_tuning_seconds is not None
            and measured_inner_tuning_seconds >= 0.0
        ):
            tuning_extrapolation_applied = bool(
                tuning_extrapolation_applied
                or abs(float(estimated_full_tuning_seconds) - float(measured_inner_tuning_seconds))
                > 1e-12
            )
            adjusted_elapsed_seconds = max(
                0.0,
                float(adjusted_elapsed_seconds)
                - float(measured_inner_tuning_seconds)
                + float(estimated_full_tuning_seconds),
            )

        fold_scale_factor = float(expected_outer_folds) / float(max(measured_outer_folds, 1))
        estimated_seconds_per_full_run = float(adjusted_elapsed_seconds) * float(fold_scale_factor)
        estimated_total_seconds = float(estimated_seconds_per_full_run) * float(len(cohort_runs))
        phase_totals[str(representative.phase)] += float(estimated_total_seconds)
        phase_runs[str(representative.phase)] += int(len(cohort_runs))
        model_name = str(run.model)
        model_totals[model_name] = float(model_totals.get(model_name, 0.0)) + float(
            estimated_total_seconds
        )

        completed_cohorts += 1
        emit_progress(
            progress_callback,
            stage="campaign",
            message=f"finished cohort {cohort_index}/{total_cohorts}",
            completed_units=float(completed_cohorts),
            total_units=float(total_cohorts),
            metadata={
                "phase": str(representative.phase),
                "cohort_id": str(cohort_id),
                "cohort_index": int(cohort_index),
                "n_cohorts": int(total_cohorts),
                "model": str(run.model),
                "target": str(run.target),
                "source_run_id": str(run.run_id),
                "estimate_source": "measured_profile",
                "profile_elapsed_seconds": float(elapsed_seconds),
                "configured_n_permutations": int(configured_n_permutations),
                "profiled_n_permutations": int(profiled_n_permutations),
                "permutation_extrapolation_applied": bool(permutation_extrapolation_applied),
                "configured_inner_fold_count": (
                    int(configured_inner_fold_count)
                    if configured_inner_fold_count is not None
                    else None
                ),
                "profiled_inner_fold_count": (
                    int(profiled_inner_fold_count)
                    if profiled_inner_fold_count is not None
                    else None
                ),
                "configured_candidate_count": (
                    int(configured_candidate_count)
                    if configured_candidate_count is not None
                    else None
                ),
                "profiled_candidate_count": (
                    int(profiled_candidate_count) if profiled_candidate_count is not None else None
                ),
                "tuning_extrapolation_applied": bool(tuning_extrapolation_applied),
            },
        )

        cohort_estimates.append(
            {
                "status": "passed",
                "cohort_id": cohort_id,
                "cohort_key": cohort_key,
                "cohort": descriptor,
                "phase": representative.phase,
                "source_run_id": run.run_id,
                "source_framework_mode": str(run.framework_mode),
                "n_planned_runs": int(len(cohort_runs)),
                "estimate_source": "measured_profile",
                "estimate_confidence": "medium",
                "profiling_subset_description": profile_validity.profiling_subset_description,
                "expected_outer_folds_per_run": int(expected_outer_folds),
                "measured_outer_folds": int(measured_outer_folds),
                "configured_n_permutations": int(configured_n_permutations),
                "profiled_n_permutations": int(profiled_n_permutations),
                "permutation_loop_measured_seconds": (
                    float(permutation_loop_measured_seconds)
                    if permutation_loop_measured_seconds is not None
                    else None
                ),
                "estimated_full_permutation_seconds": (
                    float(estimated_full_permutation_seconds)
                    if estimated_full_permutation_seconds is not None
                    else None
                ),
                "permutation_extrapolation_applied": bool(permutation_extrapolation_applied),
                "configured_inner_fold_count": (
                    int(configured_inner_fold_count)
                    if configured_inner_fold_count is not None
                    else None
                ),
                "profiled_inner_fold_count": (
                    int(profiled_inner_fold_count)
                    if profiled_inner_fold_count is not None
                    else None
                ),
                "configured_candidate_count": (
                    int(configured_candidate_count)
                    if configured_candidate_count is not None
                    else None
                ),
                "profiled_candidate_count": (
                    int(profiled_candidate_count) if profiled_candidate_count is not None else None
                ),
                "measured_inner_tuning_seconds": (
                    float(measured_inner_tuning_seconds)
                    if measured_inner_tuning_seconds is not None
                    else None
                ),
                "estimated_full_tuning_seconds": (
                    float(estimated_full_tuning_seconds)
                    if estimated_full_tuning_seconds is not None
                    else None
                ),
                "tuning_extrapolation_applied": bool(tuning_extrapolation_applied),
                "specialized_linearsvc_tuning_used": specialized_linearsvc_tuning_used,
                "specialized_logreg_tuning_used": specialized_logreg_tuning_used,
                "profile_elapsed_seconds": float(elapsed_seconds),
                "fold_scale_factor": float(fold_scale_factor),
                "estimated_seconds_per_full_run": float(estimated_seconds_per_full_run),
                "estimated_total_seconds": float(estimated_total_seconds),
                "estimated_total_human": _humanize_seconds(estimated_total_seconds),
                "feature_matrix_cache_hit": bool(feature_matrix_cache_hit),
                "feature_matrix_cache_key": feature_matrix_cache_key,
                "feature_matrix_cache_sample_signature": feature_matrix_cache_sample_signature,
                "profile_reports_root": str(cohort_reports_root),
                "profile_run_id": run_id,
                "profile_report_dir": (
                    str(run_result.get("report_dir"))
                    if isinstance(run_result, dict) and run_result.get("report_dir")
                    else None
                ),
                "profile_run_status_path": (
                    str(run_result.get("run_status_path"))
                    if isinstance(run_result, dict) and run_result.get("run_status_path")
                    else None
                ),
            }
        )

    estimated_total_wall_time_seconds = float(sum(phase_totals.values()))
    phase_estimates: dict[str, dict[str, Any]] = {}
    for phase_name in ("confirmatory", "comparison"):
        seconds = float(phase_totals.get(phase_name, 0.0))
        phase_estimates[phase_name] = {
            "estimated_total_seconds": seconds,
            "estimated_total_human": _humanize_seconds(seconds),
            "n_planned_runs": int(phase_runs.get(phase_name, 0)),
            "n_cohorts": int(
                sum(
                    1
                    for item in cohort_estimates
                    if str(item.get("status")) == "passed" and str(item.get("phase")) == phase_name
                )
            ),
        }

    model_estimates = {
        model: {
            "estimated_total_seconds": float(seconds),
            "estimated_total_human": _humanize_seconds(float(seconds)),
        }
        for model, seconds in sorted(
            model_totals.items(),
            key=lambda item: float(item[1]),
            reverse=True,
        )
    }

    warnings, recommendations = _aggregate_warnings_and_recommendations(
        estimated_total_seconds=estimated_total_wall_time_seconds,
        phase_totals=phase_totals,
        model_totals=model_totals,
        profiling_runs_executed=profiling_runs_executed,
        fallback_estimates_used=fallback_estimates_used,
        cohort_count=len(cohorts),
        issues=issues,
    )

    passed = bool(
        not issues
        and len(cohorts) > 0
        and (profiling_runs_executed > 0 or fallback_estimates_used > 0)
    )

    emit_progress(
        progress_callback,
        stage="campaign",
        message="finished campaign runtime profiling",
        completed_units=float(total_cohorts),
        total_units=float(total_cohorts),
        metadata={
            "n_cohorts": int(total_cohorts),
            "profiling_runs_executed": int(profiling_runs_executed),
            "fallback_estimates_used": int(fallback_estimates_used),
        },
    )

    return {
        "schema_version": "campaign-runtime-profile-summary-v1",
        "generated_at_utc": _utc_now(),
        "passed": passed,
        "inputs": {
            "index_csv": str(index_csv_path),
            "data_root": str(data_root_path),
            "cache_dir": str(cache_dir_path),
            "confirmatory_protocol": str(confirmatory_protocol_path),
            "comparison_specs": [str(path) for path in comparison_spec_paths],
            "profile_root": str(profile_root_path),
            "profile_permutations_override": (
                int(resolved_profile_permutations)
                if resolved_profile_permutations is not None
                else None
            ),
            "profile_inner_folds_override": (
                int(resolved_profile_inner_folds)
                if resolved_profile_inner_folds is not None
                else None
            ),
            "profile_tuning_candidates_override": (
                int(resolved_profile_tuning_candidates)
                if resolved_profile_tuning_candidates is not None
                else None
            ),
        },
        "profiling_runs_executed": int(profiling_runs_executed),
        "fallback_estimates_used": int(fallback_estimates_used),
        "n_cohorts": int(len(cohorts)),
        "cohort_estimates": cohort_estimates,
        "phase_estimates": phase_estimates,
        "model_estimates": model_estimates,
        "estimated_total_wall_time_seconds": float(estimated_total_wall_time_seconds),
        "estimated_total_wall_time_human": _humanize_seconds(estimated_total_wall_time_seconds),
        "warnings": warnings,
        "recommendations": recommendations,
        "assumptions": {
            "profiling_subset": {
                "max_outer_folds": 1,
                "representative_run_per_cohort": True,
                "single_repeat_per_profiled_run": True,
            },
            "scaling_rules": {
                "outer_fold_scaling": "linear",
                "compiled_run_count_scaling": "linear",
            },
            "permutation_profiling": {
                "profile_permutations_override": (
                    int(resolved_profile_permutations)
                    if resolved_profile_permutations is not None
                    else None
                ),
                "full_permutation_extrapolation": "linear_when_permutation_metrics_available",
            },
            "nested_tuning_profiling": {
                "profile_inner_folds_override": (
                    int(resolved_profile_inner_folds)
                    if resolved_profile_inner_folds is not None
                    else None
                ),
                "profile_tuning_candidates_override": (
                    int(resolved_profile_tuning_candidates)
                    if resolved_profile_tuning_candidates is not None
                    else None
                ),
                "full_tuning_extrapolation": "linear_when_tuning_timing_metadata_available",
            },
            "notes": [
                "Profiling runs are precheck-only and non-canonical.",
                "Estimated totals are projections from representative cohort runs.",
            ],
        },
        "feature_matrix_memoization": feature_matrix_memoizer.summary_payload(),
        "profile_artifact_paths": sorted(set(profile_artifact_paths)),
        "issues": issues,
    }


__all__ = ["verify_campaign_runtime_profile"]
