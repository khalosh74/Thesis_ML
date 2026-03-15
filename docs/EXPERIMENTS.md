# Experiments Guide (Framework Modes)

This framework now has three hard-separated execution modes:

- exploratory experiments
- locked comparison experiments
- confirmatory thesis runs

Mode identity is explicit in run artifacts via `framework_mode` and enforced by runner guardrails.

## Mode commands and intent

Exploratory mode (`thesisml-run-experiment`):
- purpose: ad hoc idea generation, debugging, flexible probing
- default reports root: `outputs/reports/exploratory/`
- metadata: `framework_mode=exploratory`, `canonical_run=false`
- science-affecting CLI flags are allowed

Locked comparison mode (`thesisml-run-comparison`):
- purpose: predeclared method-lock comparisons from registered specs
- default reports root: `outputs/reports/comparisons/`
- metadata: `framework_mode=locked_comparison`, `canonical_run=false`
- science-affecting settings are frozen in comparison JSON

Confirmatory mode (`thesisml-run-protocol`):
- purpose: final thesis evidence only
- default reports root: `outputs/reports/confirmatory/`
- metadata: `framework_mode=confirmatory`, `canonical_run=true`
- science-affecting settings come only from canonical protocol JSON

## Exploratory command

```bash
thesisml-run-experiment \
  --index-csv Data/processed/dataset_index.csv \
  --data-root Data \
  --cache-dir Data/processed/feature_cache \
  --target coarse_affect \
  --model ridge \
  --cv within_subject_loso_session \
  --subject sub-001 \
  --seed 42
```

## Locked comparison commands

Dry-run:

```bash
thesisml-run-comparison \
  --comparison configs/comparisons/model_family_comparison_v1.json \
  --all-variants \
  --dry-run
```

Execute all registered variants:

```bash
thesisml-run-comparison \
  --comparison configs/comparisons/model_family_comparison_v1.json \
  --all-variants
```

## Confirmatory protocol commands

Dry-run:

```bash
thesisml-run-protocol \
  --protocol configs/protocols/thesis_canonical_v1.json \
  --all-suites \
  --dry-run
```

Execute official suites:

```bash
thesisml-run-protocol \
  --protocol configs/protocols/thesis_canonical_v1.json \
  --all-suites
```

## Canonical protocol and comparison sources

Confirmatory source of truth:
- `configs/protocols/thesis_canonical_v1.json`

Locked comparison source of truth:
- `configs/comparisons/model_family_comparison_v1.json`

For confirmatory and comparison runs, parameters such as target, split mode, model selection,
metric policy, controls, and interpretability policy are loaded from JSON contracts, not ad hoc CLI flags.
Each comparison/protocol contract must choose exactly one methodology policy:
- `fixed_baselines_only`
- `grouped_nested_tuning`
Thesis-facing active contracts do not include ROI-switch, dimensionality-reduction, or scaling-policy
variation dimensions; those remain out of scope unless explicitly introduced as new locked specs/protocols.

Metric consistency rules for official runs:
- each contract declares exactly one `primary_metric` (`balanced_accuracy`, `macro_f1`, or `accuracy`)
- decision selection, nested tuning objective, permutation testing, and headline summaries must use that same primary metric
- secondary metrics remain descriptive and cannot silently drive decisions
- official permutation metric must equal primary metric
- run and mode-level artifacts expose resolved metric policy in `metric_policy_effective`

## Artifacts by mode

Run-level artifacts (all modes):
- `config.json`
- `metrics.json`
- `fold_metrics.csv`
- `fold_splits.csv`
- `predictions.csv`
- `subgroup_metrics.json`
- `subgroup_metrics.csv`
- `tuning_summary.json`
- `best_params_per_fold.csv`
- `spatial_compatibility_report.json`
- `interpretability_summary.json`

Locked comparison execution artifacts:
- `comparison.json`
- `compiled_comparison_manifest.json`
- `comparison_summary.json`
- `comparison_decision.json`
- `execution_status.json`
- `report_index.csv`

Confirmatory protocol execution artifacts:
- `protocol.json`
- `compiled_protocol_manifest.json`
- `claim_to_run_map.json`
- `suite_summary.json`
- `execution_status.json`
- `report_index.csv`

## Guardrail policy

- exploratory path cannot emit confirmatory mode labels
- comparison path cannot emit confirmatory mode labels
- comparison path rejects variants outside registered specs
- confirmatory path rejects ad hoc science-affecting overrides
- protocol and comparison runs stamp mode + identity metadata into run artifacts
