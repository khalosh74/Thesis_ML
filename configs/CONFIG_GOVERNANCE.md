# Config Governance (Phase 1)

## Purpose
This document defines the governance layer for versioned JSON configs under `configs/`.
It introduces a central lifecycle registry (`configs/config_registry.json`) plus an explicit
authority manifest (`configs/authority_manifest.json`).

Authority layers are:
- scientific authority: what the thesis says should exist
- thesis runtime authority: what thesis runtime commands execute
- generation authority: workbook/template sources that generate runtime registries
- derived/package/archive artifacts: mirrors and backups for distribution or reproducibility only

Phase 1 is governance-only and does not change runtime behavior.

## Lifecycle Definitions
- `active_default`: current recommended default for new official work.
- `active_variant`: actively supported non-default variant.
- `frozen_confirmatory`: frozen confirmatory replay/hard-gate contract.
- `compatibility`: retained for backward compatibility and reproducibility replay.
- `archived_deprecated`: retained for traceability/replay, not for new work.

## Archive vs Retired (Phase 7–8)
- Archived configs stay in the live registry and remain valid replay/runtime inputs.
- Retired configs are removed from the live registry and are no longer valid runtime inputs.
- Active defaults do not change in Phase 7-8.
- `target.affect_mapping_v1` is retired and replaced by `target.affect_mapping_v2`.

## Rules
- Versioned configs used for official runs or thesis claims are immutable.
- New defaults are introduced by adding a new config and updating the registry.
- Frozen confirmatory configs are retained for replay/hard-gate compatibility.
- Deprecated configs are not deleted until references and replay needs are retired.
- Thesis runtime authority for decision-support execution is `configs/decision_support_registry_revised_execution.json`.
- `configs/decision_support_registry.json` is package/demo default registry, not thesis runtime authority.
- Workbook template authority is `templates/thesis_experiment_program.xlsx`.
- `templates/thesis_experiment_program_revised.xlsx` is a study workbook instance, not a generic template.
- Packaged assets under `src/Thesis_ML/assets/` are derived mirrors and must not be hand-maintained as peer authorities.
- Backup registries are archive-only and must not be discoverable as active runtime defaults.

## Current Defaults
| Alias | Resolved path |
|---|---|
| coarse affect target default | `configs/targets/affect_mapping_v2.json` |
| canonical thesis protocol default | `configs/protocols/thesis_canonical_nested_v2.json` |
| frozen confirmatory protocol | `configs/protocols/thesis_confirmatory_v1.json` |
| grouped-nested comparison default | `configs/comparisons/model_family_grouped_nested_comparison_v2.json` |
| thesis runtime decision-support registry | `configs/decision_support_registry_revised_execution.json` |
| package/demo decision-support registry | `configs/decision_support_registry.json` |
| legacy decision-support default alias | `configs/decision_support_registry_revised_execution.json` |

## Current Lifecycle Table (All Current JSON Files in `configs/`)
| config_id | path | lifecycle | replay_allowed | superseded_by |
|---|---|---|---|---|
| `target.affect_mapping_v2` | `configs/targets/affect_mapping_v2.json` | `active_default` | `true` | `null` |
| `protocol.thesis_canonical_v1` | `configs/protocols/thesis_canonical_v1.json` | `active_variant` | `true` | `null` |
| `protocol.thesis_canonical_nested_v1` | `configs/archive/protocols/thesis_canonical_nested_v1.json` | `compatibility` | `true` | `protocol.thesis_canonical_nested_v2` |
| `protocol.thesis_canonical_nested_v2` | `configs/protocols/thesis_canonical_nested_v2.json` | `active_default` | `true` | `null` |
| `protocol.thesis_confirmatory_v1` | `configs/protocols/thesis_confirmatory_v1.json` | `frozen_confirmatory` | `true` | `null` |
| `comparison.model_family_comparison_v1` | `configs/comparisons/model_family_comparison_v1.json` | `active_variant` | `true` | `null` |
| `comparison.model_family_grouped_nested_comparison_v1` | `configs/archive/comparisons/model_family_grouped_nested_comparison_v1.json` | `compatibility` | `true` | `comparison.model_family_grouped_nested_comparison_v2` |
| `comparison.model_family_grouped_nested_comparison_v2` | `configs/comparisons/model_family_grouped_nested_comparison_v2.json` | `active_default` | `true` | `null` |
| `registry.decision_support_thesis_runtime` | `configs/decision_support_registry_revised_execution.json` | `active_default` | `true` | `null` |
| `registry.decision_support_package_default` | `configs/decision_support_registry.json` | `active_variant` | `true` | `null` |

## Retired Configs
| config_id | former_path | replacement | notes |
|---|---|---|---|
| `target.affect_mapping_v1` | `configs/targets/affect_mapping_v1.json` | `target.affect_mapping_v2` | Deprecated six-label mapping retired after zero operational references remained. |

## Bundle Compatibility Rules (Phase 6)
Bundle compatibility rules are defined centrally in `configs/config_registry.json` under the top-level `bundles` field.

Phase 6 currently covers official protocol/comparison/target bundles only.
Single-config runners are not restricted by bundle rules in this phase.

| bundle_id | protocol | comparison | target | lifecycle | notes |
|---|---|---|---|---|---|
| `bundle.thesis_canonical_v1_fixed_linear` | `protocol.thesis_canonical_v1` | `comparison.model_family_comparison_v1` | `target.affect_mapping_v2` | `active_variant` | Supported fixed-baseline official bundle. |
| `bundle.thesis_canonical_nested_v1_grouped_nested` | `protocol.thesis_canonical_nested_v1` | `comparison.model_family_grouped_nested_comparison_v1` | `target.affect_mapping_v2` | `compatibility` | Backward-compatible grouped-nested replay bundle. |
| `bundle.thesis_canonical_nested_v2_grouped_nested` | `protocol.thesis_canonical_nested_v2` | `comparison.model_family_grouped_nested_comparison_v2` | `target.affect_mapping_v2` | `active_default` | Current canonical grouped-nested modeling bundle. |
| `bundle.thesis_confirmatory_v1_publishable` | `protocol.thesis_confirmatory_v1` | `comparison.model_family_grouped_nested_comparison_v2` | `target.affect_mapping_v2` | `frozen_confirmatory` | Current publishable/replay official bundle. |
