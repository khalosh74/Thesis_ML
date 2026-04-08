# Release-System Migration Note

## Baseline Authority Conflicts
- `configs/confirmatory/confirmatory_scope_v1.json`
- `configs/protocols/thesis_confirmatory_v1.json`
- `configs/protocols/thesis_canonical_nested_v2.json`
- `configs/decision_support_registry_revised_execution.json`
- `configs/authority_manifest.json`

Current state before migration: multiple parallel top-level authorities define overlapping science/runtime behavior for thesis-critical execution.

## New Precedence (Official Path)
1. `release.json`
2. `science.json`
3. dataset manifest (`dataset_manifest.json` instance)
4. `environment.json`
5. `execution.json`
6. `evidence.json`

## Official/Non-Official Split After Migration
- Official thesis path: release bundle under `releases/<release_id>/` with release runner + promotion.
- Non-official (legacy/exploratory compatibility): protocol/comparison/decision-support/workbook authorities and their CLIs remain runnable but are no longer official top-level authority.

