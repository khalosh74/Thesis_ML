# Segment Execution

Segment execution allows running a contiguous section range instead of the full pipeline.

Implementation: `src/Thesis_ML/experiments/segment_execution.py`

## Supported section order

Execution order is fixed:

1. `dataset_selection`
2. `feature_cache_build`
3. `feature_matrix_load`
4. `spatial_validation`
5. `model_fit`
6. `interpretability`
7. `evaluation`

`start_section` must be before or equal to `end_section`.

## CLI usage

Full pipeline (default):

```bash
thesisml-run-experiment ... --run-id run_full
```

Segment example:

```bash
thesisml-run-experiment \
  ... \
  --start-section feature_matrix_load \
  --end-section evaluation \
  --base-artifact-id <feature_cache_or_feature_matrix_artifact_id> \
  --reuse-policy require_explicit_base \
  --run-id run_segment
```

## Base artifact rules

- Starting before `feature_matrix_load`: `base_artifact_id` is not allowed.
- Starting at `feature_matrix_load`: expects `feature_cache` artifact.
- Starting after `feature_matrix_load`: expects `feature_matrix_bundle` artifact.
- `reuse_policy=disallow` is invalid for segmented starts that require base artifacts.
- `reuse_policy=require_explicit_base` requires `base_artifact_id`.

## Resume/reuse integration

`run_experiment` forwards:

- `resume=True` -> enables controlled same-run completed artifact reuse.
- `reuse_completed_artifacts=True` -> explicit reuse for compatible same-run section outputs.
- `force=True` -> clears the run directory and disables resume behavior for that invocation.
- `force=True` and `resume=True` are mutually exclusive.
- same-run reuse only applies within the current `run_id` output directory.
- cross-run reuse requires explicit `base_artifact_id` with compatible segmented start.

Result payload includes:

- `planned_sections`
- `executed_sections`
- `reused_sections`
- `artifact_ids`

## Typical validation errors

- invalid section name
- start section after end section
- missing/incompatible base artifact
- missing prerequisite outputs for requested section path
