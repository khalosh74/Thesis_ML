# Release System

## Bundle

`releases/thesis_final_v1/release.json` references:

- `science.json`
- `execution.json`
- `environment.json`
- `evidence.json`
- `claims.json`

## Validation

```bash
uv run thesisml-validate-dataset --dataset-manifest <dataset_manifest.json>
uv run thesisml-validate-release --release releases/thesis_final_v1/release.json --dataset-manifest <dataset_manifest.json>
```

## Execution

`thesisml-run-release` accepts only:

- `--run-class scratch`
- `--run-class exploratory`
- `--run-class candidate`

`official` creation is blocked in the runner and allowed only through promotion.

## Scope Contract

`science.json` is the only scientific scope authority.

The runner compiles scope before model execution into:

- `artifacts/scope/selected_samples.csv`
- `artifacts/scope/scope_manifest.json`

Execution must align to the compiled sample ids. Mismatch fails the run.

## Promotion

```bash
uv run thesisml-promote-run --candidate-run <candidate_run_path>
```

Promotion requires passed evidence and scope-alignment verification.
A second official promotion for the same release id fails.
