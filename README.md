# Thesis_ML

Thesis_ML is a release-governed fMRI decoding system.

The only thesis-final runtime path is:

1. `thesisml-validate-dataset`
2. `thesisml-validate-release`
3. `thesisml-run-release`
4. `thesisml-promote-run`

## Official Flow

```bash
uv run thesisml-validate-dataset --dataset-manifest demo_data/synthetic_v1/dataset_manifest.json
uv run thesisml-validate-release --release releases/thesis_final_v1/release.json --dataset-manifest demo_data/synthetic_v1/dataset_manifest.json
uv run thesisml-run-release --release releases/thesis_final_v1/release.json --dataset-manifest demo_data/synthetic_v1/dataset_manifest.json --run-class candidate
uv run thesisml-promote-run --candidate-run runs/candidate/thesis_final_v1/<run_id>
```

## Authority

- Top-level bundle: `releases/thesis_final_v1/release.json`
- Scientific scope: `releases/thesis_final_v1/science.json`
- Dataset contract: `data/contracts/fmri_beta_dataset_v1.json`

Authority precedence:

`release.json > science.json > dataset manifest > environment.json > execution.json > evidence.json`

## Scope Enforcement

Official release scope is compiled once into:

- `artifacts/scope/selected_samples.csv`
- `artifacts/scope/scope_manifest.json`

Execution consumes that compiled subset. Runtime scope filters are not authoritative.

Every candidate/official run must include:

- `verification/scope_alignment_verification.json`
- `verification/evidence_verification.json`

## Run Layout

```text
runs/
  scratch/<release_id>/<run_id>/
  exploratory/<release_id>/<run_id>/
  candidate/<release_id>/<run_id>/
  official/<release_id>/<official_run_id>/
```

Candidate runs become official only through promotion.
Exactly one official run is allowed per release id.
