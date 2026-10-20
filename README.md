# Thesis_ML — reproducible fMRI decoding experiments

**Master's thesis research | Python · scikit-learn · reproducibility · testing**

This repository contains an experimental pipeline for studying **emotion decoding from fMRI data**, including how machine-learning results hold up across scanning sessions. The research question is about model stability and generalization, not just obtaining a high score on one dataset.

For a technical reviewer, the interesting parts are the explicit dataset contracts, run validation, automated checks, and traceable artifacts. The project separates exploratory work from a controlled thesis-final execution path.

## Start here

- **Research problem:** Within-person stability and cross-session generalization in fMRI emotion decoding.
- **Code:** [src/Thesis_ML](src/Thesis_ML) for the Python package; [tests](tests) for validation and regression coverage.
- **Synthetic example:** [demo_data/synthetic_v1](demo_data/synthetic_v1) allows inspection of the release workflow without access to the research MRI data.
- **Implementation notes:** [Architecture](docs/ARCHITECTURE.md) · [Runbook](docs/RUNBOOK.md) · [Release specification](docs/RELEASE.md).
- **CI:** [GitHub Actions](../../actions) runs Python tests, package-build checks, and a Docker build.

## Running the demonstration

Install [uv](https://docs.astral.sh/uv/) and use the repository root:

```bash
uv sync --extra dev
uv run pytest
```

The example below uses the synthetic dataset manifest. Its outputs demonstrate the workflow; **they are not thesis performance results or claims of clinical validity**.

## Controlled thesis-final workflow
Thesis_ML is a release-governed fMRI decoding system.

The thesis-final runtime path consists of four steps:

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
