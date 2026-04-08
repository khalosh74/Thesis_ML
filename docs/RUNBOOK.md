# Runbook

## 1) Environment

```bash
uv sync --frozen --extra dev
```

## 2) Validate Dataset

```bash
uv run thesisml-validate-dataset --dataset-manifest demo_data/synthetic_v1/dataset_manifest.json
```

## 3) Validate Release Bundle

```bash
uv run thesisml-validate-release --release releases/thesis_final_v1/release.json --dataset-manifest demo_data/synthetic_v1/dataset_manifest.json
```

## 4) Run Candidate

```bash
uv run thesisml-run-release --release releases/thesis_final_v1/release.json --dataset-manifest demo_data/synthetic_v1/dataset_manifest.json --run-class candidate
```

## 5) Verify Output

Required candidate artifacts include:

- `run_manifest.json`
- `release_manifest.json`
- `release_summary.json`
- `dataset_snapshot.json`
- `artifacts/scope/selected_samples.csv`
- `artifacts/scope/scope_manifest.json`
- `verification/scope_alignment_verification.json`
- `verification/evidence_verification.json`

## 6) Promote

```bash
uv run thesisml-promote-run --candidate-run runs/candidate/thesis_final_v1/<run_id>
```

Expected behavior:

- first promotion succeeds
- second promotion for the same release id fails
