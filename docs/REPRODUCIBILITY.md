# Reproducibility Workflow

This is the canonical reproducibility workflow for official paths (locked comparison + confirmatory).

## Canonical environment (source of truth)

Use:

- Python `3.13` (see `.python-version`)
- `uv.lock`
- `uv sync --frozen --extra dev`

```bash
python -m pip install --upgrade pip
python -m pip install uv
python -m uv sync --frozen --extra dev
```

## Checked-in demo dataset

The reproducibility dataset is checked in at:

- `demo_data/synthetic_v1/`

It contains:

- tiny deterministic synthetic NIfTI payloads and masks
- `dataset_index.csv` with required official columns
- `demo_dataset_manifest.json` with file hashes

Regenerate deterministically:

```bash
python scripts/generate_demo_dataset.py --output demo_data/synthetic_v1 --force
```

## One-command official replay

Run official replay with artifact verification and reproducibility manifest emission:

```bash
python scripts/replay_official_paths.py \
  --mode both \
  --use-demo-dataset \
  --reports-root outputs/reproducibility/official_replay \
  --summary-out outputs/reproducibility/official_replay/replay_summary.json \
  --verification-summary-out outputs/reproducibility/official_replay/replay_verification_summary.json \
  --manifest-out outputs/reproducibility/official_replay/reproducibility_manifest.json
```

Optional deterministic rerun check (double execution per selected mode):

```bash
python scripts/replay_official_paths.py \
  --mode both \
  --use-demo-dataset \
  --verify-determinism
```

## Deterministic replay comparator

The deterministic comparator remains available directly:
for frozen confirmatory replay / hard-gate validation (`thesis_confirmatory_v1.json`).

```bash
python scripts/replay_official_paths.py \
  --mode confirmatory \
  --protocol configs/protocols/thesis_confirmatory_v1.json \
  --index-csv demo_data/synthetic_v1/dataset_index.csv \
  --data-root demo_data/synthetic_v1/data_root \
  --cache-dir demo_data/synthetic_v1/cache \
  --suite confirmatory_primary_within_subject \
  --verify-determinism \
  --skip-confirmatory-ready
```

## Publishable bundle

Build canonical directory bundle + manifest:

```bash
python scripts/build_publishable_bundle.py \
  --output-dir outputs/reproducibility/publishable_bundle \
  --comparison-output outputs/reproducibility/official_replay/comparison/comparison_runs/model-family-within-subject__1.0.0 \
  --confirmatory-output outputs/reproducibility/official_replay/confirmatory/protocol_runs/thesis_confirmatory_v1__v1.0 \
  --replay-summary outputs/reproducibility/official_replay/replay_summary.json \
  --replay-verification-summary outputs/reproducibility/official_replay/replay_verification_summary.json \
  --repro-manifest outputs/reproducibility/official_replay/reproducibility_manifest.json
```

Verify bundle structure + hashes + official artifact contracts:

```bash
python scripts/verify_publishable_bundle.py \
  --bundle-dir outputs/reproducibility/publishable_bundle
```

## RC gate integration

RC gate can run official replay directly and verify bundles:

```bash
python scripts/rc1_release_gate.py \
  --run-official-replay \
  --replay-use-demo-dataset \
  --replay-verify-determinism \
  --verify-bundle-dir outputs/reproducibility/publishable_bundle
```

## Primary reproducibility outputs

- `replay_summary.json`
- `replay_verification_summary.json`
- `reproducibility_manifest.json`
- `bundle_manifest.json`
