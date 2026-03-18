# Synthetic Demo Dataset (v1)

This directory contains a tiny deterministic synthetic dataset used for framework reproducibility checks.

- Intended use: validate official comparison/confirmatory workflow execution, artifact generation, and deterministic replay.
- Not intended use: scientific claims, model quality conclusions, or any clinical/cognitive interpretation.

Generation:
- Source script: `scripts/generate_demo_dataset.py`
- Structure: `data_root/sub-*/ses-*/BAS2/` with synthetic `beta_*.nii`, `mask.nii`, and `regressor_labels.csv`
- Index: `dataset_index.csv` built via `Thesis_ML.data.index_dataset.build_dataset_index`
- Manifest: `demo_dataset_manifest.json` with deterministic file hashes
