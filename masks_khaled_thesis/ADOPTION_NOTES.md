# masks_khaled_thesis Adoption Notes

This folder is now the canonical path for thesis ROI mask consumption.

## Official masks used by feature specs

- `masks_khaled_thesis/official_beta2mm/julich_v303_v1_bilateral_beta2mm.nii.gz`
- `masks_khaled_thesis/official_beta2mm/julich_v303_pac_te1_bilateral_beta2mm.nii.gz`

## Implementation detail

Due to disk-space limits on this workstation, these files are created as hard links
pointing to the original validated assets under:

- `configs/feature_spaces/assets/julich_v3_0_3/masks/`

Hard links preserve identical bytes while avoiding duplicate storage.

## Why the raw updated_atlas pair is not used directly

The provided `.hdr`/`.img` files do not form a directly loadable Analyze pair by filename.
A direct pipeline smoke test fails for those exact paths.

Validated report:
- `outputs/validation_runs/masks_khaled_thesis_validation/masks_khaled_thesis_validation_report.json`
