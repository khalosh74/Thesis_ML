# Release Gate

This repository uses a tag-triggered release validation workflow:

- Workflow: `.github/workflows/release_gate.yml`
- Trigger: push tag matching `v*` (for example `v0.2.0`)
- Manual trigger: GitHub Actions `workflow_dispatch`

## What it validates

1. Builds `sdist` and `wheel` with `python -m build`.
2. Validates package metadata with `python -m twine check dist/*`.
3. Installs the built wheel (not editable mode).
4. Runs gold acceptance smoke:
   - `python scripts/acceptance_smoke.py`
5. Verifies installed-wheel default registry behavior with canonical CLI:
   - `thesisml-run-decision-support --all --dry-run ...` (no explicit `--registry`)
6. Runs a lightweight runtime test:
   - `python -m pytest tests/test_import_decoupling.py -q`
7. Uploads `dist/*` as workflow artifacts.

## Local pre-tag check (recommended)

```bash
python -m pip install --upgrade pip
python -m pip install build twine pytest
python -m build --sdist --wheel
python -m twine check dist/*
python -m pip install --force-reinstall dist/*.whl
python scripts/acceptance_smoke.py
python - <<'PY'
from pathlib import Path
import pandas as pd

temp_root = Path("/tmp/thesisml_release_gate")
temp_root.mkdir(parents=True, exist_ok=True)
pd.DataFrame(
    [{"subject": "sub-001", "task": "passive", "modality": "audio"}]
).to_csv(temp_root / "dataset_index.csv", index=False)
PY
thesisml-run-decision-support \
  --index-csv /tmp/thesisml_release_gate/dataset_index.csv \
  --data-root /tmp/thesisml_release_gate/Data \
  --cache-dir /tmp/thesisml_release_gate/cache \
  --output-root /tmp/thesisml_release_gate/outputs \
  --all \
  --dry-run
python -m pytest tests/test_import_decoupling.py -q
```

Additional local hygiene/performance checks:

```bash
python scripts/release_hygiene_check.py
python scripts/performance_smoke.py --output outputs/performance/performance_smoke_summary.json
```

## RC-1 hardening gate (official paths)

RC-1 adds explicit official-run release checks for:
- strict official artifact completeness/invariants
- deterministic rerun verification on a small official path
- optional one-command wrapper for hygiene/lint/tests/performance
- structured failure payloads in mode-level execution artifacts (`error_code`, `error_type`, `failure_stage`, `error_details`)
- evidence-layer artifact invariants for official modes:
  - confirmatory: `repeated_run_metrics.csv`, `repeated_run_summary.json`,
    `confidence_intervals.json`, `metric_intervals.csv`
  - comparison: all confirmatory evidence artifacts plus
    `paired_model_comparisons.json` and `paired_model_comparisons.csv`
- data-layer artifact invariants for official modes:
  - `dataset_card.json`, `dataset_card.md`
  - `dataset_summary.json`, `dataset_summary.csv`
  - `data_quality_report.json`, `class_balance_report.csv`, `missingness_report.csv`
  - `leakage_audit.json`
  - `external_dataset_card.json`, `external_dataset_summary.json`,
    `external_validation_compatibility.json` (required when external compatibility checks are configured)

### 1) Official artifact verification

Verify one locked-comparison or confirmatory output directory:

```bash
python scripts/verify_official_artifacts.py --output-dir <official_output_dir>
```

Optional mode hint:

```bash
python scripts/verify_official_artifacts.py --output-dir <official_output_dir> --mode confirmatory
python scripts/verify_official_artifacts.py --output-dir <official_output_dir> --mode comparison
```

### 2) Deterministic rerun verification

Run a small official path twice and compare deterministic invariants:

```bash
python scripts/verify_official_reproducibility.py \
  --mode protocol \
  --config configs/protocols/thesis_confirmatory_v1.json \
  --index-csv <dataset_index.csv> \
  --data-root <data_root> \
  --cache-dir <cache_dir> \
  --suite confirmatory_primary_within_subject
```

Comparison example:

```bash
python scripts/verify_official_reproducibility.py \
  --mode comparison \
  --config configs/comparisons/model_family_comparison_v1.json \
  --index-csv <dataset_index.csv> \
  --data-root <data_root> \
  --cache-dir <cache_dir> \
  --variant ridge
```

### 3) RC wrapper gate

Run release hygiene plus optional lint/tests/performance and official checks:

```bash
python scripts/rc1_release_gate.py \
  --run-ruff \
  --run-pytest \
  --run-performance-smoke \
  --verify-official-dir <official_output_dir>
```

Use `--repro-command` to include a reproducibility check in the same gate run:

```bash
python scripts/rc1_release_gate.py \
  --run-ruff \
  --run-pytest \
  --repro-command "python scripts/verify_official_reproducibility.py --mode protocol --config configs/protocols/thesis_confirmatory_v1.json --index-csv <dataset_index.csv> --data-root <data_root> --cache-dir <cache_dir> --suite confirmatory_primary_within_subject"
```

## Official RC checklist

Before freezing an experiment campaign:

1. Run baseline release checks (`build`, `twine`, `acceptance_smoke`).
2. Run `python scripts/release_hygiene_check.py`.
3. Validate at least one official comparison/protocol output with `verify_official_artifacts.py`.
4. Run deterministic rerun verification with `verify_official_reproducibility.py` on a small official path.
5. Capture performance smoke output:
   `python scripts/performance_smoke.py --output outputs/performance/performance_smoke_summary.json`.
6. Archive RC gate summary:
   `python scripts/rc1_release_gate.py --summary-out outputs/release/rc1_gate_summary.json`.
