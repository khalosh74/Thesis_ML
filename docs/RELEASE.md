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
