# Release Gate

This repository uses a tag-triggered release validation workflow:

- Workflow: `.github/workflows/release_gate.yml`
- Trigger: push tag matching `v*` (for example `v0.2.0`)
- Manual trigger: GitHub Actions `workflow_dispatch`

## What it validates

1. Builds `sdist` and `wheel` with `python -m build`.
2. Validates package metadata with `python -m twine check dist/*`.
3. Installs the built wheel (not editable mode).
4. Runs CLI smoke checks:
   - `thesisml-run-decision-support --help`
   - `thesisml-workbook --output /tmp/thesis_experiment_program_release_gate.xlsx`
5. Runs a lightweight runtime test:
   - `python -m pytest tests/test_import_decoupling.py -q`
6. Uploads `dist/*` as workflow artifacts.

## Local pre-tag check (recommended)

```bash
python -m pip install --upgrade pip
python -m pip install build twine pytest
python -m build --sdist --wheel
python -m twine check dist/*
python -m pip install --force-reinstall dist/*.whl
thesisml-run-decision-support --help
thesisml-workbook --output /tmp/thesis_experiment_program_release_gate.xlsx
python -m pytest tests/test_import_decoupling.py -q
```
