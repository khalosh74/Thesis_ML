# Thesis_ML

Local-only, cross-platform ML investigation baseline with reproducible tooling and CI.

## Principles

- Keep all private data local and untracked.
- Keep core logic in `src/Thesis_ML`, not notebooks.
- Keep scripts cross-platform by using Python entrypoints.
- Keep quality gates (`ruff`, `pytest`) green locally and in CI.

## Repository Layout

```
src/Thesis_ML/
notebooks/
scripts/
tests/
reports/
models/
data/
  raw/
  interim/
  processed/
docs/
.github/
```

## Local Data Policy

- Place private data only under local `data/` (or existing `Data/` on case-insensitive filesystems).
- Do not commit any real data, models, or report artifacts.
- Do not upload dataset files to cloud services from this repository.

## Setup and Commands

### Windows (PowerShell)

1. Activate `.venv`:

```powershell
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

3. Lint/format check:

```powershell
python -m ruff check .
python -m ruff format --check .
```

4. Tests:

```powershell
python -m pytest -q
```

5. Run baseline experiment:

```powershell
python scripts/run_baseline.py
```

### Debian/Ubuntu (bash)

1. Activate `.venv`:

```bash
source .venv/bin/activate
```

2. Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

3. Lint/format check:

```bash
python -m ruff check .
python -m ruff format --check .
```

4. Tests:

```bash
python -m pytest -q
```

5. Run baseline experiment:

```bash
python scripts/run_baseline.py
```

## Baseline Output

- Metrics JSON: `reports/metrics.json`
- Model artifact: `models/baseline_model.npz`
