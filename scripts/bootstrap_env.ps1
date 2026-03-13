$ErrorActionPreference = "Stop"

python -m pip install --upgrade pip
python -m pip install uv

python -m uv sync --frozen --extra dev
python -m uv run python -m pytest -q
python -m uv run thesisml-run-decision-support --help

$outputPath = "outputs/workbooks/bootstrap_thesis_experiment_program.xlsx"
New-Item -ItemType Directory -Path (Split-Path -Parent $outputPath) -Force | Out-Null
python -m uv run thesisml-workbook --output $outputPath

Write-Host "Bootstrap complete. Workbook written to $outputPath"
