param(
    [switch]$InstallGpuTorch,
    [string]$TorchIndexUrl = "https://download.pytorch.org/whl/cu130"
)

$ErrorActionPreference = "Stop"

python -m pip install --upgrade pip
python -m pip install uv

python -m uv sync --frozen --extra dev

if ($InstallGpuTorch) {
	python -m uv run python -m pip install --upgrade torch --index-url $TorchIndexUrl
	python -m uv run python -c "import torch; print('torch_version', torch.__version__); print('cuda_compiled', torch.version.cuda); print('cuda_available', torch.cuda.is_available()); print('device_count', torch.cuda.device_count()); assert torch.cuda.is_available(), 'Installed torch but CUDA is unavailable in the project environment.'"
}

python -m uv run python -m pytest -q
python -m uv run thesisml-run-decision-support --help

$outputPath = "outputs/workbooks/bootstrap_thesis_experiment_program.xlsx"
New-Item -ItemType Directory -Path (Split-Path -Parent $outputPath) -Force | Out-Null
python -m uv run thesisml-workbook --output $outputPath

Write-Host "Bootstrap complete. Workbook written to $outputPath"
