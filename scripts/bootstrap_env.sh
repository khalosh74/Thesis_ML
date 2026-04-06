#!/usr/bin/env bash
set -euo pipefail

INSTALL_GPU_TORCH=0
TORCH_INDEX_URL="https://download.pytorch.org/whl/cu130"

while [[ $# -gt 0 ]]; do
	case "$1" in
		--install-gpu-torch)
			INSTALL_GPU_TORCH=1
			shift
			;;
		--torch-index-url)
			TORCH_INDEX_URL="${2:?--torch-index-url requires a value}"
			shift 2
			;;
		*)
			echo "Unknown argument: $1" >&2
			exit 1
			;;
	esac
done

python -m pip install --upgrade pip
python -m pip install uv

python -m uv sync --frozen --extra dev

if [[ "$INSTALL_GPU_TORCH" -eq 1 ]]; then
	python -m uv run python -m pip install --upgrade torch --index-url "$TORCH_INDEX_URL"
	python -m uv run python -c "import torch; print('torch_version', torch.__version__); print('cuda_compiled', torch.version.cuda); print('cuda_available', torch.cuda.is_available()); print('device_count', torch.cuda.device_count()); assert torch.cuda.is_available(), 'Installed torch but CUDA is unavailable in the project environment.'"
fi

python -m uv run python -m pytest -q
python -m uv run thesisml-run-decision-support --help

mkdir -p outputs/workbooks
python -m uv run thesisml-workbook --output outputs/workbooks/bootstrap_thesis_experiment_program.xlsx

echo "Bootstrap complete. Workbook written to outputs/workbooks/bootstrap_thesis_experiment_program.xlsx"
