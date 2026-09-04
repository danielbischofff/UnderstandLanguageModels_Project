#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
config_path=${1:-"$repo_root/configs/experiments/tylenol-demon.yaml"}
runtime_root=${ULM_RUNTIME_ROOT:-"$repo_root/runtime"}

if [[ "$config_path" != /* ]]; then
  config_path="$repo_root/$config_path"
fi

export ULM_RUNTIME_ROOT="$runtime_root"
export UV_PROJECT_ENVIRONMENT=${UV_PROJECT_ENVIRONMENT:-"$runtime_root/venv"}
export HF_HOME=${HF_HOME:-"$runtime_root/models/huggingface"}
export JLENS_DATA_ROOT=${JLENS_DATA_ROOT:-"$runtime_root/models/j-lens/qwen3-32b"}
export STEERING_OUTPUT_ROOT=${STEERING_OUTPUT_ROOT:-"$runtime_root/outputs/experiments"}
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE UV_OFFLINE
mkdir -p "$HF_HOME" "$JLENS_DATA_ROOT" "$STEERING_OUTPUT_ROOT" "$runtime_root/logs"

if ! command -v uv >/dev/null 2>&1; then
  uv_bin="$runtime_root/bin"
  mkdir -p "$uv_bin"
  curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$uv_bin" sh
  export PATH="$uv_bin:$PATH"
fi

cd "$repo_root"
uv sync --frozen --extra dev
uv run steering-prefetch --config "$config_path"
uv run python direct_character_vectors_j_lens_analysis/src/fetch_data.py
uv run steering-prefetch --config "$config_path" --check

echo "Setup complete. Submit $config_path with slurm/experiment.sbatch."
