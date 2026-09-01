#!/usr/bin/env bash
#SBATCH --job-name=persona-steering
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=logs/steering-%j.out

set -euo pipefail

config_path=${1:?Usage: sbatch slurm/steering.sbatch CONFIG [RUN_DIR]}
resume_dir=${2:-}
submit_dir=${SLURM_SUBMIT_DIR:-$PWD}
repo_root=${ULM_REPO_ROOT:-$submit_dir}

if [[ "$config_path" != /* ]]; then
  config_path="$submit_dir/$config_path"
fi
if [[ -n "$resume_dir" && "$resume_dir" != /* ]]; then
  resume_dir="$submit_dir/$resume_dir"
fi

cd "$repo_root"

export ULM_RUNTIME_ROOT=${ULM_RUNTIME_ROOT:-"$repo_root/runtime"}
export UV_PROJECT_ENVIRONMENT=${UV_PROJECT_ENVIRONMENT:-"$ULM_RUNTIME_ROOT/venv"}
export HF_HOME=${HF_HOME:-"$ULM_RUNTIME_ROOT/models/huggingface"}
export JLENS_DATA_ROOT=${JLENS_DATA_ROOT:-"$ULM_RUNTIME_ROOT/models/j-lens/qwen3-32b"}
export STEERING_OUTPUT_ROOT=${STEERING_OUTPUT_ROOT:-"$ULM_RUNTIME_ROOT/outputs/steering"}
export TOKENIZERS_PARALLELISM=false
mkdir -p "$HF_HOME" "$JLENS_DATA_ROOT" "$STEERING_OUTPUT_ROOT" "$ULM_RUNTIME_ROOT/logs"

uv run steering-preflight \
  --model "${STEERING_MODEL_ID:-Qwen/Qwen3-32B-FP8}" \
  --minimum-vram-gb "${STEERING_MIN_VRAM_GB:-75}" \
  --minimum-free-disk-gb "${STEERING_MIN_FREE_DISK_GB:-80}"

if [[ -n "$resume_dir" ]]; then
  uv run python -m steering.cli.sweep --resume "$resume_dir"
else
  uv run python -m steering.cli.sweep \
    --config "$config_path" \
    --output-root "$STEERING_OUTPUT_ROOT"
fi
