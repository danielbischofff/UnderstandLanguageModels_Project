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

export HF_HOME=${HF_HOME:-"$PWD/models/huggingface"}
export TOKENIZERS_PARALLELISM=false

if [[ -n "$resume_dir" ]]; then
  uv run python -m steering.cli.sweep --resume "$resume_dir"
else
  uv run python -m steering.cli.sweep --config "$config_path"
fi
