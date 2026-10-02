#!/usr/bin/env bash
#SBATCH --job-name=bank-analysis
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=runtime/logs/bank-analysis-%j.out

set -euo pipefail
repo_root=${ULM_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}
cd "$repo_root"
export ULM_RUNTIME_ROOT=${ULM_RUNTIME_ROOT:-"$repo_root/runtime"}
export PYTHONPATH="$repo_root/src:$ULM_RUNTIME_ROOT/analysis-deps${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=${HF_HOME:-"$ULM_RUNTIME_ROOT/models/huggingface"}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS MKL_NUM_THREADS=$OMP_NUM_THREADS
python_path=${UV_PROJECT_ENVIRONMENT:-"$ULM_RUNTIME_ROOT/venv"}/bin/python
# Pass the bank_pipeline CLI arguments, including a fresh --output-dir.
time "$python_path" -m quantitive_analysis.bank_pipeline "$@"
