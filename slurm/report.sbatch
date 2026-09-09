#!/usr/bin/env bash
#SBATCH --job-name=steering-report
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=runtime/logs/report-%j.out

set -euo pipefail
repo_root=${ULM_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}
export ULM_RUNTIME_ROOT=${ULM_RUNTIME_ROOT:-"$repo_root/runtime"}
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python_path=${UV_PROJECT_ENVIRONMENT:-"$ULM_RUNTIME_ROOT/venv"}/bin/python
# Arguments are the same as steering-report. No model or GPU is loaded.
/usr/bin/time -v "$python_path" -m steering.cli.report "$@"
