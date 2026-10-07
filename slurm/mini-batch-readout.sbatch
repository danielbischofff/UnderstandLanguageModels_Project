#!/usr/bin/env bash
#SBATCH --job-name=mini-readout
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=runtime/logs/mini-readout-%j.out

set -euo pipefail
repo_root=${ULM_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}
cd "$repo_root"
runtime_root=${ULM_RUNTIME_ROOT:-"$repo_root/runtime"}
python_path=${UV_PROJECT_ENVIRONMENT:-"$runtime_root/venv"}/bin/python
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS MKL_NUM_THREADS=$OMP_NUM_THREADS
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
if [[ ! -x "$python_path" ]]; then
  echo "Missing Python environment: $python_path" >&2
  echo "Set UV_PROJECT_ENVIRONMENT to your analysis environment directory." >&2
  exit 1
fi
# Submit from the repository root after creating runtime/logs.
# CLI options below may be overridden by arguments passed to sbatch.
exec "$python_path" -u -m quantitive_analysis.local_readout \
  --run "$repo_root/outputs/mini_batch_exp_results" \
  --data "$runtime_root/models/j-lens/qwen3-32b" \
  --output "$repo_root/outputs/analysis/mini-batch-cluster-readout" \
  --threads "$OMP_NUM_THREADS" "$@"
