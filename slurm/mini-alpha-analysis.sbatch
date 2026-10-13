#!/usr/bin/env bash
#SBATCH --job-name=alpha-analysis
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=runtime/logs/alpha-analysis-%j.out

set -euo pipefail
repo_root=${ULM_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}
cd "$repo_root"
if [[ $# -lt 1 ]]; then
  echo 'Usage: sbatch slurm/mini-alpha-analysis.sbatch /absolute/path/to/alpha-results [readout-output] [report-output]' >&2
  exit 1
fi
alpha_run=$1
alpha_readouts=${2:-"$repo_root/outputs/analysis/mini-alpha-readout"}
alpha_reports=${3:-"$repo_root/results/mini-alpha-readout"}
alpha_python=${UV_PROJECT_ENVIRONMENT:-"$repo_root/.venv-analysis"}/bin/python
alpha_assets=${ULM_READOUT_ASSETS:-"${ULM_RUNTIME_ROOT:-$repo_root/runtime}/models/j-lens/qwen3-32b"}
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS MKL_NUM_THREADS=$OMP_NUM_THREADS
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
"$alpha_python" -u -m quantitive_analysis.local_readout --run "$alpha_run" --data "$alpha_assets" \
  --output "$alpha_readouts" --threads "$OMP_NUM_THREADS" --layers 29 32 35 62
"$alpha_python" -u scripts/build_alpha_analysis.py --run "$alpha_run" --readouts "$alpha_readouts" --output "$alpha_reports"
"$alpha_python" -u scripts/build_mini_batch_token_html.py --run "$alpha_run" --readouts "$alpha_readouts" \
  --comparisons "$alpha_reports/comparisons.json" --output "$alpha_reports/token-comparison.html"
"$alpha_python" -c 'from pathlib import Path; import sys; p=Path(sys.argv[1])/"index.html"; p.write_text(p.read_text().replace("The token-level page is built separately by the cluster job after these tables.", "<a href=\"token-comparison.html\">Open token trajectories and answer pairs →</a>"))' "$alpha_reports"
