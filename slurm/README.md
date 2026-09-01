# Slurm jobs

Batch scripts for running experiments on the compute cluster.

Submit a steering YAML with:

```bash
export ULM_REPO_ROOT=/absolute/path/to/UnderstandLanguageModels_Project
export ULM_RUNTIME_ROOT="$ULM_REPO_ROOT/runtime"
export UV_PROJECT_ENVIRONMENT="$ULM_RUNTIME_ROOT/venv"
export HF_HOME="$ULM_RUNTIME_ROOT/models/huggingface"
export JLENS_DATA_ROOT="$ULM_RUNTIME_ROOT/models/j-lens/qwen3-32b"
export STEERING_OUTPUT_ROOT="$ULM_RUNTIME_ROOT/outputs/steering"
export STEERING_MODEL_ID=Qwen/Qwen3-32B-FP8
mkdir -p "$ULM_RUNTIME_ROOT/logs" "$HF_HOME" "$JLENS_DATA_ROOT" \
  "$STEERING_OUTPUT_ROOT"

sbatch --output="$ULM_RUNTIME_ROOT/logs/steering-%j.out" \
  slurm/steering.sbatch configs/steering/role_direction.yaml
```

The runtime directory is ignored by Git and contains the environment, model
cache, generated experiments, and logs. It can still be overridden with an
external path when required by the cluster.

The job runs `steering-preflight` before loading Qwen. Its defaults require 75
GB VRAM and 80 GB free runtime storage; override them only when intentionally
using another model or loading mode with `STEERING_MIN_VRAM_GB` and
`STEERING_MIN_FREE_DISK_GB`.

To resume, pass an existing run directory as the second argument. Slurm inherits
the configured paths and the wrapper resolves both repository-relative and
absolute config/run paths.
