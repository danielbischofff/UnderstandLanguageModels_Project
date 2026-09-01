# Slurm jobs

Batch scripts for running experiments on the compute cluster.

Submit a steering YAML with:

```bash
export ULM_REPO_ROOT=/absolute/path/to/UnderstandLanguageModels_Project
export ULM_RUNTIME_ROOT="$ULM_REPO_ROOT/runtime"
export UV_PROJECT_ENVIRONMENT="$ULM_RUNTIME_ROOT/venv"
export HF_HOME="$ULM_RUNTIME_ROOT/models/huggingface"
export STEERING_OUTPUT_ROOT="$ULM_RUNTIME_ROOT/outputs/steering"
mkdir -p "$ULM_RUNTIME_ROOT/logs" "$HF_HOME" "$STEERING_OUTPUT_ROOT"

sbatch --output="$ULM_RUNTIME_ROOT/logs/steering-%j.out" \
  slurm/steering.sbatch configs/steering/role_direction.yaml
```

The runtime directory is ignored by Git and contains the environment, model
cache, generated experiments, and logs. It can still be overridden with an
external path when required by the cluster.

To resume, pass an existing run directory as the second argument. Slurm inherits
the configured paths and the wrapper resolves both repository-relative and
absolute config/run paths.
