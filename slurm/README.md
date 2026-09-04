# Slurm

After `bash setup.sh`, submit an experiment from the repository root:

```bash
sbatch slurm/experiment.sbatch configs/experiments/tylenol-demon.yaml
```

To resume an interrupted run, pass its directory as the second argument. The
wrapper keeps the environment, model cache, lens artifacts, logs, and output
inside the ignored `runtime/` directory unless the corresponding environment
variables are explicitly overridden.
