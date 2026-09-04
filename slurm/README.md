# Slurm

Run setup on an internet-connected login node, then submit from the repository
root:

```bash
bash setup.sh configs/experiments/tylenol-demon.yaml
sbatch slurm/experiment.sbatch configs/experiments/tylenol-demon.yaml
```

To resume an interrupted run, pass its directory as the second argument. The
wrapper keeps the environment, model cache, lens artifacts, logs, and output
inside the ignored `runtime/` directory unless the corresponding environment
variables are explicitly overridden. That directory must be visible from both
login and compute nodes. The batch job sets Hugging Face, Transformers, and uv
to offline mode, and its preflight fails before model loading if setup did not
cache a required model snapshot or persona vector.

After copying out the results, preview removal of the large model snapshot with:

```bash
uv run steering-clean-model --model Qwen/Qwen3-32B \
  --cache-dir "$ULM_RUNTIME_ROOT/models/huggingface/hub"
```

Repeat with `--yes` to execute it. This leaves the environment, persona vectors,
J-Lens artifacts, and experiment results intact.
