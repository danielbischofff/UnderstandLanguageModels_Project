# Slurm

Run setup on an internet-connected login node, then submit from the repository
root:

```bash
bash setup.sh configs/experiments/tylenol-demon.yaml
sbatch slurm/experiment.sbatch configs/experiments/tylenol-demon.yaml
```

## GPU allocation and preflight

Select a partition/GPU with **at least 75 GB of VRAM** for the current
Qwen3-32B bfloat16 experiments. This is a model/setup requirement, not a
universal requirement for every model. `--mem=96G` requests host RAM, **not**
GPU memory; `--gres=gpu:1` alone can allocate an undersized GPU.
The site's resource flags must constrain the allocation before submission.
For example, Helix supports:

```bash
sbatch --partition=gpu-single --gres=gpu:1,gpumem_per_gpu:75GB \
  --time=01:30:00 slurm/experiment.sbatch configs/experiments/tylenol-demon.yaml
```

Add your site's account flag if required. `gpumem_per_gpu` is site-specific;
on other clusters request an appropriate GPU type/partition instead.
`steering-preflight` then checks GPU memory, free disk (80 GB by default), and
cached model/vector/lens prerequisites before expensive model loading. It
cannot repair an undersized allocation. Override `STEERING_MIN_VRAM_GB` only
when deliberately running a configuration with different requirements.

Inspect queue state with `squeue -u "$USER"`, and finished jobs with
`sacct -j JOB_ID --format=JobID,State,Elapsed,MaxRSS`. The log at
`runtime/logs/experiment-JOB_ID.out` records preflight, progress and output paths.
Keep the submitted checkout and environment unchanged while jobs use them.

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

## CPU-only report export

Use `steering-report RUN --publication-dir DESTINATION` after inference to make
compact downloadable results and per-question pages; see [the format and rebuild
instructions](../configs/README.md#compact-reports-and-publication-bundles).
No model or GPU is needed. Conversion processes one trial at a time and renders
one question at a time, rather than loading a whole suite's combined JSON.
For large suites, use a CPU batch allocation under your site's policy instead
of assuming a shared login node can tolerate the work. Measure with
`/usr/bin/time -v` (elapsed time and maximum resident memory) before reducing
resource requests. There is no measured universal runtime estimate.

On Helix, after setup, submit from the repository root (add your account flag):

```bash
sbatch --partition=cpu-single slurm/report.sbatch RUN_DIRECTORY \
  --publication-dir runtime/publication-candidates
```

The wrapper requests one CPU and 8 GB host RAM, limits numerical-library
threads, and records elapsed time. Query peak memory afterward with
`sacct -j JOB_ID --format=JobID,State,Elapsed,MaxRSS`. For an isolated report-code
checkout, set `ULM_REPO_ROOT` to it and `ULM_RUNTIME_ROOT` to the existing runtime;
this reuses the environment without modifying it.

Measured on Helix: exporting the 140-trial band and 98-trial alpha suites
together (28 question pages) took **7m10s elapsed**, about **148 CPU seconds**,
and **621 MiB peak host RAM** on one CPU. This is a reference measurement, not
a runtime guarantee; shared filesystem performance and output size matter.
Report-only imports do not load Torch or Transformers. The original inference
results remain untouched, and the portable ZIPs can rebuild the pages offline.

Keep personal connection details (WSL/SSH invocation, username, VPN, account,
absolute paths and copy commands) in the ignored `runtime/helix-runbook.md`,
not in these shared instructions. This separates reproducible job requirements
from one person's access setup.
