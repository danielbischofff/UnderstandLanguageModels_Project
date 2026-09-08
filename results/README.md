# Published experiment results

Convention: `configs/experiments/<name>.yaml` has `name: <name>`, and its
published runs live under `results/<name>/<run-id>/`. Preserve the original
experiment YAML when adding a new experiment, and keep each run's resolved
configuration alongside its results.

## Tylenol / demon: initial experiment

[Browser report](tylenol-demon/0001_20260907T205129Z_e2191eaef3780817/report.html)
· [Machine-readable results](tylenol-demon/0001_20260907T205129Z_e2191eaef3780817/experiment-results.json)
· [Exact experiment configuration](tylenol-demon/0001_20260907T205129Z_e2191eaef3780817/resolved-config.yaml)

Download or clone the repository and open `report.html` locally. GitHub does not
render the interactive HTML report. It is self-contained and needs no server or
model download.

This run used the **1000 mg** prompt, Qwen3-32B in bfloat16, and compared an
unsteered control with demon steering at alpha 1 on layer 32. It completed on
2026-09-07 as Helix job `14597147` on an A100-80GB in 2 minutes 56 seconds.
Both trials retain top-10 J-Lens words at 175 token positions across fitted
layers 0–62. Activations were discarded after readout.

Both generated answers expressed concern rather than malice. Both reached the
128-token generation limit and are incomplete. This is a qualitative observation
about the generated text; the J-Lens viewpoint shift has not been evaluated.

The run was submitted from commit `3744bb3`. Automatic Git provenance fields in
the output are null; the commit is recorded here from the deployment record.
The original manifest and per-trial metadata are included. Per-trial `result.json`
copies are omitted because `experiment-results.json` already contains both complete
results. The manifest's trial paths refer to those original trial directories;
their metadata remains available under the same paths. The report embeds the
same results for offline browsing.

The original [tylenol-demon.yaml](../configs/experiments/tylenol-demon.yaml)
is preserved. The **8000 mg** experiment is defined separately in
[tylenol-demon-8000mg-alpha-sweep.yaml](../configs/experiments/tylenol-demon-8000mg-alpha-sweep.yaml);
its published results belong under `results/tylenol-demon-8000mg-alpha-sweep/`.
