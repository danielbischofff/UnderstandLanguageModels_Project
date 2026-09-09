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

The HTML reports were refreshed on 2026-09-08 with offline Chinese dictionary
glosses and instant top-10 tables. Their embedded experiment data is unchanged;
all results JSON, resolved configurations, manifests, and metadata retain the
original run content. Dictionary-derived annotations are separate from model
outputs and attributed under CC BY-SA 4.0 in the report. Select a layer by hover,
keyboard focus, or tap to update the table; English-gloss search is also supported.

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

## Tylenol / demon: 8000 mg alpha sweep

[Browser report](tylenol-demon-8000mg-alpha-sweep/0001_20260908T101330Z_0b9ece86a83844be/report.html)
· [Machine-readable results](tylenol-demon-8000mg-alpha-sweep/0001_20260908T101330Z_0b9ece86a83844be/experiment-results.json)
· [Exact experiment configuration](tylenol-demon-8000mg-alpha-sweep/0001_20260908T101330Z_0b9ece86a83844be/resolved-config.yaml)

Open the downloaded report locally and select **demon** to enable the alpha
selector. The default unsteered control has no alpha or steering scheme.

This run used the **8000 mg** prompt and Qwen3-32B in bfloat16. It contains one
unsteered control and seven demon trials at alpha 1–2.5 in steps of 0.25, each
steered simultaneously at layers 29, 32, and 35. It completed on 2026-09-08 as
Helix job `14687548` on an H200 in 3 minutes 22 seconds, including model loading.
All eight trials retain top-10 J-Lens words at 175 positions (47 prompt and 128
generated tokens) across fitted layers 0–62. Activations were discarded.

The control expressed concern. Alpha 1 still urged emergency help, but in a
demonic voice; 1.25–1.75 became increasingly theatrical, with dangerous
minimization in some responses. At 2–2.5 the output collapsed into repetitive
loops. All responses reached the 128-token limit. These are qualitative
observations about generated text, not medical guidance or a completed J-Lens
viewpoint analysis.

The deployed source was commit `6a3d8f1`, recorded from deployment because the
automatic Git provenance fields are null. Data artifacts are unchanged copies of
the completed run; only the report presentation was refreshed as noted above.
As above, per-trial metadata is retained while duplicate
`result.json` files and the runtime writer lock are omitted. The aggregate JSON
contains all eight complete results, also embedded in the self-contained HTML.
