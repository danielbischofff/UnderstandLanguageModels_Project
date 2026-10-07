# Mini-batch J-Lens readout

This directory contains the completed CPU readout summary for
`outputs/mini_batch_exp_results`: 396 trials, four layers (29, 32, 35 and 62),
exact full-vocabulary mean reciprocal rank (MRR), and activationWordCosine.

Open [`index.html`](index.html) locally for the filterable overview. It is a
dependency-free HTML file and embeds the compact comparison data. The two JSON
files contain the source data used by the page:

- `comparisons.json`: per-question paired control deltas and persona margins.
- `topic-balanced-summary.json`: topic-balanced summaries, including the
  calibration-question sensitivity view.

The large per-trial `.npz` readouts and raw activation export remain on Helix;
they are intentionally not committed here. Answer-position scores omit the
final generated token because it has no cached input activation.
