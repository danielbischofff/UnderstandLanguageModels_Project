# Quantitative analysis

Run `uv run --extra plots quantitive-analysis RUN --sets WORDS.yaml --layer 32`.
`RUN` accepts a run directory or `results.zip`; repeat `--layer` or omit it for
all saved layers. Use `--question ID` to select a question and `--output-dir DIR`
to choose the destination (default: `outputs/plots/quantitive-analysis`).

Set files contain a list of `{id, label, words}` mappings; see
[the Tylenol example](examples/tylenol.yaml). Edit or replace them and rerun
without changing experiment YAML or regenerating outputs.

The CPU analysis uses the fetched J-Lens and unembedding (`--j-lens-data`,
`--lens`, `--analysis-root`). Saved activations provide full-vocabulary ranks;
saved top words provide probability-weighted cosine similarity to each set.
Ranks are unavailable without activations. `--top-k` is capped by saved top-word
counts for cosine-only runs. Multi-token words use their first token as a proxy.

Outputs are interactive HTML timelines, an index, `metrics.csv` and `metrics.json`,
under `<output-dir>/<experiment>/<run>/`. Each rerun replaces these analysis files.
