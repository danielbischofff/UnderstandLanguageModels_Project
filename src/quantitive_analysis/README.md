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

## Word-tokenization audit

Audit the question banks with the experiment's pinned tokenizer, without loading
model weights or changing the banks:

```bash
PYTHONPATH=src runtime/venv/bin/python -m quantitive_analysis.tokenization_audit \
  --config configs/experiments/question-bank-pilot.yaml \
  --banks configs/question_bank.yaml configs/question_bank_contrastive.yaml
```

Mappings and provenance go to `data/word_tokenization/`; the readable summary goes
to `outputs/analysis/word-tokenization/summary.md`. The validated audit mappings and summary are explicitly versioned; other
artifacts under these locations remain Git-ignored. The audit records all spacing/capitalization variants and flags
first-token proxies and collisions using the existing analysis resolver. It also
checks vocabulary and audited-encoding agreement with the J-Lens tokenizer.
This does not yet change the analysis CLI to consume mappings or filter proxies.

## Question-bank pipeline

Prepare or execute all per-question analyses with audited word selection:

```bash
PYTHONPATH=src:runtime/analysis-deps runtime/venv/bin/python -m quantitive_analysis.bank_pipeline \
  --config configs/experiments/question-bank-pilot.yaml \
  --prepare-only --layer 32 \
  --output-dir outputs/analysis/pilot-prepared
```

Use a fresh output directory for each invocation. The prepared pilot has 16 jobs:
two questions × two banks × two set types × two token policies. `complete_only`
retains words with a single-token representation; `including_proxies` retains all
original words. Selected surface variants are fixed to the audited mappings.
`manifest.json` records retained/original word counts, unique token counts, excluded
words, and provenance. `commands.json` records the individual scoring commands.
The preparation checks bank hashes, resolver hash, tokenizer hash, all selected IDs,
and model/tokenizer revision. It refuses stale audits or an existing destination.

When the pilot finishes, execute using its original run directory and a new output:

```bash
PYTHONPATH=src:runtime/analysis-deps runtime/venv/bin/python -m quantitive_analysis.bank_pipeline \
  --config configs/experiments/question-bank-pilot.yaml \
  --run RUN_DIRECTORY --layer 32 \
  --output-dir outputs/analysis/pilot-results
```

Plotly 6.9.0 is installed separately in `runtime/analysis-deps`; the commands above
include it without modifying the queued inference environment. A standard install
can instead use the project's `plots` extra. Use a CPU Slurm allocation for substantial
rank recomputation; do not assume this is cheap on a shared login node. No Qwen
weights or GPU are needed, but J-Lens/unembedding files are required. The command
validates the run's resolved configuration and trial count before scoring. Missing
activations yield cosine-only results; unavailable ranks remain blank, never zero.
Repeat `--layer` for more fitted layers or omit it for all saved readout layers.
Each scoring job uses a separate output directory. Jobs currently run sequentially;
repeated activation readout across jobs is correct but can be expensive.

Alongside the existing HTML timelines and per-position metrics, this writes:

- `response_scores.csv`: answer-position averages for every trial and word set,
  including valid-position counts. Prompt positions are excluded.
- `comparisons.csv`: intended-set score minus the same set's unsteered score;
  intended-set score minus the mean of the other seven persona-set scores within
  its bank; and the change in that margin relative to the unsteered response.
- `topic_balanced_summary.csv`: questions averaged within topics, then topics
  weighted equally, separately for each run/persona/method-strength scheme/layer,
  bank, set type and token policy. Valid-question counts accompany every measure.

Higher mean reciprocal rank and centroid cosine indicate stronger association.
The competing sets include personas not steered in the pilot. Cross-bank trials
are scored for inspection but excluded from that bank's intended-persona summary.
Controls are paired within the same experiment/run/question/readout layer. Missing
controls yield blank deltas; duplicate controls are rejected. These are descriptive
summaries, not significance tests; tokens are not independent experimental samples.
The `set_counts` in the manifest must accompany interpretation: filtering can leave
unequal set sizes and alter vocabulary content. Complete-token status can reflect
an alternate capitalization or spacing form. This pipeline does not implement
multi-token sequence probabilities, deduplicate colliding words, automatically
judge response coherence, or pool separate runs/methods into one comparison.

## Activation-to-word cosine

`activation_word_cosine` compares the J-Lens-transformed activation, after the
model's final RMSNorm (including its learned gain), with each selected bank
word's raw unembedding vector. At each position and layer it computes
`mean_w cosine(RMSNorm(J_layer @ h), U[w])`. Words receive equal weight: there
is no top-k cutoff, probability weighting, or normalization of the bank centroid.
Higher values mean stronger directional alignment, not higher probability.
Zero-vector cosines are undefined and exported as missing values.

The metric is included in CSV/JSON when retained activations are available.
Select its timeline with `--cosine-measure activation_word_cosine`; existing
plot defaults are unchanged. Saved top-word archives cannot reconstruct this
metric, so it remains unavailable there. The question-bank pipeline also computes
its response-position average, control delta, persona margin, margin delta, and
topic-balanced summaries using the same rules as MRR and centroid cosine.
Word-token resolution and proxy policies apply to this metric too.
