# Local readout of the mini-batch export

The extracted mini-batch export has retained cached activations but no saved
J-Lens top words. `local_readout` computes exactly the two requested metrics from
these activations without running the language model or generating new answers.
It leaves the generic plotting/pipeline reader unchanged.

```bash
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 PYTHONPATH=src \
  .venv/bin/python -u -m quantitive_analysis.local_readout
```

Defaults: `outputs/mini_batch_exp_results`, assets in
`runtime/models/j-lens/qwen3-32b`, four readout layers (32, 29, 35, 62), both banks,
both set types, and complete-token and proxy-inclusive policies. Jobs keep each
question's control before its steered trials. No top-k approximation is used.
`--limit 1` runs one new trial/layer job for benchmarking; `--output` selects a
separate destination. Run the same command to resume interrupted work. The output
lock rejects simultaneous writers; signatures reject changed inputs, assets,
word mappings or implementation. Use a fresh output directory when changing the
implementation. Do not change these inputs while a run is active.

The actual model tokenizer revision is pinned from the run. The downloaded
J-Lens checkpoint is pinned to repository revision
`b25d72a96b79c8e309d6625955a98751da47e67a` and its file SHA-256 is
`0af5ccf2e1c251bedc202607a7ca214f3b9f60980bab73d67a3d0b887db213d8`.
The fitted lens reports Qwen3-32B, 557 fitting prompts and target layer 63;
its metadata does not specify a Qwen model commit. The model ID agrees with this
run, but exact fit-model revision agreement cannot be verified from that metadata.
The unembedding and RMSNorm tensors use the experiment's pinned model commit.

Outputs under `outputs/analysis/mini-batch-local-readout`:

- `provenance.json`: input and asset hashes, tokenizer revalidation, metric conventions.
- `NNNN-layer-L.npz`: absolute input positions, selected word token IDs, full-vocabulary
  ranks and individual word cosines. This is the sufficient readout for these
  metrics; full vocabulary logits and top-word display lists are not persisted.
- `NNNN-layer-L.json`: word-set definitions, counts and prompt/answer scores.
- `comparisons.json`: paired control deltas, margins against all seven other
  personas in the same bank, and margin deltas.
- `topic-balanced-summary.json`: equal-topic summaries, with valid-question counts;
  includes a sensitivity view excluding calibration questions Q01 and Q23.
- `summary.md`: readable main comparison, with explicit partial coverage.
- `progress.json` and, for the launched background run, `run.log` and `process.pid`.

Summaries refresh after the first two jobs, every 33 new jobs and at normal exit.
They can also be refreshed separately while the worker continues:

```bash
PYTHONPATH=src .venv/bin/python -m quantitive_analysis.local_summary \
  outputs/analysis/mini-batch-local-readout
```

Only fully written job JSON files are summarized. An NPZ without its accompanying
JSON is recomputed on resume. Missing controls yield missing deltas, not zero.

The first layer-32 control took 10.9 seconds after asset loading on the local
16 GB Intel i7 machine. 1,584 jobs at that rate take about 4.8 hours, plus startup,
summary generation and variation in throughput. Keep the machine awake for a
continuous run; interrupted runs can resume. There is no model inference or cloud
compute in this step.

Interpretation: answer metrics use activations of generated input tokens, so the
last output token is absent (normal for cached generation). Prompt scores are
stored separately. Both metrics average the selected words equally, retaining
word multiplicity when different words map to the same token. MRR is the average
of reciprocal ranks, where ties receive the best rank. Cosine uses the transported
activation after final RMSNorm with learned gain. Zero-vector cosines remain
undefined. Use word counts, cap flags and output-quality diagnostics alongside
association scores; 379/396 responses reached the generation cap.
