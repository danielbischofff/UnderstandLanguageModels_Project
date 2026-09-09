# Command-line tools

The main entry point runs a question × character × alpha × steering-scheme experiment:

```bash
uv run steering-experiment --config configs/experiments/tylenol-demon.yaml
```

It writes generated responses, compact J-Lens top-word results, metadata, and a
resumable manifest. It writes activation tensors only when the experiment sets
`outputs.keep_activations: true`. Resume an interrupted matrix with:

```bash
uv run steering-experiment --resume path/to/run
```

Supporting commands are deliberately narrow:

- `steering-preflight` checks the GPU, storage, and runtime before loading Qwen.
- `steering-prefetch --config ...` downloads or locally verifies every model
  snapshot and persona vector needed by an experiment.
- `steering-fetch --role-direction demon` fetches the required persona vector.
- `steering-j-lens` is a low-level reader for retained activation artifacts.
- `steering-clean-model` previews or removes one cached model repository; add
  `--yes` only after checking its reported revisions and recoverable space.

## Refresh an existing report

```bash
uv run steering-report path/to/run [path/to/another/run ...]
```

Reads an existing `results.zip`, or exports per-trial JSON to that compact binary
archive first. Multi-question runs get an index and one HTML page per question.
Source checkpoints, YAML and metadata are unchanged. No inference, model loading
or network access is needed. Reload already-open report tabs afterward.

Use `--publication-dir DESTINATION` for a separate copy, or rebuild directly from
a downloaded ZIP with `steering-report results.zip --output-dir reports`.
An older download containing only `experiment-results.json` retains a legacy
single-HTML rebuild path (which loads that complete JSON document).
See [compact export details](../../../configs/README.md#compact-reports-and-publication-bundles).

The bundled [CC-CEDICT snapshot](../data/README.md) covers simplified and
traditional Chinese. Component lookups are marked approximate; mixed-script and
unknown fragments are labelled untranslated. Glosses are context-free hints,
not evidence that the model represents a particular English interpretation.
The original token remains visible. Hover/focus/tap a layer for a persistent
top-10 table, without native-tooltip delay. Probabilities are the saved J-Lens
values, not renormalized over the displayed words or translations.

Displayed glosses use at most five words. Complete short senses are preferred;
an ellipsis marks shortening and `approx.` marks component lookup. Search still
matches the fuller dictionary meanings. The desktop layout gives most of the
width to a nine-column layer overview and uses a 300-pixel readout pane on the
right. All 63 fitted layers fit at 1366 × 768; smaller screens reflow. The text
panel scrolls independently, and the plain generated answer can be expanded.
