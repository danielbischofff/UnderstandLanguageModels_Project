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

Reads each run's `experiment-results.json` and rebuilds only `report.html` with
the current UI and offline Chinese glosses. No model is loaded, no inference or
network request runs, and the saved JSON, resolved YAML, and metadata are not
changed. Reload any already-open report tab afterward. The same renderer is
used automatically for future experiments.

The bundled [CC-CEDICT snapshot](../data/README.md) covers simplified and
traditional Chinese. Component lookups are marked approximate; mixed-script and
unknown fragments are labelled untranslated. Glosses are context-free hints,
not evidence that the model represents a particular English interpretation.
The original token remains visible. Hover/focus/tap a layer for a persistent
top-10 table, without native-tooltip delay. Probabilities are the saved J-Lens
values, not renormalized over the displayed words or translations.
