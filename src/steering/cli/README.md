# Command-line tools

The main entry point runs a complete question-by-character experiment:

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
