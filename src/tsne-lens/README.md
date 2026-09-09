# Residual-stream role map

`steering-tsne` uses openTSNE to fit a fixed 2D role map and overlay token-by-token post-MLP residuals, defaulting to Qwen3 layer 32 (zero-based).
Role inputs are raw `.pt` tensors shaped `[layers, hidden]`, labelled by filename, or NPZ files with `vectors` and `labels` arrays.
Residuals and roles must share the same model, layer, and post-MLP coordinate system; use raw role vectors rather than steering differences.

```bash
uv run --extra plots steering-tsne \
  --roles models/assistant-axis-vectors/qwen-3-32b/role_vectors \
  --layer 32 --formats html png \
  --output outputs/plots/qwen3-layer32
```

Open `outputs/plots/qwen3-layer32/roles.html` to zoom and hover over role names; the same directory contains the PNG, `projection.json`, and reusable `role-map.pkl`.
PNG export requires Chrome/Chromium, installable with `uv run --extra plots plotly_get_chrome -y`.
HDBSCAN finds clusters in the original latent space using cosine distance by default, labels each with up to three representative roles, and leaves noise unclustered.

Enable `outputs.keep_activations: true` before running an experiment, then supply a trial directory, activation manifest, or trajectory NPZ with `[tokens, hidden]` vectors.
Retain the trial's sibling `metadata.json` for token text and prompt/generated alignment, and repeat `--trajectory` with `--name` to compare runs.

```bash
uv run --extra plots steering-tsne \
  --map outputs/plots/qwen3-layer32/role-map.pkl \
  --trajectory /path/to/run/trials/00001_steered --name steered \
  --generated-only --output outputs/plots/qwen3-layer32-steered
```

Saved maps keep role positions fixed while residual trajectories add hoverable tokens, latent-space nearest roles, and playback controls.
Treat 2D gaps and path lengths qualitatively; inspect latent-space distances and use `steering-tsne --help` for metric, perplexity, cluster-size, and capture options.
