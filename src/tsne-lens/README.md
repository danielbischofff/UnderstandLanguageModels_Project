# Residual-stream role map

`steering-tsne` maps raw role vectors and token residuals from the same model and post-MLP layer, defaulting to Qwen3 layer 32 (zero-based).
Inputs are a directory of role `.pt` tensors shaped `[layers, hidden]`, or NPZ `vectors` and `labels` arrays.
The [tuned schedule](tuning-notes.md) uses PCA initialization, perplexity annealing 90 → 30, early exaggeration 2, and ordinary-KL optimization until a measured plateau.

```bash
uv run --extra plots steering-tsne \
  --roles models/assistant-axis-vectors/qwen-3-32b/role_vectors \
  --default-vector models/assistant-axis-vectors/qwen-3-32b/default_vector.pt \
  --layer 32 --training-gif --formats html png \
  --output outputs/plots/qwen3-layer32-annealed
```

Open `roles.html` for three broad Ward groups, or `subclusters.html` for the six-group hierarchy cut, with individual role hover labels and highlighted core roles.
`--training-gif` saves checkpoint PNGs, a GIF, and the KL curve under `training/`; `projection.json` records metrics and convergence, while `role-map.pkl` preserves the reference for overlays.
An explicit `--perplexity` selects a fixed scale unless `--anneal-perplexity` supplies subsequent lower scales; `--help` lists iteration limits and convergence tolerance.
PNG export uses Kaleido/Chrome (`uv run --extra plots plotly_get_chrome -y` if needed); training PNGs and GIFs use Matplotlib/Pillow.

Enable `outputs.keep_activations: true` before an experiment, then pass a trial directory with its metadata, an activation manifest, or a trajectory NPZ shaped `[tokens, hidden]`.

```bash
uv run --extra plots steering-tsne \
  --map outputs/plots/qwen3-layer32-annealed/role-map.pkl \
  --trajectory /path/to/run/trials/00001_steered --name steered \
  --generated-only --output outputs/plots/qwen3-layer32-steered
```

Repeat `--trajectory` and `--name` to compare runs against unchanged role coordinates.
Treat 2D gaps qualitatively, inspect original-space neighbors, and compare KL only while the affinity is unchanged.
