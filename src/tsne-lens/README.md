# Residual-stream role map

`steering-tsne` maps raw role vectors and token residuals from the same model and post-MLP layer, defaulting to Qwen3 layer 32 (zero-based).
Inputs are a directory of role `.pt` tensors shaped `[layers, hidden]`, or NPZ `vectors` and `labels` arrays.
The [tuned schedule](tuning-notes.md) uses PCA initialization, one fixed combined 15/90 Gaussian affinity objective, exaggeration 2 followed by a 125-iteration ramp to 1, and ordinary-KL optimization until a measured plateau.

```bash
uv run --extra plots steering-tsne \
  --roles models/assistant-axis-vectors/qwen-3-32b/role_vectors \
  --default-vector models/assistant-axis-vectors/qwen-3-32b/default_vector.pt \
  --layer 32 --training-gif --formats html png \
  --output outputs/plots/tsne/roles
```

Open `roles.html` for six Ward groups with hover labels, three central roles labeled per cluster, and persistent labels for assistant, demon, bard, criminal and the starred default vector.
`--training-gif` saves checkpoint coordinates, an adaptive-scale GIF and a KL curve under `training/`, without individual frame PNGs; `projection.json` records metrics and convergence, while `role-map.pkl` preserves the reference for overlays.
`--perplexity 15 90` averages both affinity scales throughout training; `--exaggeration-anneal-iter` controls the ramp, with 0 disabling it.
PNG export uses Kaleido/Chrome (`uv run --extra plots plotly_get_chrome -y` if needed); training PNGs and GIFs use Matplotlib/Pillow.

Enable `outputs.keep_activations: true` before an experiment, then pass a trial directory with its metadata, an activation manifest, or a trajectory NPZ shaped `[tokens, hidden]`.

```bash
uv run --extra plots steering-tsne \
  --map outputs/plots/tsne/roles/role-map.pkl \
  --trajectory /path/to/run/trials/00001_steered --name steered \
  --generated-only --output outputs/plots/qwen3-layer32-steered
```

Repeat `--trajectory` and `--name` to compare runs against unchanged role coordinates.
Treat 2D gaps qualitatively, inspect original-space neighbors, and compare KL only while the affinity is unchanged.
