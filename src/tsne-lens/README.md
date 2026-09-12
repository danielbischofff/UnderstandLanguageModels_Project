# t-SNE lens

`tsne-lens` fits a 2D role-vector map and projects post-MLP residual streams into that fixed map.

```bash
uv sync --extra plots
uv run --extra plots tsne-lens \
  --roles models/assistant-axis-vectors/qwen-3-32b/role_vectors \
  --default-vector models/assistant-axis-vectors/qwen-3-32b/default_vector.pt \
  --layer 32 --training-gif --formats html png \
  --output outputs/plots/tsne/roles
```

Defaults: PCA, cosine distance, fixed perplexities `15/25/90`, heavy tails (`dof=0.7`), Ward6, and exaggeration `2 → 1` over 125 iterations. The practical budget is about **3,375 total iterations**; later optimization changes the map negligibly.

Current layer-32 result: **50.4% recall@10**, **0.406 Ward6 silhouette**, **0.746 global distance correlation**, and **0.935 trustworthiness@10**. Open [roles.html](../../outputs/plots/tsne/roles/roles.html), [training.gif](../../outputs/plots/tsne/roles/training/training.gif), or inspect `projection.json` for full settings and metrics.

Project a captured experiment into the saved map:

```bash
uv run --extra plots tsne-lens \
  --map outputs/plots/tsne/roles/role-map.pkl \
  --trajectory /path/to/trial --name steered --generated-only \
  --output outputs/plots/tsne/steered
```

Code is in `tsne_lens/`; tuning tools and tests are in `scripts/` and `tests/`.
