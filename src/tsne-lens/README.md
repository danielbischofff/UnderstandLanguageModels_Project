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

`--map` never refits: the role coordinates stay fixed and only the residuals are
transformed into them, so plots from different trials are directly comparable.

## Projecting a whole experiment run

`--run` projects every completed trial of a run separately, one plot per trial,
all into the same fixed map. It requires `--map` and cannot be combined with
`--trajectory`:

```bash
uv run --extra plots tsne-lens \
  --map outputs/plots/tsne/roles/role-map.pkl \
  --run runtime/outputs/experiments/<experiment>/runs/<run-id> \
  --prompt-only --formats html png \
  --output outputs/plots/tsne/<experiment>-prompt
```

Select the token segment with `--prompt-only` (the prompt as the model saw it,
chat-template tokens included) or `--generated-only`; the default plots both.
Use `--point pre_intervention` to see the same positions before the intervention
at that layer. A run must have been captured with `outputs.activation_capture`
for generation-time residuals; `results.zip` contains no tensors, so projection
needs the original run directory.

Each trajectory is drawn as square markers in token order, labelled `position:
token`, joined by thin arrows pointing from each token to the next, over the role
scatter. Diamond `start`/`end` markers and a token-order colour bar disambiguate
direction where a path crosses itself.

Code is in `tsne_lens/`; tuning tools and tests are in `scripts/` and `tests/`.
