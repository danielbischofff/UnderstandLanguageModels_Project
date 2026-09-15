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

## Cluster colours

Ward numbers its clusters by merge order, so the same group of roles can be
cluster 3 in one fit and cluster 0 in the next. `tsne_lens/palette.py` therefore
sorts the clusters by their medoid — the member closest to the rest of its own
cluster, the role that opens the cluster's name — and hands out six fixed colours
in that order. A cluster's colour follows its membership, so the roles map, the
training GIF and the ribbons agree, and a refit of the same roles reproduces the
same six. For our Ward-6 cut of the 275 role vectors that fixes: 0 *fool* green,
1 *writer* vermillion, 2 *ancient* orange, 3 *synthesizer* blue, 4 *fixer* sky,
5 *retiree* purple. The order is not preserved across different cuts of the
hierarchy — at Ward 5 two clusters merge, one medoid disappears, and the slots
below it shift up by one.

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

## Neighbourhood-cluster ribbons

One plot per trial is unreadable across a 103-trial sweep, so
`scripts/cluster_ribbons.py` drops the 2D positions and asks a discrete question
per token instead: **is the residual actually close to any role vector, and if so
whose cluster?** Neighbours are cosine, taken in the original hidden space
against the role vectors in the saved map — the embedding is only the source of
the clustering.

A cell is coloured only if at least one of the token's ten nearest role vectors
clears `--min-similarity` (0.5 by default); otherwise it stays white. The point
is the evidence that a residual really sits near the roles, not which of six
distant clusters happens to be least distant: under strong additive steering
whole rows go white, because nothing in the map is near them any more. Among the
neighbours that do clear the bar, the cluster holding most of them takes the
cell, and the tint is the strongest similarity inside that cluster, on a fixed
0.5-to-0.9 scale so two figures are comparable.

Each trial becomes one horizontal ribbon of token colours, and trials whose
steered character shares a Ward cluster are stacked into one figure, so a single
plot covers a cluster's whole steering sweep across every run, injection layer
and coefficient. Red dots in the right gutter mark trials the judge graded 2
(harmful), amber dots 1 (ignorant); `own` is the share of the row's tokens
coloured with the steered character's own cluster and `near` the share coloured
at all.

```bash
uv run --extra plots python src/tsne-lens/scripts/cluster_ribbons.py \
  --map outputs/plots/tsne/roles/role-map.pkl \
  --run alpha=runtime/outputs/experiments/<experiment>/<sweep-a>/runs/<run-id> \
  --run band=runtime/outputs/experiments/<experiment>/<sweep-b>/runs/<run-id> \
  --prompt-only --neighbors 10 --min-similarity 0.5 \
  --output outputs/plots/tsne/roles-cluster-ribbons
```

`--run LABEL=DIR` is repeatable and the label is what each row shows; every run
must share one token sequence. Judgments are read from
`<run>/judging/outcomes.json` unless `--outcomes LABEL=FILE` overrides it.
The run writes `ribbons.json` with each token's top-k neighbour table and the
cell it was given — pass it back as `--from-json` to restyle the plots, or to try
another `--min-similarity`, without touching the tensors again.
`--store-neighbors` also keeps the labelled neighbour lists in that file.

Current result for `17roles-tylenol`:
[roles-cluster-ribbons/index.html](../../outputs/plots/tsne/roles-cluster-ribbons/index.html).

Code is in `tsne_lens/`; tuning tools and tests are in `scripts/` and `tests/`.
