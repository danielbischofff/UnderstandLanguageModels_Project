# Residual-stream role map

A small CPU-only, file-based pipeline for roughly 200 role vectors. The implementation
is in `src/steering/tsne_lens/`; run `steering-tsne` or the compatibility entry point
`python src/tsne-lens/pipeline.py`. No database, Quadfeather, label propagation,
incremental ingestion, SVD, or large-dataset checkpoint machinery is needed.

## Fit a role map

```bash
uv sync --extra plots
uv run --extra plots steering-tsne \
  --roles /path/to/qwen-3-32b/role_vectors \
  --model-id Qwen/Qwen3-32B --layer 32 \
  --output outputs/plots/qwen3-layer32
```

The directory contains the published **raw role vectors** (`RoleVector`, `R_role`),
one `.pt` file per role, each shaped `[layers, hidden]`. Filenames become labels.
Supported tensor dictionaries use `vector`, `axis`, `vectors`, or `tensor`, matching
the project's vector store. Loading is CPU-only with `weights_only=True`.
Use the role centroid itself, not the steering difference `R_role - D` or a
role-specific contrast: residual states and direction vectors have different origins.
All inputs must come from the same model, zero-based layer, and post-MLP block output.
Layer 32 is the Qwen3 default here; choose explicitly for other models. A local tensor
directory has no identity metadata: `--model-id` asserts its provenance.

For portable input, use an NPZ with:

- `vectors`: `[roles, hidden]` at the selected layer, or `[roles, layers, hidden]`;
- `labels`: one unique nonempty Unicode string per row;
- optional scalar `model_id` and `layer`, checked against the requested map.

```python
np.savez("roles.npz", vectors=role_matrix, labels=np.asarray(role_names, dtype=str),
         model_id="Qwen/Qwen3-32B", layer=32)
```

The pipeline uses openTSNE with exact neighbors, PCA initialization (only an initial
2D placement, not a feature reduction), seed 42, and one worker. Default perplexity
is 30, capped at `(number_of_roles - 1) / 3` for small inputs. `--metric cosine`
compares direction; `--metric euclidean` includes magnitude. No centering, whitening,
or normalization is applied to the input vectors. The chosen metric is used for
both fitting and residual neighbors.

HDBSCAN searches for clusters **in the original latent space**, using that metric.
`--min-cluster-size 5` controls the minimum size; noise remains “Unclustered”. It does
not force a chosen number of groups or assign every role a cluster. Each group gets
the comma-separated names of up to three members with the smallest mean within-group
distance. These are representative role names, not generated semantic summaries.
Changing `--perplexity` or `--seed` only changes the map, not the clustering input.

## Add experiment trajectories

Enable retention in a new experiment configuration before running it:

```yaml
outputs:
  keep_activations: true
```

The experiment captures the fitted J-Lens's available source layers; the selected
layer must be among them. Existing compact J-Lens
readouts do not contain enough information to reconstruct residuals. The checked-in
result archives currently have no activation artifacts.

```bash
uv run --extra plots steering-tsne \
  --map outputs/plots/qwen3-layer32/role-map.pkl \
  --trajectory /path/to/run/trials/00000_control --name unsteered \
  --trajectory /path/to/run/trials/00001_steered --name steered \
  --generated-only --formats html png \
  --output outputs/plots/qwen3-layer32-comparison
```

You can also supply `--trajectory` while fitting with `--roles`. Each trajectory
argument accepts a trial directory, its `activations/` directory, its activation
`manifest.json`, or an NPZ. For archived runs with retained tensors, extract the
trial's `provenance/trials/<trial>/activations/` and sibling `metadata.json` first.
The default capture point is `post_intervention`; use `--point pre_intervention`
if that boundary was recorded. `--batch-index` selects one batch, never joining
unrelated sequences. Model identity is checked when trial metadata is present.

The experiment runner saves a **causal replay of the full prompt and answer** after
generation. Its event phase is `prefill`, even at answer positions; the loader uses
the prompt length from `metadata.json` to classify prompt/generated tokens. Low-level
cached generation captures are also supported, ordered by forward index. Hover
labels describe the **input token whose residual was captured**, not the token
predicted next. The last sampled token may be absent from a live cached capture
because it was never fed back; the replay includes it. Missing cache positions stay
unknown rather than being invented. Token strings are taken from metadata; no
model/tokenizer downloads are needed.

A trajectory NPZ has:

- `vectors`: `[tokens, hidden]` at the map's layer, in processing order;
- optional `tokens`, `token_ids`, `positions`, `segments`: one value per row;
- `segments`, if supplied: `prompt`, `generated`, or `unknown`;
- optional scalar `model_id`, `layer`, and `point` (checked when present).

`--generated-only` needs known segment labels; for causal replay it needs the sibling
trial metadata. Without metadata, the NPZ row number is the position and token text
is blank. For an NPZ the caller is responsible for selecting its batch and capture
point. No arbitrary flattening of tensor axes is performed.

## Outputs and interpretation

Every output directory contains:

- `role-map.pkl`: the fitted reference, role vectors, labels, clusters and settings;
- `projection.json`: role coordinates, cluster labels, configuration, source paths,
  and all token coordinates/metadata, norms and nearest-role distances;
- `roles.html` by default: a self-contained offline Plotly view with zoom, hover,
  legend toggles, cluster annotations, trajectories, start/end markers and a
  playback cursor/slider;
- `roles.png` with `--formats png` or `--formats html png`: a static overview.

PNG export uses Kaleido and needs Chrome/Chromium. Install it once if unavailable:

```bash
uv run --extra plots plotly_get_chrome -y
```

HTML works without Chrome. If PNG export fails, the map, JSON and requested HTML
remain available. Rerun with the saved map and the same trajectories to retry PNG.
Pickled maps must be trusted local files; keep the same locked dependency environment
when loading them. Coordinates and summaries can be read without pickle from JSON.

Residuals are optimized independently against the fixed role anchors using
[openTSNE's partial embedding API](https://opentsne.readthedocs.io/en/latest/api/index.html),
with median initialization and `--transform-perplexity 5`. This avoids `transform()`'s
in-place recentering, keeping map coordinates unchanged across overlays. No temporal
smoothing or residual-to-residual attraction is added. All requested trajectories
share the same reference; loading `--map` reuses its saved fit settings. The slider
indexes rows within each filtered trajectory, not aligned words across answers.

Hover lists and JSON report exact nearest roles in the original latent space
(default `--top-k 5`). A residual is always placeable even when far from every role;
its neighbors are similarities, not a role classification. Individual token states
also have a different distribution from role-averaged activations. Inspect distances
and norms alongside the map. t-SNE gaps, island sizes and trajectory lengths do not
measure latent distance or drift magnitude. Cluster labels describe the role set
only and are not propagated onto residuals. Compare several seeds/perplexities when
assessing whether visible structure is stable. Static export follows
[Plotly's Kaleido workflow](https://plotly.com/python/static-image-export/).
