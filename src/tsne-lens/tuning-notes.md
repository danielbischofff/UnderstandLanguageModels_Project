# Qwen3 layer-32 tuning log — 2026-09-09

The recommended [interactive map](../../outputs/plots/qwen3-layer32-tuned/roles.html)
and [PNG](../../outputs/plots/qwen3-layer32-tuned/roles.png) contain all 275 raw role
vectors at zero-based post-MLP layer 32. The original map remains in
`outputs/plots/qwen3-layer32/`; residual overlays should load the new
`outputs/plots/qwen3-layer32-tuned/role-map.pkl` to use the tuned coordinates.

## Selected settings

Use cosine distance, PCA initialization, exact neighbors and repulsion (`theta=0`),
multiscale perplexities **15 and 90**, seed **42**, and automatic learning rates.
Optimize for **500 iterations at exaggeration 12**, **750 at 1.5**, then **1500 at 1**.
The final ordinary-KL pass releases the crowding introduced by exaggeration while
retaining a useful layout; these are the CLI and Python API defaults.

```bash
uv run --extra plots steering-tsne \
  --roles models/assistant-axis-vectors/qwen-3-32b/role_vectors \
  --layer 32 --perplexity 15 90 --seed 42 --theta 0 \
  --early-exaggeration 12 --early-exaggeration-iter 500 \
  --exaggeration 1.5 --n-iter 750 --relaxation-iter 1500 \
  --cluster-selection-method leaf --min-cluster-size 3 --min-samples 2 \
  --formats html png --output outputs/plots/qwen3-layer32-tuned
```

The [openTSNE global-structure example](https://opentsne.readthedocs.io/en/stable/examples/03_preserving_global_structure/03_preserving_global_structure.html)
motivated testing multiscale affinities and high-to-low perplexity annealing.
Its large-data perplexities do not transfer directly to 275 roles: 90 already covers
roughly a third of this set.
The [optimization guide](https://opentsne.readthedocs.io/en/stable/parameters.html#optimization-parameters)
motivated testing exaggeration schedules while retaining automatic learning rates
and momentum 0.8; stronger exaggeration proved unsuitable as a permanent setting here.

## Measurements and visual inspection

Evaluated **115 maps**, including the baseline, refinement continuations and final
export, plus **32 clustering configurations**. Sweeps covered single perplexities
10–90; multiscale pairs from 5/50 to 30/90; main exaggerations 1–4; early exaggerations
4, 12, 24 and 32; longer optimization, exact repulsion, annealing, final relaxation,
and multiple seeds. PNG contact sheets were inspected between the broad sweep,
refinement and final selection, followed by full-size final PNG and browser checks.

KL below is recomputed exactly as ordinary `sum(P * log(P/Q))`, even for exaggerated
maps. **Native KL changes its target when perplexity changes and cannot rank those
maps directly.** The common-P30 column evaluates every layout against the *same*
original perplexity-30 affinity matrix. Recall is the fraction of original cosine
neighbors retained in 2D; global correlation is Spearman over all pairwise distances.
These diagnostics measure fidelity to these vectors, not semantic ground truth.

| Map | Native KL | Common-P30 KL ↓ | Recall@5 ↑ | Recall@10 ↑ | Trust@10 ↑ | Global correlation ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original P30 | 0.6278 | 0.6278 | 0.5018 | 0.4967 | 0.9336 | 0.7240 |
| **Selected multiscale + relaxation** | **0.4346** | **0.6251** | **0.5098** | **0.4836** | **0.9294** | **0.7553** |
| P30, early exaggeration 4 | 0.6037 | 0.6037 | 0.4829 | 0.4796 | 0.9337 | 0.7649 |
| P90, main exaggeration 1 | 0.2672 | 0.8665 | 0.3127 | 0.3847 | 0.9190 | 0.7992 |
| P30, main exaggeration 4 | 1.4552 | 1.4552 | 0.2182 | 0.2942 | 0.8819 | 0.7152 |
| P10, main exaggeration 1.5 | 1.1149 | 0.6971 | 0.5178 | 0.4833 | 0.9234 | 0.7069 |

The selected result is a compromise, not an improvement on every metric. Its common
KL improvement is small, while its global correlation and immediate neighborhoods
improve; ten-neighbor recall and trustworthiness decrease slightly. The lower native
KL alone is **not** evidence of improvement over the baseline.

Visual findings:

- High single perplexity blends nearby subgroups into a broad continuous cloud;
  its attractive native KL conceals substantial loss of local neighbors.
- Permanent exaggeration 2–4 compresses groups and eventually produces narrow bands,
  rather than faithfully separating the small semantic neighborhoods.
- P10 exposes the most separated local islands but weakens broad relationships;
  this was rejected as the default despite its stronger visual cluster separation.
- Multiscale 15/90 retains broad ordering and small neighborhoods. A moderate main
  exaggeration followed by relaxation lowers its own KL from approximately 0.441
  to 0.435; simply extending an already converged run makes little difference.
- The selected map's separation of a fixed 12-group latent reference improves
  slightly: 2D silhouette 0.4644 → 0.4765. This reference is held constant across
  the sweep and is distinct from the final 19-group labeling.

For a stricter KL-first alternative, inspect
[the P30 low-KL HTML](../../outputs/plots/qwen3-layer32-lowkl/roles.html): use
`--perplexity 30 --early-exaggeration 4 --exaggeration 1 --n-iter 1800 --relaxation-iter 0`.
It lowers the same-target KL more, but loses some immediate-neighbor retention and
small-group separation (fixed-reference 2D silhouette 0.406), so it was not selected
as the balanced default. Rotations, gaps and island sizes are not calibrated latent
distances in either map.
The separately regenerated low-KL CLI artifact records KL 0.6039 and global
correlation 0.7619; the table's 0.6037/0.7649 values are from the exploratory run.

Across six seeds (1, 7, 13, 21, 42, 73), the selected schedule gave native KL
0.4215–0.4404, common-P30 KL 0.6116–0.6297, recall@5 0.4996–0.5113, and global correlation
0.7347–0.7755. Seed 42 remains the default as a representative compromise, rather
than choosing a seed solely for the lowest KL; exact coordinates can vary with the
numerical environment, so reuse the saved map for overlays.

## Smaller clusters

Projection tuning cannot change labels computed in the original latent space.
The former HDBSCAN EOM selection (`min_cluster_size=5`, default `min_samples=5`)
returned two broad groups, so clustering was tuned separately.
[Leaf selection](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.HDBSCAN.html)
with minimum size 3 and `min_samples=2` finds **19 groups covering 92/275 roles**,
with original-space silhouette **0.3602** and final-map silhouette **0.4952** on those
92 roles. The **183 noise roles remain unassigned**; these silhouettes exclude them.
Minimum size 4 yields 12 groups covering 80 roles with original silhouette 0.3752;
size 3 was chosen to retain meaningful triads rather than maximize silhouette alone.

Examples include instructor/teacher/tutor, caregiver/counselor/therapist,
criminal/hacker/spy, adolescent/procrastinator/teenager, and guru/pilgrim/sage.
Numbered map labels and unique colors keep the 19 groups readable; the legend gives
representative role names and point hover gives the individual role.

In 50 random 90%-subsamples, original within-group pairs that were still present
remained co-clustered 81.8–100% of the time across groups. This is a sensitivity
check, not a probability of semantic correctness; the smallest groups can disappear
when members are removed. Full memberships and retention counts are logged.

## Artifacts and reproduction

The vector source is `lu-christina/assistant-axis-vectors`, revision
`3b3b788432ad33e3a28d9ff08e88a530c0740814`, using the same vectors as the baseline.
The final `projection.json` records all effective settings, package versions, native
KL, neighborhood metrics, coverage and silhouettes. Environment: openTSNE 1.0.4,
NumPy 2.4.6, scikit-learn 1.9.0 and Plotly 6.9.0.

`outputs/plots/qwen3-layer32-tuning/` contains `results.csv`, `results.json`, per-map
metrics/coordinates, `comparison.png`, `clustering.json`, `cluster-stability.json`,
and the exact exploratory scripts/logs. The contact sheet uses fixed 12-group colors
to isolate changes in the projection; the final maps use the selected 19 groups.
Trial coordinates are retained as compact NPZ files; intermediate full embedding
pickles were removed after selection, while the original and final maps remain reusable.
Run the scripts from the repository root using `.venv/bin/python`, in order:
`tune.py`, `refine.py`, `finalists.py`, `polish.py`, `stability.py`,
`cluster_stability.py`, then `summarize.py`; each script lives in that output directory.
Use `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1` for those exploratory runs.

Validation: 111 local tests passed (GPU/download tests excluded), including exact-KL
calculation, multiscale save/load, fixed anchors and independent residual projection;
the final HTML was checked in Chromium for all 275 unique hover labels and 19 groups.
