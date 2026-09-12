# Recall search findings

80 trials varied fixed perplexity scales, kernel tail weight (`dof`), seeds, early exaggeration and retained final exaggeration; PCA, cosine neighbors and original-space Ward6 assignments stayed fixed.

| Selection | Recall@10 | Ward6 silhouette | Global correlation | Perplexity / dof |
|---|---:|---:|---:|---|
| Current map | 48.25% | 0.359 | 0.768 | 15+90 / 1 |
| Recall winner | **51.16%** | 0.435 | 0.697 | 22 / 0.6 |
| Joint winner | 50.62% | **0.456** | 0.691 | 25 / 0.5 |

- The gains are **2.91 / 2.36 percentage points**, with a real loss of global structure. Recall@5 also improves (49.75% → 53.24% / 53.16%); recall@20 improves only slightly. The smallest Ward group loses neighbors, so gains are not uniform.
- Heavier tails produce tighter islands and clearer boundaries. Strong retained exaggeration generally sacrificed recall; both winners finish at exaggeration 1 after the existing 2 → 1 ramp.
- Joint selection maximizes the geometric mean of relative recall/silhouette gains, requiring both to beat baseline. Resumed ancestors are excluded; this is a bounded, in-sample search, not a proven optimum.
- Winners ran **14,375 / 13,375 iterations**. Recall is stable over the last five checkpoints (joint varies by 0.04 percentage points), but KL changes of **1.5e-5 / 3.2e-5** still exceed the strict 1e-6 tolerance. Both are explicitly marked nonconverged; KL was not used to rank different objectives.

[Recall HTML](../../outputs/plots/tsne/roles-exp/recall/roles.html) · [GIF](../../outputs/plots/tsne/roles-exp/recall/training/training.gif) · [Joint HTML](../../outputs/plots/tsne/roles-exp/joint/roles.html) · [GIF](../../outputs/plots/tsne/roles-exp/joint/training/training.gif)

[All metrics](../../outputs/plots/tsne/roles-exp/trials.csv), [selection/Pareto metadata](../../outputs/plots/tsne/roles-exp/summary.json) and each line's `trials/` retain settings, histories and coordinates; only winners have plots/GIFs. The original map is unchanged.

Reproduce batches with `.venv/bin/python src/tsne-lens/research.py --batch outputs/plots/tsne/roles-exp/screening.json`; subsequent batch JSONs are alongside it. `refinement.json` and `final-extension.json` resume earlier trials. Regenerate selection with `outputs/plots/tsne/roles-exp/summarize.py`, then export using `research.py --select LINE TRIAL`.
