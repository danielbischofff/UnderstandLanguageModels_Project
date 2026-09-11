# Layer-32 findings

- **Current map:** 275 roles, fixed combined Gaussian affinities **15/90**, cosine distance, all 274 neighbors, PCA initialization, Ward6 and three central labels per cluster. Perplexity switching has been removed.
- **Exaggeration only:** 250 iterations at 2, then a 125-iteration linear ramp to 1 in 25-iteration steps. Total **2,375 iterations**, including 1,000 final plateau checks; another 2,000 iterations changed KL by **0.0**.
- The fixed-objective KL falls **1.57678 → 0.42163** from PCA to final. Do not compare this KL with previous single-scale runs.
- **Early exaggeration end → final:** recall@10 **40.3% → 48.3%**, trustworthiness **0.915 → 0.931**, global distance correlation **0.765 → 0.768**; Ward6 2D silhouette falls **0.443 → 0.359**. Looser visual boundaries accompany better neighborhood fidelity here. Cross-cluster neighbors actually decrease **15.9% → 14.3%** (original space: **20.2%**).
- This is a tradeoff: PCA preserves global distance ranks better (**0.876**) but fewer neighbors (**29.1%** recall@10). A clearer partition or lower KL alone does not prove a more faithful map.
- **Default-vector display fixed:** the old “PCA” frame applied 250 t-SNE insertion steps. It was not a PCA projection and placed default about **6.9×** farther from assistant than the true role-fitted PCA transform does. Assistant is its nearest original-space role (cosine distance **0.000748**). Later default positions are independent insertions; the first animated update spans 25 optimization iterations.
- One adaptive-scale **25-frame GIF**; checkpoint coordinates are retained without individual PNG exports. Prior sweep plots were not recreated.

[HTML](../../outputs/plots/tsne/roles/roles.html) · [PNG](../../outputs/plots/tsne/roles/roles.png) · [GIF](../../outputs/plots/tsne/roles/training/training.gif) · [Diagnostics](../../outputs/plots/tsne/roles/diagnostics.json)

Code: [objective/schedule](../steering/tsne_lens/optimization.py), [reference placement/GIF](../steering/tsne_lens/training_plots.py).
[openTSNE explanation of exaggeration](https://opentsne.readthedocs.io/en/stable/parameters.html#exaggeration).
