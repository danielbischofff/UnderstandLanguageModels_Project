# Layer-32 findings

- **Selected: PCA initialization, perplexity 90 → 30, early exaggeration 2, then 1.** Three broad regions remain recognizable from PCA; `default_vector` projects beside assistant.
- **3,250 iterations:** 250 early + 1,000 at P90 + 1,000 at P30 + 1,000 plateau checks. Another 2,000 iterations changed KL by **0.0**.
- At fixed P90, KL drops **0.9493 → 0.2528**; at P30 after annealing, **0.8354 → 0.6000**. Evaluating PCA and final coordinates against the same final P30 affinity gives **1.9646 → 0.6000 (69.5%)**.
- Exaggeration 1/2/4 gave similar broad layouts; 2 retained more immediate neighbors than 4 in direct P30 runs. Exaggeration 12 was unnecessary here.
- **15/90 meant simultaneous affinity scales**, not annealing. The simpler sequential schedule works well. All runs include all 274 other roles; perplexity changes their weights.
- Requested P300 exceeds the available neighbors and is explicitly capped to **274**. That near-uniform map collapses: recall@5 **7.9%**, global correlation **0.028**, versus **49.7% / 0.766** for the selected map.
- Ward hierarchy cuts replace HDBSCAN. Original-space silhouettes for **2/3/6 groups: 0.321/0.314/0.273**. Three is a useful broad view, six an exploratory refinement, not proven cluster counts. Average linkage's high silhouette instead isolated two outliers from 273 roles.
- Inspected PNGs across **14 runs**; used neighborhood preservation, global correlation and fixed-partition silhouettes to compare layouts. KL was used only within unchanged affinities to check convergence.

[Map](../../outputs/plots/qwen3-layer32-annealed/roles.html) ·
[Six groups](../../outputs/plots/qwen3-layer32-annealed/subclusters.html) ·
[GIF](../../outputs/plots/qwen3-layer32-annealed/training/training.gif) ·
[KL curve](../../outputs/plots/qwen3-layer32-annealed/training/kl-history.png)

Raw histories, external metrics, 209 checkpoint PNGs and 14 GIFs:
`outputs/plots/qwen3-layer32-retune/`.
