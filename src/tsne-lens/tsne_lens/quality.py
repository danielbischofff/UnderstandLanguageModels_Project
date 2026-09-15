"""Small-map diagnostics; KL values are comparable only for the same affinities."""

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from sklearn.manifold import trustworthiness
from sklearn.metrics import silhouette_score


def embedding_quality(distances, embedding, clusters):
    """Measure original-space fidelity and separation of original-space clusters.

    Exact, ordinary (unexaggerated) KL is recomputed independently of openTSNE's
    approximate optimizer report, even when the map uses late exaggeration.
    Silhouettes exclude noise; coverage is reported alongside them.
    """
    xy = np.asarray(embedding)
    low = squareform(pdist(xy))
    dof = getattr(embedding, "gradient_descent_params", {}).get("dof", 1)
    q = (1 + low**2 / dof) ** -dof
    np.fill_diagonal(q, 0)
    q /= q.sum()
    p = embedding.affinities.P.toarray()
    positive = p > 0
    n = len(xy)
    k = min(10, (n - 1) // 2)
    high_neighbors = distances.copy()
    low_neighbors = low.copy()
    np.fill_diagonal(high_neighbors, np.inf)
    np.fill_diagonal(low_neighbors, np.inf)
    high_order = np.argsort(high_neighbors, axis=1)[:, :k]
    low_order = np.argsort(low_neighbors, axis=1)[:, :k]
    mask = clusters >= 0
    count = len(set(clusters) - {-1})
    result = {
        "kl_unexaggerated": float(
            np.sum(p[positive] * np.log(p[positive] / q[positive]))
        ),
        "neighborhood_k": k,
        "trustworthiness": float(
            trustworthiness(distances, xy, n_neighbors=k, metric="precomputed")
        ),
        "neighbor_recall": float(
            np.mean([len(set(a) & set(b)) / k for a, b in zip(high_order, low_order)])
        ),
        "global_distance_spearman": float(
            spearmanr(
                distances[np.triu_indices(n, 1)], low[np.triu_indices(n, 1)]
            ).statistic
        ),
        "clusters": count,
        "cluster_coverage": float(mask.mean()),
    }
    if 1 < count < mask.sum():
        result["cluster_silhouette_original"] = float(
            silhouette_score(
                distances[np.ix_(mask, mask)], clusters[mask], metric="precomputed"
            )
        )
        result["cluster_silhouette_2d"] = float(
            silhouette_score(
                low[np.ix_(mask, mask)], clusters[mask], metric="precomputed"
            )
        )
    # Degenerate, constant-distance inputs have undefined rank correlation.
    return {
        key: value if not isinstance(value, float) or np.isfinite(value) else None
        for key, value in result.items()
    }
