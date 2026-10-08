"""C11: is Ward clustering of the role vectors meaningful in the 5120-d space?

The role map clusters the unit-normalised layer-32 role vectors with Ward in the
original space and only colours the 2D t-SNE map with the result. This control
checks that choice and how well the map shows those clusters:

- intrinsic dimension: centered PCA (as in Lu et al. 2026, Sec. 2.1.3) and TwoNN;
- distance concentration: relative contrast against random unit vectors;
- cluster reality: Ward-6 silhouette against a matched-covariance Gaussian null;
- agreement (ARI) of the 5120-d Ward-6 cut with clusterings of the 2D map and of
  the leading principal components.

    uv run --extra plots python src/control/c11_clustering.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics import pairwise_distances, silhouette_score
from sklearn.mixture import GaussianMixture

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "tsne-lens"))

from tsne_lens.projection import RoleMap  # noqa: E402


def ward(x: np.ndarray, k: int = 6) -> np.ndarray:
    return AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(x)


def unit(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def pca_profile(x: np.ndarray) -> dict:
    """Variance profile of mean-centered vectors, the convention of Lu et al."""
    pca = PCA().fit(x - x.mean(0))
    ratio = pca.explained_variance_ratio_
    cumulative = np.cumsum(ratio)
    return dict(
        pc1_share=float(ratio[0]),
        cumulative={k: float(cumulative[k - 1]) for k in (1, 2, 5, 8, 10, 20, 50)},
        components_for={
            f"{share:.0%}": int(np.searchsorted(cumulative, share) + 1)
            for share in (0.5, 0.7, 0.9)
        },
        participation_ratio=float(1 / np.sum(ratio**2)),
        pc1=pca.components_[0],
    )


def two_nn(distances: np.ndarray) -> float:
    """TwoNN intrinsic dimension (Facco et al. 2017), maximum-likelihood form."""
    d = distances.copy()
    np.fill_diagonal(d, np.inf)
    nearest = np.sort(d, axis=1)[:, :2]
    mu = nearest[:, 1] / nearest[:, 0]
    mu = mu[np.isfinite(mu) & (mu > 1)]
    return float(len(mu) / np.sum(np.log(mu)))


def concentration(distances: np.ndarray) -> dict:
    """Relative contrast (farthest - nearest) / nearest per point, and distance CV.

    Both approach 0 when all pairwise distances look alike, i.e. when the curse
    of dimensionality makes nearest neighbours meaningless.
    """
    d = distances.copy()
    np.fill_diagonal(d, np.nan)
    contrast = (np.nanmax(d, 1) - np.nanmin(d, 1)) / np.nanmin(d, 1)
    return dict(
        relative_contrast_median=float(np.median(contrast)),
        distance_cv=float(np.nanstd(d) / np.nanmean(d)),
    )


def cluster_2d(xy: np.ndarray, method: str, k: int, seed: int) -> np.ndarray:
    if method in {"ward", "average", "complete"}:
        return AgglomerativeClustering(n_clusters=k, linkage=method).fit_predict(xy)
    if method == "kmeans":
        return KMeans(n_clusters=k, n_init=20, random_state=seed).fit_predict(xy)
    return GaussianMixture(k, n_init=10, random_state=seed).fit(xy).predict(xy)


def run(role_map: RoleMap, default_vector: np.ndarray, seed: int, draws: int) -> dict:
    rng = np.random.default_rng(seed)
    raw = role_map.vectors.astype(np.float64)
    vectors = unit(raw)
    xy = np.asarray(role_map.embedding)
    labels = np.asarray(role_map.clusters)
    k = len(set(labels))
    n, width = vectors.shape

    # The map's clusters must be exactly Ward on unit vectors; Ward only sees
    # Euclidean variance, so subtracting the shared mean cannot change it.
    recomputed = ward(vectors, k)
    centered = vectors - vectors.mean(0)

    raw_profile = pca_profile(raw)
    unit_profile = pca_profile(vectors)
    pc1 = raw_profile.pop("pc1")
    unit_profile.pop("pc1")
    # Lu et al. define the Assistant Axis as default minus the mean role vector.
    axis = unit(default_vector - raw.mean(0))
    pc1_scores = (raw - raw.mean(0)) @ pc1
    sign = np.sign(pc1_scores @ (raw @ axis - (raw @ axis).mean()))

    cosine = pairwise_distances(vectors, metric="cosine")
    cosine = np.maximum((cosine + cosine.T) / 2, 0)
    np.fill_diagonal(cosine, 0)
    centered_cosine = pairwise_distances(centered, metric="cosine")
    random = unit(rng.standard_normal((n, width)))

    null_pca = PCA().fit(centered)
    null = []
    for _ in range(draws):
        z = rng.standard_normal((n, len(null_pca.explained_variance_)))
        sample = (z * np.sqrt(null_pca.explained_variance_)) @ null_pca.components_
        null.append(silhouette_score(sample, ward(sample, k)))

    grid = {
        f"{method}-{count}": float(
            adjusted_rand_score(labels, cluster_2d(xy, method, count, seed))
        )
        for method in ("ward", "average", "complete", "kmeans", "gmm")
        for count in range(max(2, k - 2), k + 3)
    }
    best = max(grid, key=grid.get)
    ward_2d = ward(xy, k)

    return dict(
        roles=n,
        hidden=width,
        layer=role_map.config["layer"],
        clusters=k,
        map_clusters_are_ward_on_unit_vectors=bool(
            adjusted_rand_score(labels, recomputed) == 1.0
        ),
        ward_unchanged_by_centering=float(adjusted_rand_score(recomputed, ward(centered, k))),
        intrinsic_dimension=dict(
            pca_raw_centered=raw_profile,
            pca_unit_centered=unit_profile,
            two_nn=two_nn(cosine),
            lu2026_qwen3_32b_components_for_70=8,
        ),
        assistant_axis=dict(
            cos_pc1_with_default_minus_mean=float(sign * pc1 @ axis),
            pc1_mean_by_cluster={
                str(c): float(sign * pc1_scores[labels == c].mean()) for c in sorted(set(labels))
            },
        ),
        concentration=dict(
            roles=concentration(cosine),
            roles_centered=concentration(centered_cosine),
            random_unit_vectors=concentration(pairwise_distances(random, metric="cosine")),
        ),
        silhouette=dict(
            cosine_5120d=float(silhouette_score(cosine, labels, metric="precomputed")),
            euclidean_unit_5120d=float(silhouette_score(vectors, labels)),
            null_euclidean_mean=float(np.mean(null)),
            null_euclidean_sd=float(np.std(null)),
            null_draws=draws,
            map_2d=float(silhouette_score(xy, labels)),
        ),
        agreement_with_5120d_ward=dict(
            ward_2d_same_k=float(adjusted_rand_score(labels, ward_2d)),
            ward_2d_same_k_nmi=float(normalized_mutual_info_score(labels, ward_2d)),
            ward_2d_same_k_silhouette_5120d=float(
                silhouette_score(cosine, ward_2d, metric="precomputed")
            ),
            map_2d_grid=grid,
            map_2d_best=dict(setting=best, ari=grid[best]),
            ward_top_pcs={
                str(pcs): float(
                    adjusted_rand_score(labels, ward(PCA(pcs).fit_transform(centered), k))
                )
                for pcs in (5, 10, 20, 50)
            },
        ),
    )


def rounded(value):
    if isinstance(value, float):
        return round(value, 4)
    if isinstance(value, dict):
        return {key: rounded(item) for key, item in value.items()}
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--map", type=Path, default=ROOT / "outputs/plots/tsne/roles/role-map.pkl")
    parser.add_argument(
        "--default-vector",
        type=Path,
        default=ROOT / "models/assistant-axis-vectors/qwen-3-32b/default_vector.pt",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--null-draws", type=int, default=20)
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).parent / "results" / "c11_clustering.json"
    )
    args = parser.parse_args()

    role_map = RoleMap.load(args.map)
    default = torch.load(args.default_vector, map_location="cpu", weights_only=True)
    default = default[role_map.config["layer"]] if default.ndim == 2 else default
    result = rounded(run(role_map, default.float().numpy().astype(np.float64), args.seed, args.null_draws))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
