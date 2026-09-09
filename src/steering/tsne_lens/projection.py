"""Fit once on role centroids; place residuals without moving those anchors."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from openTSNE import TSNE, affinity, initialization
from sklearn.cluster import HDBSCAN
from sklearn.metrics import pairwise_distances

from .inputs import matrix
from .quality import embedding_quality


@dataclass
class RoleMap:
    vectors: np.ndarray
    labels: list[str]
    embedding: object
    clusters: np.ndarray
    cluster_names: dict[int, str]
    config: dict

    @classmethod
    def fit(
        cls,
        vectors,
        labels,
        *,
        layer=32,
        model_id="Qwen/Qwen3-32B",
        metric="cosine",
        perplexity=(15.0, 90.0),
        seed=42,
        min_cluster_size=3,
        min_samples=2,
        cluster_selection_method="leaf",
        early_exaggeration=12.0,
        early_exaggeration_iter=500,
        exaggeration=1.5,
        n_iter=750,
        relaxation_iter=1500,
        theta=0.0,
    ):
        vectors = matrix(vectors, "roles")
        if len(vectors) < 4 or vectors.shape[1] < 2:
            raise ValueError(
                "t-SNE needs at least four roles and two hidden dimensions"
            )
        if (
            len(labels) != len(vectors)
            or len(set(labels)) != len(labels)
            or any(not isinstance(x, str) or not x.strip() for x in labels)
        ):
            raise ValueError("provide one unique nonempty label per role")
        if layer < 0 or metric not in {"cosine", "euclidean"}:
            raise ValueError("invalid layer or metric")
        perplexities = np.atleast_1d(perplexity).astype(float)
        if (
            perplexities.ndim != 1
            or not len(perplexities)
            or not np.isfinite(perplexities).all()
            or np.any(perplexities <= 0)
            or min_cluster_size < 2
        ):
            raise ValueError(
                "perplexity must be positive; min_cluster_size must be >= 2"
            )
        if min_samples is not None and not 1 <= min_samples <= len(vectors):
            raise ValueError("min_samples must be between 1 and the number of roles")
        if cluster_selection_method not in {"eom", "leaf"}:
            raise ValueError("cluster_selection_method must be eom or leaf")
        if (
            not np.isfinite([early_exaggeration, exaggeration, theta]).all()
            or early_exaggeration < 1
            or exaggeration < 1
            or not 0 <= theta <= 1
            or early_exaggeration_iter < 0
            or n_iter < 1
            or relaxation_iter < 0
        ):
            raise ValueError(
                "exaggerations must be >= 1, theta in [0, 1], and iterations valid"
            )
        if metric == "cosine" and np.any(np.linalg.norm(vectors, axis=1) == 0):
            raise ValueError("cosine distance is undefined for zero vectors")
        perplexities = sorted(
            set(np.minimum(perplexities, (len(vectors) - 1) / 3).tolist())
        )
        estimator = TSNE(
            n_components=2,
            perplexity=perplexities[0],
            metric=metric,
            initialization="pca",
            neighbors="exact",
            n_jobs=1,
            negative_gradient_method="bh",
            random_state=seed,
            learning_rate="auto",
            early_exaggeration=early_exaggeration,
            early_exaggeration_iter=early_exaggeration_iter,
            exaggeration=exaggeration,
            n_iter=n_iter,
            theta=theta,
        )
        if len(perplexities) == 1:
            embedding = estimator.fit(vectors)
        else:
            affinities = affinity.Multiscale(
                vectors,
                perplexities=perplexities,
                metric=metric,
                method="exact",
                n_jobs=1,
                random_state=seed,
            )
            embedding = estimator.fit(
                affinities=affinities,
                initialization=initialization.pca(vectors, random_state=seed),
            )
        if relaxation_iter:
            # Compact groups first, then optimize ordinary KL to release
            # crowding introduced by exaggeration without restarting the map.
            embedding = embedding.optimize(
                n_iter=relaxation_iter,
                exaggeration=1,
                learning_rate="auto",
                momentum=0.8,
            )
        distances = pairwise_distances(vectors, metric=metric)
        distances = np.maximum((distances + distances.T) / 2, 0)
        np.fill_diagonal(distances, 0)
        clusters = np.full(len(vectors), -1, dtype=int)
        if min_cluster_size <= len(vectors):
            clusters = HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                cluster_selection_method=cluster_selection_method,
                metric="precomputed",
                copy=True,
                allow_single_cluster=False,
            ).fit_predict(distances)
        names = {-1: "Unclustered"}
        for cluster in sorted(set(clusters) - {-1}):
            members = np.flatnonzero(clusters == cluster)
            central = np.argsort(distances[np.ix_(members, members)].mean(axis=1))[:3]
            names[int(cluster)] = ", ".join(labels[members[i]] for i in central)
        config = dict(
            layer=layer,
            model_id=model_id,
            metric=metric,
            perplexity=max(perplexities),
            perplexities=perplexities,
            affinity="multiscale" if len(perplexities) > 1 else "single",
            seed=seed,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_method=cluster_selection_method,
            initialization="pca",
            learning_rate="auto",
            early_exaggeration=early_exaggeration,
            early_exaggeration_iter=early_exaggeration_iter,
            exaggeration=exaggeration,
            n_iter=n_iter,
            relaxation_iter=relaxation_iter,
            final_exaggeration=1 if relaxation_iter else exaggeration,
            theta=theta,
            residual_stream="post_block_post_mlp",
            preprocessing="none",
            cluster_space="original",
            schema_version=1,
            quality=embedding_quality(distances, embedding, clusters),
        )
        return cls(vectors, list(labels), embedding, clusters, names, config)

    def project(self, vectors, rows, *, top_k=5, perplexity=5.0):
        vectors = matrix(vectors, "trajectory")
        if vectors.shape[1] != self.vectors.shape[1] or len(rows) != len(vectors):
            raise ValueError("trajectory width/metadata does not match role map")
        if top_k < 1 or not np.isfinite(perplexity) or perplexity <= 0:
            raise ValueError("top_k and transform perplexity must be positive")
        if self.config["metric"] == "cosine" and np.any(
            np.linalg.norm(vectors, axis=1) == 0
        ):
            raise ValueError("cosine distance is undefined for zero vectors")
        # The same partial optimization as transform(), without its in-place
        # recentering of the reference: saved anchors and earlier paths stay fixed.
        transform_perplexity = min(perplexity, self.config["perplexity"])
        affinity_params = (
            {"perplexities": [transform_perplexity]}
            if isinstance(self.embedding.affinities, affinity.Multiscale)
            else {"perplexity": transform_perplexity}
        )
        partial = self.embedding.prepare_partial(
            vectors,
            **affinity_params,
            initialization="median",
            k=min(25, len(self.vectors)),
        )
        xy = partial.optimize(
            n_iter=250,
            learning_rate=0.1,
            exaggeration=1.5,
            momentum=0.8,
            max_grad_norm=0.25,
            inplace=True,
        )
        distances = pairwise_distances(
            vectors, self.vectors, metric=self.config["metric"]
        )
        nearest = np.argsort(distances, axis=1)[:, : min(top_k, len(self.labels))]
        return [
            {
                **row,
                "x": float(coord[0]),
                "y": float(coord[1]),
                "norm": float(np.linalg.norm(vector)),
                "neighbors": [
                    {"label": self.labels[j], "distance": float(distance[j])}
                    for j in indices
                ],
            }
            for row, coord, vector, distance, indices in zip(
                rows, xy, vectors, distances, nearest
            )
        ]

    def role_records(self):
        return [
            {
                "label": label,
                "x": float(xy[0]),
                "y": float(xy[1]),
                "cluster": int(cluster),
                "cluster_label": self.cluster_names[int(cluster)],
                "norm": float(np.linalg.norm(vector)),
            }
            for label, xy, cluster, vector in zip(
                self.labels, self.embedding, self.clusters, self.vectors
            )
        ]

    def save(self, path: Path):
        """Persist the full openTSNE reference for later transform calls."""
        with path.open("wb") as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: Path):
        """Load a trusted local pickle, using the same dependency environment."""
        with path.open("rb") as handle:
            result = pickle.load(handle)
        if not isinstance(result, cls) or result.config.get("schema_version") != 1:
            raise ValueError("unsupported role map")
        return result
