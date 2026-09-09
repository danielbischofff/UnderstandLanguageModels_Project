"""Fit once on role centroids; place residuals without moving those anchors."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from openTSNE import TSNE
from sklearn.cluster import HDBSCAN
from sklearn.metrics import pairwise_distances

from .inputs import matrix


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
        perplexity=30.0,
        seed=42,
        min_cluster_size=5,
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
        if not np.isfinite(perplexity) or perplexity <= 0 or min_cluster_size < 2:
            raise ValueError(
                "perplexity must be positive; min_cluster_size must be >= 2"
            )
        if metric == "cosine" and np.any(np.linalg.norm(vectors, axis=1) == 0):
            raise ValueError("cosine distance is undefined for zero vectors")
        perplexity = min(float(perplexity), (len(vectors) - 1) / 3)
        embedding = TSNE(
            n_components=2,
            perplexity=perplexity,
            metric=metric,
            initialization="pca",
            neighbors="exact",
            n_jobs=1,
            negative_gradient_method="bh",
            random_state=seed,
        ).fit(vectors)
        distances = pairwise_distances(vectors, metric=metric)
        distances = np.maximum((distances + distances.T) / 2, 0)
        np.fill_diagonal(distances, 0)
        clusters = np.full(len(vectors), -1, dtype=int)
        if min_cluster_size <= len(vectors):
            clusters = HDBSCAN(
                min_cluster_size=min_cluster_size,
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
            perplexity=perplexity,
            seed=seed,
            min_cluster_size=min_cluster_size,
            residual_stream="post_block_post_mlp",
            preprocessing="none",
            cluster_space="original",
            schema_version=1,
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
        partial = self.embedding.prepare_partial(
            vectors,
            perplexity=min(perplexity, self.config["perplexity"]),
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
