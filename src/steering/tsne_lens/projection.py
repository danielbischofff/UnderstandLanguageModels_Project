"""Fit role maps, diagnose convergence and project into a fixed reference."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from openTSNE import affinity
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances, silhouette_score

from .inputs import matrix
from .optimization import train_embedding
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
        perplexity=None,
        anneal_perplexities=None,
        seed=42,
        n_clusters=3,
        subclusters=6,
        early_exaggeration=2.0,
        early_exaggeration_iter=250,
        exaggeration=1.0,
        n_iter=1000,
        relaxation_iter=1000,
        max_iter=10000,
        tolerance=1e-6,
        theta=0.0,
        training_output=None,
        default_vector=None,
    ):
        vectors = matrix(vectors, "roles")
        if perplexity is None:
            perplexity = min(90.0, len(vectors) - 1)
            if anneal_perplexities is None:
                anneal_perplexities = (min(30.0, (len(vectors) - 1) / 3),)
        if anneal_perplexities is None:
            anneal_perplexities = ()
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
        if (
            not np.isfinite([early_exaggeration, exaggeration, theta]).all()
            or early_exaggeration < 1
            or exaggeration < 1
            or not 0 <= theta <= 1
        ):
            raise ValueError("exaggerations must be >= 1 and theta in [0, 1]")
        if not 2 <= n_clusters < len(vectors) or subclusters < n_clusters:
            raise ValueError(
                "cluster counts must be >= 2, with subclusters >= n_clusters"
            )
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if metric == "cosine" and np.any(norms == 0):
            raise ValueError("cosine distance is undefined for zero vectors")
        if default_vector is not None:
            default_vector = matrix(np.asarray(default_vector)[None], "default vector")[
                0
            ]
            if (
                len(default_vector) != vectors.shape[1]
                or np.linalg.norm(default_vector) == 0
            ):
                raise ValueError(
                    "default vector must have the same hidden width and nonzero norm"
                )
        embedding, history, frames, convergence = train_embedding(
            vectors,
            perplexities=np.atleast_1d(perplexity).tolist(),
            anneal_perplexities=anneal_perplexities,
            metric=metric,
            seed=seed,
            early_exaggeration=early_exaggeration,
            early_iter=early_exaggeration_iter,
            n_iter=n_iter,
            exaggeration=exaggeration,
            relaxation_iter=relaxation_iter,
            max_iter=max_iter,
            tolerance=tolerance,
            theta=theta,
        )
        distances = pairwise_distances(vectors, metric=metric)
        distances = np.maximum((distances + distances.T) / 2, 0)
        np.fill_diagonal(distances, 0)
        # Ward gives a deterministic broad hierarchy; cosine geometry uses unit vectors.
        cluster_vectors = vectors / norms if metric == "cosine" else vectors
        cuts = {}
        counts = sorted(
            set(range(2, min(8, len(vectors) - 1) + 1))
            | {n_clusters, min(subclusters, len(vectors) - 1)}
        )
        for count in counts:
            cuts[count] = AgglomerativeClustering(
                n_clusters=count, linkage="ward"
            ).fit_predict(cluster_vectors)
        clusters = cuts[n_clusters]

        def names_for(assignments):
            names = {}
            for cluster in sorted(set(assignments)):
                members = np.flatnonzero(assignments == cluster)
                central = np.argsort(distances[np.ix_(members, members)].mean(axis=1))[
                    :3
                ]
                names[int(cluster)] = ", ".join(labels[members[i]] for i in central)
            return names

        sub = cuts[min(subclusters, len(vectors) - 1)]
        config = dict(
            layer=layer,
            model_id=model_id,
            metric=metric,
            perplexity=max(convergence["final_perplexities"]),
            perplexities=convergence["final_perplexities"],
            initial_perplexities=np.atleast_1d(perplexity).tolist(),
            anneal_perplexities=list(anneal_perplexities),
            seed=seed,
            affinity="multiscale"
            if len(convergence["final_perplexities"]) > 1
            else "single",
            clustering="ward",
            n_clusters=n_clusters,
            subcluster_count=len(set(sub)),
            subcluster_ids=sub.tolist(),
            subcluster_names=names_for(sub),
            cluster_candidates=[
                dict(
                    count=k,
                    silhouette_original=float(
                        silhouette_score(distances, lab, metric="precomputed")
                    ),
                    sizes=np.bincount(lab).tolist(),
                )
                for k, lab in cuts.items()
            ],
            initialization="pca",
            learning_rate="auto",
            early_exaggeration=early_exaggeration,
            early_exaggeration_iter=early_exaggeration_iter,
            exaggeration=exaggeration,
            n_iter=n_iter,
            relaxation_iter=relaxation_iter,
            max_iter=max_iter,
            tolerance=tolerance,
            final_exaggeration=1,
            theta=theta,
            residual_stream="post_block_post_mlp",
            preprocessing="none",
            cluster_space="original",
            schema_version=1,
            convergence=convergence,
            training_history=history,
            quality=embedding_quality(distances, embedding, clusters),
        )
        result = cls(
            vectors, list(labels), embedding, clusters, names_for(clusters), config
        )
        if default_vector is not None:
            point = result.project(default_vector[None], [{"label": "default_vector"}])[
                0
            ]
            result.config["reference_points"] = [point]
        if training_output is not None:
            from .training_plots import export_training

            export_training(
                embedding,
                list(labels),
                history,
                frames,
                training_output,
                clusters=clusters,
                default_vector=default_vector,
            )
        return result

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
                **(
                    {"subcluster": self.config["subcluster_ids"][i]}
                    if "subcluster_ids" in self.config
                    else {}
                ),
                "x": float(xy[0]),
                "y": float(xy[1]),
                "cluster": int(cluster),
                "cluster_label": self.cluster_names[int(cluster)],
                "norm": float(np.linalg.norm(vector)),
            }
            for i, (label, xy, cluster, vector) in enumerate(
                zip(self.labels, self.embedding, self.clusters, self.vectors)
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
