"""Observable openTSNE optimization with explicit annealing and a final KL plateau."""

import logging

import numpy as np
from openTSNE import TSNEEmbedding, affinity, initialization
from sklearn.decomposition import PCA


def exact_kl(embedding):
    xy = np.asarray(embedding)
    squared = np.sum((xy[:, None] - xy[None, :]) ** 2, axis=2)
    dof = getattr(embedding, "gradient_descent_params", {}).get("dof", 1)
    q = (1 + squared / dof) ** -dof
    np.fill_diagonal(q, 0)
    q /= q.sum()
    p = embedding.affinities.P.toarray()
    mask = p > 0
    return float(np.sum(p[mask] * np.log(p[mask] / q[mask])))


def train_embedding(
    vectors,
    *,
    perplexities=(15, 25, 90),
    dof=0.7,
    reference_vector=None,
    metric="cosine",
    seed=42,
    early_exaggeration=4,
    early_iter=250,
    exaggeration_anneal_iter=125,
    n_iter=1000,
    exaggeration=1,
    relaxation_iter=1000,
    max_iter=2000,
    tolerance=1e-6,
    theta=0,
    checkpoint_interval=250,
):
    """Return embedding, scalar history, coordinates and convergence status.

    All N-1 neighbors are used, independently of entropy/perplexity. Affinities
    stay fixed; only exaggeration is annealed. Ordinary-KL relaxation continues
    until two checkpoint changes are below tolerance, after relaxation_iter steps.
    """
    requested = [float(p) for p in perplexities]
    if not requested or not np.isfinite(requested).all() or min(requested) <= 0:
        raise ValueError("perplexities must be positive and finite")
    if (
        not np.isfinite(dof)
        or dof <= 0
        or early_iter < 0
        or exaggeration_anneal_iter < 0
        or n_iter < 1
        or relaxation_iter < 0
        or max_iter < relaxation_iter
        or checkpoint_interval < 1
        or not np.isfinite(tolerance)
        or tolerance <= 0
    ):
        raise ValueError(
            "invalid iterations, checkpoint interval, or convergence tolerance"
        )
    limit = len(vectors) - 1

    def clamp(ps):
        if any(p > limit for p in ps):
            logging.getLogger(__name__).warning(
                "Requested perplexity exceeds N-1=%s; using %s", limit, limit
            )
        return sorted(set(min(p, limit) for p in ps))

    scales = clamp(requested)

    def build(ps):
        # Adaptive Gaussian weights, not binary neighbors; include every role.
        single = affinity.PerplexityBasedNN(
            vectors,
            perplexity=ps[0],
            k_neighbors=limit,
            metric=metric,
            method="exact",
            n_jobs=1,
            random_state=seed,
        )
        if len(ps) == 1:
            return single
        return affinity.Multiscale(
            perplexities=ps,
            knn_index=single.knn_index,
            metric=metric,
            n_jobs=1,
            random_state=seed,
        )

    # Retain the actual role-fitted PCA transform for the held-out reference.
    pca = PCA(n_components=2, random_state=seed)
    initial = pca.fit_transform(vectors)
    scale = np.std(initial[:, 0]) / 1e-4
    initial /= scale
    reference_pca = (
        (pca.transform(np.asarray(reference_vector)[None])[0] / scale).tolist()
        if reference_vector is not None
        else None
    )
    initialization.jitter(initial, inplace=True, random_state=seed)
    embedding = TSNEEmbedding(
        np.ascontiguousarray(initial),
        build(scales),
        dof=dof,
        negative_gradient_method="bh",
        theta=theta,
        n_jobs=1,
        random_state=seed,
    )
    if isinstance(embedding.affinities, affinity.Multiscale):
        # openTSNE may clamp multiscale requests for small input sets.
        scales = list(embedding.affinities.effective_perplexities_)
    history, frames = [], []
    iteration = 0

    def snapshot(phase, exag):
        history.append(
            dict(
                iteration=iteration,
                phase=phase,
                perplexities=list(scales),
                exaggeration=exag,
                kl=exact_kl(embedding),
            )
        )
        frames.append(np.asarray(embedding).copy())

    def optimize(steps, phase, exag, interval=checkpoint_interval):
        nonlocal iteration
        for start in range(0, steps, interval):
            count = min(interval, steps - start)
            embedding.optimize(
                count,
                exaggeration=exag,
                learning_rate="auto",
                momentum=0.8,
                inplace=True,
            )
            iteration += count
            snapshot(phase, exag)

    snapshot("pca", 1)
    # Resolve early motion and approximate a linear ramp in 25-iteration steps.
    optimize(early_iter, "early", early_exaggeration, min(25, checkpoint_interval))
    if early_iter and early_exaggeration != exaggeration:
        for start in range(0, exaggeration_anneal_iter, 25):
            count = min(25, exaggeration_anneal_iter - start)
            fraction = (start + count) / exaggeration_anneal_iter
            value = early_exaggeration + fraction * (exaggeration - early_exaggeration)
            optimize(count, "exaggeration-anneal", value)
    optimize(n_iter, "fit", exaggeration)
    # A saturation claim applies only to ordinary KL at the final fixed affinity.
    converged = False
    stable = 0
    elapsed = 0
    snapshot("relax-start", 1)
    previous = history[-1]["kl"]
    while elapsed < max_iter:
        count = min(checkpoint_interval, max_iter - elapsed)
        optimize(count, "relax", 1)
        elapsed += count
        current = history[-1]["kl"]
        stable = stable + 1 if abs(previous - current) < tolerance else 0
        previous = current
        if elapsed >= relaxation_iter and stable >= 2:
            converged = True
            break
    return (
        embedding,
        history,
        np.stack(frames),
        dict(
            converged=converged,
            iterations=iteration,
            final_perplexities=scales,
            requested_perplexities=requested,
            reference_pca=reference_pca,
            final_kl=history[-1]["kl"],
            last_checkpoint_change=abs(history[-2]["kl"] - history[-1]["kl"]),
            tolerance=tolerance,
            plateau_checks=stable,
            neighbors=limit,
            pca_kl_final_affinity=exact_kl(
                TSNEEmbedding(frames[0], embedding.affinities)
            ),
        ),
    )
