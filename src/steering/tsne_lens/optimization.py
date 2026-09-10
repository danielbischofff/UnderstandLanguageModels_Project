"""Observable openTSNE optimization with explicit annealing and a final KL plateau."""

import logging

import numpy as np
from openTSNE import TSNEEmbedding, affinity, initialization


def exact_kl(embedding):
    xy = np.asarray(embedding)
    squared = np.sum((xy[:, None] - xy[None, :]) ** 2, axis=2)
    q = 1 / (1 + squared)
    np.fill_diagonal(q, 0)
    q /= q.sum()
    p = embedding.affinities.P.toarray()
    mask = p > 0
    return float(np.sum(p[mask] * np.log(p[mask] / q[mask])))


def train_embedding(
    vectors,
    *,
    perplexities=(30,),
    anneal_perplexities=(),
    metric="cosine",
    seed=42,
    early_exaggeration=4,
    early_iter=250,
    n_iter=1000,
    exaggeration=1,
    relaxation_iter=1000,
    max_iter=10000,
    tolerance=1e-6,
    theta=0,
    checkpoint_interval=250,
):
    """Return embedding, scalar history, coordinates and convergence status.

    All N-1 neighbors are used, independently of entropy/perplexity. Annealing
    replaces the affinity after each n_iter-long stage. The final affinity is
    held fixed during ordinary-KL relaxation until two consecutive checkpoint
    changes are below tolerance, after at least relaxation_iter steps.
    """
    requested = [float(p) for p in perplexities]
    anneal_requested = [float(p) for p in anneal_perplexities]
    if (
        not requested
        or not np.isfinite(requested + anneal_requested).all()
        or min(requested + anneal_requested) <= 0
    ):
        raise ValueError("perplexities must be positive and finite")
    if (
        early_iter < 0
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
    stages = [clamp([p]) for p in anneal_requested]
    if any(next_p[0] >= prev_p[0] for prev_p, next_p in zip([scales] + stages, stages)):
        raise ValueError("annealing perplexities must decrease strictly")

    def build(ps):
        # Explicitly include every role, even for small perplexity.
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

    embedding = TSNEEmbedding(
        initialization.pca(vectors, random_state=seed),
        build(scales),
        negative_gradient_method="bh",
        theta=theta,
        n_jobs=1,
        random_state=seed,
    )
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

    def optimize(steps, phase, exag):
        nonlocal iteration
        for start in range(0, steps, checkpoint_interval):
            count = min(checkpoint_interval, steps - start)
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
    optimize(early_iter, "early", early_exaggeration)
    for index, next_scales in enumerate([scales] + stages):
        if index:
            scales = next_scales
            embedding.affinities = build(scales)
            snapshot(f"anneal-{index}-start", 1)
        optimize(n_iter, f"fit-{index}", exaggeration)
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
            requested_anneal_perplexities=anneal_requested,
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
