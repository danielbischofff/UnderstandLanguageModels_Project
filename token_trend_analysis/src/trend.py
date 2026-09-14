"""Stratified monotone-trend test of per-word series over strength ladders, without scipy.

Per replicate (a steering scheme or a question) the evidence is Spearman's rho
between a word's coverage and strength order, computed on average ranks so that
ties (a word absent at several strengths) are handled. The test statistic is the
**sum of rho over replicates**. Its null distribution comes from shuffling the
strength labels independently within every replicate: the stratified permutation
test, i.e. Page's trend test with a Monte-Carlo p-value.

Why not a per-replicate exact p combined with Fisher's method (the first version of
this analysis): with three strengths a single replicate can never give p below 1/6,
and Fisher's method on such coarse discrete p-values is very conservative. Summing
rho instead lets weakly consistent replicates add up. The permutations are shared
by all words (each word's p-value is still valid), and the smallest reportable p is
1 / (N_PERMUTATIONS + 1). Benjamini-Hochberg then corrects across words.

Caveat: in the Tylenol sweeps all replicates share the one unsteered control, so
they are not fully independent. The persona suites are not tested this way: each
question is a single 3-4 point ladder, described rather than tested (see run.py).
"""

from __future__ import annotations

import numpy as np

N_PERMUTATIONS = 100_000
CHUNK = 5_000
SEED = 0


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Ranks along the last axis with ties sharing their mean rank (the Spearman convention).

    Rank = (#smaller) + (#equal - 1) / 2; pairwise comparison is cheap for <= 7 strengths.
    """
    less = (values[..., :, None] > values[..., None, :]).sum(axis=-1)
    equal = (values[..., :, None] == values[..., None, :]).sum(axis=-1)
    return less + (equal - 1) / 2


def _unit_ranks(coverage: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Centred, unit-length average ranks [W, R, A] and the matching strength-order vector."""
    n_str = coverage.shape[1]
    ranks = _average_ranks(coverage.transpose(2, 0, 1))                   # [W, R, A]
    centred = ranks - ranks.mean(axis=-1, keepdims=True)
    norm = np.linalg.norm(centred, axis=-1, keepdims=True)
    unit = np.divide(centred, norm, out=np.zeros_like(centred), where=norm > 0)
    order = np.arange(n_str, dtype=float) - (n_str - 1) / 2
    return unit, order / np.linalg.norm(order)


def spearman_rho(coverage: np.ndarray) -> np.ndarray:
    """Spearman rho [R, W] of each series with strength order; 0 for constant series."""
    unit, order = _unit_ranks(coverage)
    return (unit @ order).T


def stratified_trend(coverage: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-replicate rho [R, W] and one-sided p-values (down, up) [W] for the summed rho.

    ``coverage`` is [replicates, strengths, words], strengths weakest to strongest.
    Constant series carry no information: rho 0 in the statistic and under the null.
    """
    n_rep, n_str, n_words = coverage.shape
    unit, order = _unit_ranks(coverage)
    rho = unit @ order                                                    # [W, R]
    observed = rho.sum(axis=1)

    flat = unit.reshape(n_words, n_rep * n_str).astype(np.float32)
    rng = np.random.default_rng(SEED)
    below = np.zeros(n_words)
    above = np.zeros(n_words)
    for start in range(0, N_PERMUTATIONS, CHUNK):
        size = min(CHUNK, N_PERMUTATIONS - start)
        shuffled = rng.permuted(np.broadcast_to(order, (size, n_rep, n_str)), axis=2)
        null = flat @ shuffled.reshape(size, n_rep * n_str).T.astype(np.float32)   # [W, size]
        below += (null <= observed[:, None] + 1e-5).sum(axis=1)
        above += (null >= observed[:, None] - 1e-5).sum(axis=1)
    down = (below + 1) / (N_PERMUTATIONS + 1)
    up = (above + 1) / (N_PERMUTATIONS + 1)
    return rho.T, down, up


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """False-discovery-rate adjusted q-values."""
    order = np.argsort(p_values)
    scaled = p_values[order] * len(p_values) / np.arange(1, len(p_values) + 1)
    q = np.minimum.accumulate(scaled[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.minimum(q, 1.0)
    return out
