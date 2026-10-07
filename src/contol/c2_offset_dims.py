"""C2: is the nearest-role readout decided by the shared offset coordinate?

Dim 731 carries the first-token massive activation and a constant offset on
every token, and it holds 60-72% of each role vector's squared norm. The model
needs it (it acts as a bias, Sun et al. 2024), so this is not a claim that the
coordinate is unimportant. The question is narrower: does the lens readout, the
role cluster nearest to each token, follow that one shared coordinate or the
remaining 5117 dimensions where roles differ? We recompute the readout
(a) without dims 731/2994/1016 (the three massive dims of the first token) and
(b) after subtracting the role mean from roles and residuals, and measure how
often the nearest cluster and the steered-role statistics stay the same.

    uv run --extra plots python src/contol/c2_offset_dims.py
"""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score

from common import load_space, residuals, runs, save, trials, unit

MASSIVE = [731, 2994, 1016]


def readouts(x, roles, mean):
    keep = np.ones(x.shape[1], dtype=bool)
    keep[MASSIVE] = False
    return {
        "full": unit(x) @ unit(roles).T,
        "no_massive_dims": unit(x[:, keep]) @ unit(roles[:, keep]).T,
        "centered": unit(x - mean) @ unit(roles - mean).T,
    }


def main() -> None:
    space = load_space()
    out = {}
    for segment in ("prompt", "generated"):
        nearest = {k: [] for k in ("full", "no_massive_dims", "centered")}
        own = {k: [] for k in nearest}
        rank = {k: [] for k in nearest}
        for method, run in runs().items():
            for trial in trials(method, run):
                if trial["character"] == "unsteered" and method == "band":
                    continue
                x = residuals(trial)[segment][0]
                x = x[1:] if segment == "prompt" else x
                target = "assistant" if trial["character"] == "unsteered" else trial["character"]
                t = space.index(target)
                for name, cos in readouts(x, space.vectors, space.mean).items():
                    cl = space.clusters[cos.argmax(1)]
                    nearest[name].append(cl)
                    own[name].append((cl == space.clusters[t]).mean())
                    rank[name].append(np.median((cos > cos[:, [t]]).sum(1) + 1))
        full = np.concatenate(nearest["full"])
        out[segment] = {
            name: dict(
                same_nearest_cluster=float((np.concatenate(nearest[name]) == full).mean()),
                kappa=float(cohen_kappa_score(full, np.concatenate(nearest[name]))),
                cluster_counts=np.bincount(np.concatenate(nearest[name]), minlength=6).tolist(),
                own_spearman_vs_full=float(spearmanr(own["full"], own[name]).statistic),
                rank_spearman_vs_full=float(spearmanr(rank["full"], rank[name]).statistic),
            )
            for name in ("no_massive_dims", "centered")
        } | {"full_cluster_counts": np.bincount(full, minlength=6).tolist(), "tokens": len(full)}
    save("c2_offset_dims", out)
    for segment, block in out.items():
        print(segment, block)


if __name__ == "__main__":
    main()
