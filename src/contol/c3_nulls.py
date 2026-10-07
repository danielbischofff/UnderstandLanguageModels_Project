"""C3: chance levels for the ribbon quantities.

1. ``maxcos`` null. All role vectors share one direction (the role mean mu, cos
   ~0.97), so a residual's cosine to *any* role is mostly its alignment with mu.
   For every real token we build a null vector with the same cos(x, mu) and a
   random remainder, and compare max-cosine and the 0.5 "near" threshold. The
   excess over the null is the part of proximity that is specific to some role.
2. ``own`` permutation. Within each (method, layer, coefficient) group of 17
   steered trials, the share of tokens in the steered role's cluster above the
   unsteered baseline is compared with the same statistic after shuffling which
   role each trial was steered towards.

Run c5_target_attraction.py first.

    uv run --extra plots python src/contol/c3_nulls.py
"""

from __future__ import annotations

import ast
import csv

import numpy as np

from common import NEAR, RESULTS, load_space, residuals, runs, save, trials, unit


def matched_null(x_unit, mu_unit, rng):
    """Vectors with the same cosine to mu and a random orthogonal remainder."""
    a = x_unit @ mu_unit
    r = rng.standard_normal(x_unit.shape)
    r -= np.outer(r @ mu_unit, mu_unit)
    r = unit(r)
    return a[:, None] * mu_unit + np.sqrt(np.clip(1 - a**2, 0, None))[:, None] * r


def maxcos_null(space, rng):
    mu = unit(space.mean)
    roles = space.unit
    out = {}
    for segment in ("prompt", "generated"):
        real, null, align, excess_specific = [], [], [], []
        for method, run in runs().items():
            for trial in trials(method, run):
                if method == "band" and trial["character"] == "unsteered":
                    continue  # the shared control appears in both runs
                x = residuals(trial)[segment][0]
                x = unit(x[1:] if segment == "prompt" else x)
                cos = x @ roles.T
                null_cos = matched_null(x, mu, rng) @ roles.T
                real.append(cos.max(1))
                null.append(null_cos.max(1))
                align.append(x @ mu)
        real, null, align = map(np.concatenate, (real, null, align))
        excess = real - null
        out[segment] = dict(
            tokens=len(real),
            near_real=(real > NEAR).mean(),
            near_null=(null > NEAR).mean(),
            corr_maxcos_align_mu=float(np.corrcoef(real, align)[0, 1]),
            excess_median=float(np.median(excess)),
            excess_p05_p95=np.percentile(excess, [5, 95]).tolist(),
            share_excess_above_0_05=float((excess > 0.05).mean()),
        )
    return out


def own_permutation(rng, draws=10000):
    with (RESULTS / "c5_trials.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    space = load_space()
    cluster_of = {r["character"]: int(space.clusters[space.index(r["character"])])
                  for r in rows if r["method"] != "unsteered"}
    base = {(r["run"], s): np.array(ast.literal_eval(r[f"{s}_clusters"])) / float(r[f"{s}_tokens"])
            for r in rows if r["method"] == "unsteered" for s in ("prompt", "generated")}
    out = {}
    for method in ("alpha", "band"):
        for layer in ("29", "32"):
            coefficients = sorted({r["coefficient"] for r in rows if r["method"] == method})
            for coefficient in coefficients:
                group = [r for r in rows if r["method"] == method and r["layer"] == layer
                         and r["coefficient"] == coefficient]
                key = f"{method}-L{layer}-{float(coefficient):g}"
                out[key] = {}
                for segment in ("prompt", "generated"):
                    shares = np.array([np.array(ast.literal_eval(r[f"{segment}_clusters"]))
                                       / float(r[f"{segment}_tokens"]) for r in group])
                    excess = shares - base[(method, segment)]
                    targets = np.array([cluster_of[r["character"]] for r in group])
                    observed = excess[np.arange(len(group)), targets].mean()
                    null = np.array([excess[np.arange(len(group)), rng.permutation(targets)].mean()
                                     for _ in range(draws)])
                    out[key][segment] = dict(observed=observed, null_mean=null.mean(),
                                             p=float((np.sum(null >= observed) + 1) / (draws + 1)))
    return out


def main() -> None:
    space = load_space()
    rng = np.random.default_rng(0)
    result = dict(maxcos=maxcos_null(space, rng), own=own_permutation(rng))
    save("c3_nulls", result)
    print(result["maxcos"])
    print({k: {s: (round(v["observed"], 3), round(v["null_mean"], 3), round(v["p"], 4)) for s, v in b.items()}
           for k, b in result["own"].items()})


if __name__ == "__main__":
    main()
