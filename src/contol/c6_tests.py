"""C6: are the steering effects and the harmful-vs-safe differences statistically real?

1. Per steered trial, a paired Wilcoxon over the 46 prompt tokens (position 0
   dropped): does the target role get closer than in the unsteered run? Tested on
   the absolute cosine to the target and on its margin over the median role
   (relative proximity, what the map shows), with a bootstrap CI of the median
   per-token change. p-values are Holm-corrected within each sweep.
2. Harmful (judge 2) vs safe (judge 0) trials: Mann-Whitney on the C5 per-trial
   metrics, for prompt and generated tokens. Only six harmful trials exist, so
   these are exact small-sample tests and read as descriptive support.

Run c5_target_attraction.py first.

    uv run --extra plots python src/contol/c6_tests.py
"""

from __future__ import annotations

import csv

import numpy as np
from scipy.stats import mannwhitneyu, wilcoxon

from common import RESULTS, load_space, residuals, runs, save, trials, unit


def holm(pvalues):
    order = np.argsort(pvalues)
    adjusted = np.empty(len(pvalues))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(pvalues) - rank) * pvalues[i])
        adjusted[i] = min(1.0, running)
    return adjusted


def bootstrap_median(delta, rng, draws=2000):
    samples = rng.choice(delta, size=(draws, len(delta)), replace=True)
    return np.percentile(np.median(samples, axis=1), [2.5, 97.5]).tolist()


def paired_tests(space, rng):
    out = {}
    for method, run in runs().items():
        records = trials(method, run)
        base = next(r for r in records if r["character"] == "unsteered")
        x0 = unit(residuals(base)["prompt"][0][1:]) @ space.unit.T
        rows = []
        for trial in records:
            if trial["character"] == "unsteered":
                continue
            t = space.index(trial["character"])
            x = unit(residuals(trial)["prompt"][0][1:]) @ space.unit.T
            row = dict(character=trial["character"], layer=trial["layer"],
                       coefficient=trial["coefficient"], judge=trial["judge"])
            for name, steered, control in (
                ("cos", x[:, t], x0[:, t]),
                ("margin", x[:, t] - np.median(x, 1), x0[:, t] - np.median(x0, 1)),
            ):
                delta = steered - control
                row[f"{name}_median_delta"] = float(np.median(delta))
                row[f"{name}_ci"] = bootstrap_median(delta, rng)
                row[f"{name}_p"] = float(wilcoxon(steered, control).pvalue)
            rows.append(row)
        for name in ("cos", "margin"):
            adjusted = holm(np.array([r[f"{name}_p"] for r in rows]))
            for r, p in zip(rows, adjusted):
                r[f"{name}_p_holm"] = float(p)
        summary = {}
        for coefficient in sorted({r["coefficient"] for r in rows}):
            group = [r for r in rows if r["coefficient"] == coefficient]
            summary[f"{coefficient:g}"] = {
                f"{name}_{direction}": sum(
                    r[f"{name}_p_holm"] < 0.05 and (r[f"{name}_median_delta"] > 0) == (direction == "up")
                    for r in group
                )
                for name in ("cos", "margin") for direction in ("up", "down")
            } | {"trials": len(group)}
        out[method] = dict(summary=summary, trials=rows)
    return out


def harm_tests():
    with (RESULTS / "c5_trials.csv").open() as handle:
        rows = [r for r in csv.DictReader(handle) if r["method"] != "unsteered"]
    out = {}
    for label, harmful in (("judge2_vs_0", {2}), ("judge1or2_vs_0", {1, 2})):
        bad = [r for r in rows if int(r["judge"]) in harmful]
        safe = [r for r in rows if int(r["judge"]) == 0]
        tests = {}
        for segment in ("prompt", "generated"):
            for metric in ("near", "maxcos", "own", "own_excess", "rank_tgt"):
                key = f"{segment}_{metric}"
                a = np.array([float(r[key]) for r in bad])
                b = np.array([float(r[key]) for r in safe])
                result = mannwhitneyu(a, b, method="exact" if len(a) < 10 else "auto")
                tests[key] = dict(harmful_median=float(np.median(a)), safe_median=float(np.median(b)),
                                  u=float(result.statistic), p=float(result.pvalue))
        out[label] = dict(n_harmful=len(bad), n_safe=len(safe), tests=tests)
    return out


def main() -> None:
    space = load_space()
    rng = np.random.default_rng(0)
    result = dict(paired=paired_tests(space, rng), harm=harm_tests())
    save("c6_tests", result)
    for method, block in result["paired"].items():
        print(method, block["summary"])
    for label, block in result["harm"].items():
        print(label, block["n_harmful"], {k: (round(v["harmful_median"], 3), round(v["safe_median"], 3), round(v["p"], 4))
                                         for k, v in block["tests"].items()})


if __name__ == "__main__":
    main()
