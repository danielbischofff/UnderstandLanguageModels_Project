"""C14: which residual-stream signatures go with the model going off the rails?

Off the rails = full role-play: the answer is spoken in character (or, on the
Tylenol runs, harmful), or it collapses into loops. Data already on disk:

- Tylenol 17-role runs (alpha, band): outcome = judge 2 (harmful).
- Question-bank mini runs (alpha, band; 8 personas x 12 benign questions x
  4 settings): outcomes from the role-play rubric (``judge-export --rubric
  roleplay``: 1 = in character, 2 = broken) when judged, plus an automatic
  ``loop`` label from the answer text (share of repeated token trigrams).

Signatures per trial at layer 32, for the prompt (position 0 excluded; read
before any answer exists, so predictive) and the answer (descriptive, partly
caused by the answer itself), each relative to the unsteered trial of the same
question:

- ``near``      share of tokens with max cosine to a role > 0.5     (expect lower)
- ``align_mu``  cos with the mean role direction                     (expect lower)
- ``own_excess`` share in the steered role's Ward cluster, minus baseline (expect higher)
- ``axis``      cos with the Assistant Axis D - mean role, minus baseline
                (Lu et al. 2026; negative = drifted away from the Assistant)   (expect lower)
- ``norm_ratio`` median residual norm over the unsteered one         (expect higher)
- ``switches``  answer only: share of adjacent tokens changing nearest cluster (expect higher)

Test, fixed before looking: AUC in the expected direction (0.5 = chance), bootstrap
95% CI, and a permutation p with labels shuffled within strata that hold the role
and the steering setting fixed (mini banks: role x scheme x coefficient, so only
the 12 questions vary; Tylenol: role). Holm correction over the signatures of one
outcome and segment. A signature counts as a sign of going off the rails when the
CI excludes 0.5 and the Holm-corrected within-stratum p < 0.05.

    uv run --extra plots python src/control/c14_offrails_signatures.py
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from common import NEAR, RESULTS, ROOT, load_space, residuals, runs, save, unit

MINI = ROOT / "runtime/outputs/experiments"
SIGNS = {"near": -1, "align_mu": -1, "own_excess": 1, "axis": -1, "norm_ratio": 1, "switches": 1}
LOOP_THRESHOLD = 0.5  # answers whose distinct-trigram share is below this are loops


def latest(path: Path) -> Path:
    return sorted(path.iterdir())[-1]


def datasets():
    tylenol = runs()
    return {
        "tylenol-alpha": tylenol["alpha"],
        "tylenol-band": tylenol["band"],
        "mini-alpha": latest(MINI / "question-bank-mini-alpha/runs"),
        "mini-band": latest(MINI / "question-bank-mini-band/runs"),
    }


def load_trials(label: str, run: Path) -> list[dict]:
    manifest = json.loads((run / "manifest.json").read_text())
    judged = run / "judging" / "outcomes.json"
    outcomes = json.loads(judged.read_text())["outcomes"] if judged.is_file() else {}
    out = []
    for trial in sorted(manifest["trials"].values(), key=lambda t: t["index"]):
        if trial["status"] != "complete":
            continue
        steering = trial.get("steering") or {}
        trial_hash = trial["path"].rsplit("_", 1)[-1]
        out.append(dict(run=label, hash=trial_hash, path=run / trial["path"], question=trial["question"],
                        character=trial["character"], layers=tuple(steering.get("layers") or ()),
                        coefficient=steering.get("coefficient"), judge=outcomes.get(trial_hash)))
    return out


def loop_share(tokens: list[str]) -> float:
    """Share of the answer's token trigrams that are distinct (1 = no repetition)."""
    grams = list(zip(tokens, tokens[1:], tokens[2:]))
    return len(set(grams)) / len(grams) if grams else 0.0


def segment_stats(x, space, target, axis, mu):
    cos = unit(x) @ space.unit.T
    nearest = space.clusters[cos.argmax(1)]
    return dict(
        near=(cos.max(1) > NEAR).mean(),
        align_mu=(unit(x) @ mu).mean(),
        shares=np.bincount(nearest, minlength=6) / len(x),
        own=(nearest == space.clusters[space.index(target)]).mean(),
        axis=(unit(x) @ axis).mean(),
        norm=float(np.median(np.linalg.norm(x, axis=1))),
        switches=float((np.diff(nearest) != 0).mean()) if len(x) > 1 else np.nan,
    )


def signatures(space, trials_by_run):
    mu = unit(space.mean)
    axis = unit(space.default - space.mean)  # Assistant Axis, Lu et al.
    rows = []
    for label, trials in trials_by_run.items():
        base = {}
        for t in trials:
            if t["character"] == "unsteered":
                seg = residuals(t)
                base[t["question"]] = {
                    "prompt": segment_stats(seg["prompt"][0][1:], space, "assistant", axis, mu),
                    "answer": segment_stats(seg["generated"][0], space, "assistant", axis, mu)
                    if len(seg["generated"][0]) > 1 else None,
                }
        for t in trials:
            if t["character"] == "unsteered":
                continue
            seg = residuals(t)
            cluster = space.clusters[space.index(t["character"])]
            row = dict(run=label, question=t["question"], character=t["character"],
                       setting=f"{'+'.join(map(str, t['layers']))}@{t['coefficient']:g}",
                       judge=t["judge"], loop=loop_share(seg["generated"][1]))
            for name, x in (("prompt", seg["prompt"][0][1:]), ("answer", seg["generated"][0])):
                b = base[t["question"]][name]
                if len(x) < 2 or b is None:
                    continue
                s = segment_stats(x, space, t["character"], axis, mu)
                row.update({
                    f"{name}_near": s["near"] - b["near"],
                    f"{name}_align_mu": s["align_mu"] - b["align_mu"],
                    f"{name}_own_excess": s["own"] - b["shares"][cluster],
                    f"{name}_axis": s["axis"] - b["axis"],
                    f"{name}_norm_ratio": s["norm"] / b["norm"],
                })
                if name == "answer":
                    row["answer_switches"] = s["switches"]
            rows.append(row)
    return rows


def auc_test(y, x, strata, rng, draws=4000, boots=2000):
    keep = ~np.isnan(x)
    y, x, strata = y[keep], x[keep], [s for s, k in zip(strata, keep) if k]
    if y.sum() < 2 or (~y).sum() < 2:
        return None
    auc = roc_auc_score(y, x)
    groups = defaultdict(list)
    for i, s in enumerate(strata):
        groups[s].append(i)
    informative = [idx for idx in groups.values() if 0 < y[idx].sum() < len(idx)]
    null = np.empty(draws)
    for d in range(draws):
        shuffled = y.copy()
        for idx in informative:
            shuffled[idx] = rng.permutation(y[idx])
        null[d] = roc_auc_score(shuffled, x)
    pos, neg = np.flatnonzero(y), np.flatnonzero(~y)
    boot = [roc_auc_score(np.r_[np.ones(len(pos)), np.zeros(len(neg))],
                          np.r_[x[rng.choice(pos, len(pos))], x[rng.choice(neg, len(neg))]]) for _ in range(boots)]
    return dict(n=int(len(y)), positives=int(y.sum()), informative_strata=len(informative),
                auc=auc, ci=np.percentile(boot, [2.5, 97.5]).tolist(),
                p_within=float((np.sum(null >= auc) + 1) / (draws + 1)))


def holm(results):
    keys = [k for k, v in results.items() if v]
    order = sorted(keys, key=lambda k: results[k]["p_within"])
    running = 0.0
    for rank, k in enumerate(order):
        running = max(running, (len(order) - rank) * results[k]["p_within"])
        results[k]["p_holm"] = min(1.0, running)
        results[k]["sign"] = bool(results[k]["ci"][0] > 0.5 and results[k]["p_holm"] < 0.05)


def main() -> None:
    space = load_space()
    rng = np.random.default_rng(0)
    trials_by_run = {label: load_trials(label, run) for label, run in datasets().items()}
    rows = signatures(space, trials_by_run)

    outcomes = {
        "tylenol": {"harmful": lambda r: r["judge"] == 2},
        "mini": {"in_character": lambda r: r["judge"] == 1, "broken": lambda r: r["judge"] == 2,
                 "loop": lambda r: r["loop"] < LOOP_THRESHOLD},
    }
    result = {"loop_threshold": LOOP_THRESHOLD, "tests": {}, "counts": {}}
    for family, defs in outcomes.items():
        group = [r for r in rows if r["run"].startswith(family)]
        strata = [(r["run"], r["character"]) if family == "tylenol" else (r["run"], r["character"], r["setting"])
                  for r in group]
        for outcome, rule in defs.items():
            if outcome != "loop" and any(r["judge"] is None for r in group):
                result["tests"][f"{family}/{outcome}"] = "not judged yet"
                continue
            y = np.array([bool(rule(r)) for r in group])
            result["counts"][f"{family}/{outcome}"] = int(y.sum())
            for segment in ("prompt", "answer"):
                block = {}
                for sig, sign in SIGNS.items():
                    key = f"{segment}_{sig}"
                    if key not in group[0] and not any(key in r for r in group):
                        continue
                    x = sign * np.array([r.get(key, np.nan) for r in group], dtype=float)
                    block[sig] = auc_test(y, x, strata, rng)
                holm(block)
                result["tests"][f"{family}/{outcome}/{segment}"] = block
    save("c14_offrails_signatures", result)
    with (RESULTS / "c14_offrails_trials.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)
    print(result["counts"])
    for key, block in result["tests"].items():
        if isinstance(block, str):
            print(key, block)
            continue
        print(key, {s: (round(v["auc"], 2), [round(c, 2) for c in v["ci"]], round(v["p_holm"], 4), v["sign"])
                    for s, v in block.items() if v})


if __name__ == "__main__":
    main()
