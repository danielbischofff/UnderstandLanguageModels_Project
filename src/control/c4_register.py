"""C4: is the "infant area" pull on user tokens about token type or about the speaker?

Unsteered trials only. For each prompt, the user-written tokens (chat template
excluded) are read out against the role map: share whose nearest role is in
cluster 0 (childlike/impulsive) or cluster 3 (assistant), mean alignment with the
role mean, and the share of dim 731 in the norm. The answer tokens give the
speaker contrast (same model, now the Assistant speaking).

Form predicts the cluster-0 share follows token types and does not move when only
the register changes; speaker predicts casual first-person text sits in cluster 0
and formal or third-person rewrites of the same content move toward cluster 3.

Local part: the casual Tylenol prompt against the 12 neutral question-bank
prompts. GPU part: configs/experiments/tsne-lens/register-control.yaml (6 topics
x casual/formal/third-person); pass its run with --run to add paired tests.

    uv run --extra plots python src/control/c4_register.py [--run RUN_DIR ...]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from common import EXPERIMENTS, LAYER, MODEL, ROOT, content_mask, load_space, runs, save, unit
from common import load_trajectory

QUESTION_BANK = ROOT / "runtime/outputs/experiments/question-bank-mini-band/runs"


def unsteered_trials(run: Path):
    manifest = json.loads((run / "manifest.json").read_text())
    for trial in sorted(manifest["trials"].values(), key=lambda t: t["index"]):
        if trial["character"] == "unsteered" and trial["status"] == "complete":
            yield trial["question"], run / trial["path"]


def readout(path: Path, space) -> dict:
    roles, mu = space.unit, unit(space.mean)
    out = {}
    for segment in ("prompt", "generated"):
        x, rows = load_trajectory(path, layer=LAYER, model_id=MODEL,
                                  prompt_only=segment == "prompt",
                                  generated_only=segment == "generated")
        x = x.astype(np.float64)
        if segment == "prompt":
            x = x[content_mask([r["token"] for r in rows])]
        nearest = space.clusters[(unit(x) @ roles.T).argmax(1)]
        out[segment] = dict(
            tokens=len(x),
            cluster0=(nearest == 0).mean(),
            cluster3=(nearest == 3).mean(),
            align_mu=(unit(x) @ mu).mean(),
            dim731_share=(x[:, 731] ** 2 / (x**2).sum(1)).mean(),
        )
    return out


def paired(table: dict) -> dict:
    topics = sorted({q.rsplit("-", 1)[0] for q in table})
    tests = {}
    for other in ("formal", "third"):
        a = [table[f"{t}-casual"]["prompt"]["cluster0"] for t in topics]
        b = [table[f"{t}-{other}"]["prompt"]["cluster0"] for t in topics]
        tests[f"casual_vs_{other}"] = dict(
            casual_mean=float(np.mean(a)), other_mean=float(np.mean(b)),
            p_exact=float(wilcoxon(a, b, method="exact").pvalue) if a != b else 1.0,
            topics_casual_higher=int(sum(x > y for x, y in zip(a, b))),
        )
    return tests


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run", type=Path, action="append", default=[],
                        help="register-control run directory (GPU part)")
    args = parser.parse_args()
    space = load_space()

    tylenol = dict(unsteered_trials(runs()["alpha"]))
    local = {"tylenol (casual)": readout(next(iter(tylenol.values())), space)}
    bank_run = sorted(QUESTION_BANK.iterdir())[-1]
    for question, path in unsteered_trials(bank_run):
        local[f"bank-{question}"] = readout(path, space)
    bank = [v["prompt"]["cluster0"] for k, v in local.items() if k.startswith("bank-")]
    result = dict(local=dict(
        prompts=local,
        tylenol_cluster0=local["tylenol (casual)"]["prompt"]["cluster0"],
        bank_cluster0_mean=float(np.mean(bank)),
        bank_cluster0_range=[float(min(bank)), float(max(bank))],
    ))

    for run in args.run:
        table = {question: readout(path, space) for question, path in unsteered_trials(run)}
        result[run.name] = dict(prompts=table, paired=paired(table))

    save("c4_register", result)
    print({k: round(v["prompt"]["cluster0"], 2) for k, v in local.items()})
    for key, block in result.items():
        if "paired" in block:
            print(key, block["paired"])


if __name__ == "__main__":
    main()
