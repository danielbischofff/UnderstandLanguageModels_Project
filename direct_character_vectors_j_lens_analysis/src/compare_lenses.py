"""J-lens against R-lens on one direction, layer by layer.

Two questions, one experiment.

1. The cluster of rare web-crawl tokens that dominates early and middle layers
   of the Qwen3-32B readouts -- carried error in the Jacobian, or in the
   representation?
2. R-lens (RelP, arXiv:2508.21258) claims better *early-layer* faithfulness.
   Does it actually recover known semantics earlier than J-lens?

R-lens is the same Jacobian estimator read through an LRP-modified backward
graph. Forward values are bit-identical and the fitting recipe is matched (same
target layer, ``skip_first``, corpus, prompt count), so the two differ only in
how gradients flow. Both arms read the *same* direction at the *same* layers:
any difference between the columns is the lens, not the input.

The direction is a recovered refusal direction, so its semantics are known in
advance -- which turns "is this readout any good" into something measurable.
Three lexicons are scored per layer:

``harm``
    the subject matter a refusal is about (dangerous, illegal, 危险, 非法).
``refusal``
    the speech act itself (sorry, cannot, 抱歉, 拒绝).
``flagged``
    the rare web-crawl cluster from the Qwen3-32B work, carried over verbatim
    so the two models are directly comparable.

Plus, per layer, the Jaccard overlap between the two arms' top-k -- where the
lenses agree and where they diverge.

Usage::

    python src/compare_lenses.py [--sign -r] [--top-k 15]
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from readout import LensTransport
from run_analysis import ROOT, decode_vocab, entropy
from unembed import Unembedding, top_k

MODEL_DIR = ROOT / "data" / "qwen3.6-27b"
ARMS = {"j": ("J-lens", MODEL_DIR / "j-lens" / "lens.pt"),
        "r": ("R-lens", MODEL_DIR / "r-lens" / "lens.pt")}

LEXICONS = {
    "harm": r"dangerous|illegal|unsafe|malicious|illicit|unethical|harmful|weapon"
            r"|危险|非法|危害|违法|违规|有害",
    "refusal": r"sorry|apolog|cannot|can't|unable|refus|decline|regret"
               r"|抱歉|对不起|拒绝|不能|无法|不允许|禁止",
    "flagged": r"erot|porn|sex|fetisch|eskort|escort|milf|creampie|cumshot|lesbi"
               r"|geile|nude|prostitu|shemale|strapon|salopes|neuken",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--direction", type=Path,
                        default=MODEL_DIR / "refusal_direction.npy")
    parser.add_argument("--sign", default="-r", choices=("+r", "-r", "both"))
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument("--out", type=Path, default=ROOT / "out" / "lens_compare.json")
    args = parser.parse_args()

    started = time.time()
    direction = np.load(args.direction).astype(np.float32)
    unembedding = Unembedding(MODEL_DIR)
    tokenizer = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
    texts, wordlike = decode_vocab(tokenizer, unembedding.vocab_size)
    masks = {name: np.array([bool(re.search(p, t, re.I)) for t in texts])
             for name, p in LEXICONS.items()}
    print(f"direction d={direction.size} | {unembedding} | "
          + " ".join(f"{k}={int(v.sum())}" for k, v in masks.items()), flush=True)

    signs = (("+r", +1), ("-r", -1)) if args.sign == "both" else \
            ((args.sign, 1 if args.sign == "+r" else -1),)

    rows, residuals, layers = [], [], None
    for arm, (label, path) in ARMS.items():
        lens = LensTransport(path)
        layers = lens.source_layers
        print(f"  {label}: {lens}", flush=True)
        for tag, sign in signs:
            for layer in layers:
                residuals.append(lens.transport(layer, sign * direction))
                rows.append((arm, tag, layer))
        del lens

    print(f"unembedding {len(rows)} readouts", flush=True)
    logits = unembedding.logits(np.concatenate(residuals, axis=0))
    ids_all, logit_all, prob_all = top_k(logits, args.top_k)
    ids_word, logit_word, prob_word = top_k(logits, args.top_k, allowed=wordlike)
    entropies = entropy(logits)

    def pack(i, ids, scores, probs):
        return [{"id": int(t), "text": texts[int(t)], "logit": round(float(s), 4),
                 "prob": float(f"{p:.6g}")}
                for t, s, p in zip(ids[i], scores[i], probs[i])]

    readouts = [
        {"arm": a, "sign": s, "layer": l,
         "entropy": round(float(entropies[i]), 4),
         "counts": {k: int(m[ids_word[i]].sum()) for k, m in masks.items()},
         "top": pack(i, ids_all, logit_all, prob_all),
         "top_wordlike": pack(i, ids_word, logit_word, prob_word)}
        for i, (a, s, l) in enumerate(rows)
    ]
    index = {(r["arm"], r["sign"], r["layer"]): r for r in readouts}

    # Where do the two lenses actually disagree?
    for r in readouts:
        other = index.get(("r" if r["arm"] == "j" else "j", r["sign"], r["layer"]))
        if other is not None:
            a = {t["id"] for t in r["top_wordlike"]}
            b = {t["id"] for t in other["top_wordlike"]}
            r["agreement"] = round(len(a & b) / len(a | b), 4)

    document = {
        "meta": {
            "model": "Qwen/Qwen3.6-27B", "d_model": int(direction.size),
            "n_layers": max(layers) + 2, "lens_layers": layers,
            "lens_source": "camilablank/workspace-lenses (qwen3.6-27b)",
            "direction_source": "recovered from huihui-ai/"
                                "Huihui-Qwen3.6-27B-abliterated by rank-1 weight diff",
            "lexicons": LEXICONS, "top_k": args.top_k,
            "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "arms": {a: {"label": ARMS[a][0]} for a in ARMS},
        "readouts": readouts,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(document), encoding="utf-8")

    bands = [("00-15", range(0, 16)), ("16-31", range(16, 32)),
             ("32-47", range(32, 48)), ("48-62", range(48, 63))]
    for tag, _ in signs:
        print(f"\n=== sign {tag}: lexicon hits per band (of {args.top_k} per layer) ===")
        print(f"  {'band':7s} {'harm J/R':>14s} {'refusal J/R':>14s} "
              f"{'flagged J/R':>14s} {'entropy J/R':>14s} {'agree':>7s}")
        for name, rng in bands:
            cells = []
            for key in ("harm", "refusal", "flagged"):
                vals = [sum(index[(a, tag, l)]["counts"][key] for l in rng
                            if (a, tag, l) in index) for a in ("j", "r")]
                cells.append(f"{vals[0]:>6d}/{vals[1]:<6d}")
            ent = [np.mean([index[(a, tag, l)]["entropy"] for l in rng
                            if (a, tag, l) in index]) for a in ("j", "r")]
            agree = np.mean([index[("j", tag, l)].get("agreement", 0) for l in rng
                             if ("j", tag, l) in index])
            print(f"  {name:7s} {cells[0]:>14s} {cells[1]:>14s} {cells[2]:>14s} "
                  f"{ent[0]:6.2f}/{ent[1]:<6.2f} {agree:7.2f}")

        first = {}
        for arm in ("j", "r"):
            for l in layers:
                if index[(arm, tag, l)]["counts"]["harm"] > 0:
                    first[arm] = l
                    break
        print(f"  first layer with a harm token -- "
              f"J-lens: {first.get('j', 'never')}, R-lens: {first.get('r', 'never')}")

    print(f"\nwrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB) "
          f"in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
