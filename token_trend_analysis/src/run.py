"""Find J-Lens words that fade out or are introduced as steering strength grows.

    uv run --no-project --with numpy python token_trend_analysis/src/run.py [experiment ...]

An *analysis unit* is what one set of lists describes: for the Tylenol sweeps, one
steered character over all its schemes (trends are tested); for the persona suites,
one steered character on one question (a single 3-4 point ladder, so trends are
described: a word counts if its series moves in one direction by at least 2x).

For each experiment in ``experiments.py`` (all by default) this writes to
``out/<experiment>/``:

- ``<character>_<metric>.csv`` (Tylenol) or ``<character>/<question>_<metric>.csv``
  (suites) with every tested word;
- ``summary.md`` with the listed words;
- ``web.json``, the data behind the explorer page (see ``build_web.py``).
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

from experiments import EXPERIMENTS, ROOT, Experiment, Ladder, ladders
from load import load_trials
from metrics import build_groups, occupancy, word_key
from trend import benjamini_hochberg, spearman_rho, stratified_trend

# Top-10 J-Lens readout of each character - default vector at every layer (see vector_topk.py there).
VECTOR_READOUT = ROOT / "direct_character_vectors_j_lens_analysis/out/vs_default_top10.json"
OUT = Path(__file__).resolve().parents[1] / "out"

MIN_CELLS = 20           # a word must fill this many cells in at least one trial to be tested
PSEUDO = 5e-4            # coverage pseudocount (0.05% of the grid) for the fold change
Q_MAX, MIN_LOG2FC = 0.05, 1.0
TABLE_ROWS = {"scheme": 40, "question": 15}   # rows per summary table; the CSVs hold every tested word
MAX_REPLICATE_COLUMNS = 3    # more schemes than this: tables show the mean over schemes
SANITY_WORDS = ("警示", "警告", "warning", "warnings")
METRICS = ("cells", "weighted")


def vector_readout_words(words: list[str], character: str) -> dict[int, list[int]]:
    """Baseline words: word index -> layers at which the word is in the top-10 J-Lens readout
    of the character - default vector itself. Any layer counts, steered or not."""
    readout = json.loads(VECTOR_READOUT.read_text(encoding="utf-8"))
    position = {w: i for i, w in enumerate(words)}
    layers: dict[int, list[int]] = {}
    for layer, ids in zip(readout["layers"], readout["ids"][character]):
        for token_id in ids:
            w = position.get(word_key(readout["vocab"][str(token_id)]))
            if w is not None and layer not in layers.setdefault(w, []):
                layers[w].append(layer)
    return layers


def layer_ranges(layers: list[int]) -> str:
    """[4, 5, 6, 9] -> "L4–6, 9"; empty for a word outside the vector readout."""
    spans, start = [], None
    for i, layer in enumerate(layers):
        start = layer if start is None else start
        if i + 1 == len(layers) or layers[i + 1] != layer + 1:
            spans.append(f"{start}" if start == layer else f"{start}–{layer}")
            start = None
    return "L" + ", ".join(spans) if spans else ""


def analyse(experiment: Experiment, lads: list[Ladder], character: str, lookup, words, variants, glossary) -> dict:
    """Trend statistics for every tested word of one analysis unit.

    ``lads`` is every scheme's ladder (Tylenol) or one question's single ladder (suites).
    """
    by_question = experiment.replicate == "question"
    n_words = len(words)
    labels = lads[0].labels
    occ = [[occupancy(t, lad.first_layer, lookup, n_words) for t in lad.trials] for lad in lads]
    vector_layers = vector_readout_words(words, character)

    cells = np.array([[o["cells"] for o in row] for row in occ])                       # [L, A, W]
    tested = np.flatnonzero(cells.max(axis=(0, 1)) >= MIN_CELLS)
    spread = np.array([[o["spread"] for o in row] for row in occ])[:, :, tested]
    # Median over schemes, so one odd scheme cannot make a word "broad" (a no-op for one ladder).
    unsteered_spread = np.median(spread[:, 0], axis=0)
    steered_spread = np.median(spread[:, 1:].max(axis=1), axis=0)
    # Strengths averaged at each end of the fold change. Short ladders use the endpoints: the
    # band cap acts almost like a switch, so averaging the control with the loosest band hides the jump.
    k = 2 if len(labels) > 5 else 1

    base = np.array([int(w) in vector_layers for w in tested], dtype=bool)
    # Every vector-readout word, tested or not, so the control's share depends only on the character and grid.
    share = cells[:, :, sorted(vector_layers)].sum(axis=2) / cells.sum(axis=2)        # [L, A]

    result = {"character": character, "mode": experiment.replicate, "labels": labels, "tested": int(len(tested)),
              "replicates": [{"name": lad.replicate, "first_layer": lad.first_layer,
                              "grid": occ[i][0]["grid_size"], "positions": occ[i][0]["n_positions"],
                              "layers": occ[i][0]["n_layers"]} for i, lad in enumerate(lads)],
              "baseline_share": share.round(4).tolist(), "metrics": {}}
    if by_question:
        result.update(question=lads[0].replicate, prompt=lads[0].trials[0].prompt)

    for metric in METRICS:
        pooled_by_ladder = np.array([[o[metric] / o["grid_size"] for o in row] for row in occ])[:, :, tested]
        pooled = pooled_by_ladder.mean(axis=0)                                          # [A, T]
        log2fc = np.log2((pooled[-k:].mean(axis=0) + PSEUDO) / (pooled[:k].mean(axis=0) + PSEUDO))
        # Share of the total control -> strongest change already present at the weakest strength:
        # ~1 = switch-like onset, ~0 = grows gradually with strength.
        total = pooled[-1] - pooled[0]
        onset = np.where(np.abs(total) > 1e-12, (pooled[1] - pooled[0]) / np.where(total == 0, 1, total), np.nan)
        if by_question:   # one ladder: nothing to test over, so a monotone series with a 2x change
            rho = spearman_rho(pooled_by_ladder)
            steps = np.diff(pooled, axis=0)
            fading = np.all(steps <= 0, axis=0) & (log2fc <= -MIN_LOG2FC)
            appearing = np.all(steps >= 0, axis=0) & (log2fc >= MIN_LOG2FC)
        else:
            rho, down, up = stratified_trend(pooled_by_ladder)
            q_down, q_up = benjamini_hochberg(down), benjamini_hochberg(up)
            n_down, n_up = (rho < 0).sum(axis=0), (rho > 0).sum(axis=0)
            need = math.ceil(experiment.min_agree * len(lads))
            fading = (q_down < Q_MAX) & (n_down >= need) & (log2fc <= -MIN_LOG2FC)
            appearing = (q_up < Q_MAX) & (n_up >= need) & (log2fc >= MIN_LOG2FC)
        rows = []
        for j, w in enumerate(tested):
            word = words[w]
            gloss = glossary.get(word.strip(), {}).get("short", "") or next(
                (glossary[v.strip()]["short"] for v in variants[word] if v.strip() in glossary), "")
            rows.append({
                "word": word, "variants": sorted(variants[word]), "gloss": gloss,
                "trend": "fades" if fading[j] else "appears" if appearing[j] else "",
                "baseline": bool(base[j]), "vector_layers": vector_layers.get(int(w), []),
                "spread_steered": round(float(steered_spread[j]), 3),
                "spread_unsteered": round(float(unsteered_spread[j]), 3),
                "q_fade": None if by_question else float(q_down[j]),
                "q_appear": None if by_question else float(q_up[j]),
                "log2fc": round(float(log2fc[j]), 3),
                "agree": None if by_question else int(n_down[j] if log2fc[j] < 0 else n_up[j]),
                "onset": None if np.isnan(onset[j]) else round(float(onset[j]), 3),
                "rho": rho[:, j].round(3).tolist(),
                "permille": (1000 * pooled_by_ladder[:, :, j]).round(2).tolist(),        # [L, A]
            })
        result["metrics"][metric] = rows
    return result


def write_csv(path: Path, result: dict, metric: str) -> None:
    labels, schemes = result["labels"], [r["name"] for r in result["replicates"]]
    by_question = result["mode"] == "question"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        head = ["word", "variants", "gloss", "trend", "baseline", "vector_layers", "spread_steered",
                "spread_unsteered", "log2fc", "onset"]
        if by_question:
            writer.writerow(head + ["rho"] + [f"{label}_permille" for label in labels])
        else:
            writer.writerow(head + ["q_fade", "q_appear", "agree"] + [f"rho_{r}" for r in schemes]
                            + [f"{r}_{label}" for r in schemes for label in labels])
        for row in sorted(result["metrics"][metric], key=lambda r: (r["trend"] == "", -abs(r["log2fc"]))):
            common = [row["word"], " | ".join(map(repr, row["variants"])), row["gloss"], row["trend"],
                      int(row["baseline"]), layer_ranges(row["vector_layers"]), row["spread_steered"], row["spread_unsteered"], row["log2fc"],
                      row["onset"]]
            series = [v for s in row["permille"] for v in s]
            if by_question:
                writer.writerow(common + row["rho"] + series)
            else:
                writer.writerow(common + [f"{row['q_fade']:.3g}", f"{row['q_appear']:.3g}", row["agree"]]
                                + row["rho"] + series)


def column_legend(experiment: Experiment, result: dict | None, metric: str | None) -> str:
    """Plain-language explanation of the word-table columns."""
    common = [
        "**Columns**",
        "",
        "- **word** — tokens that differ only in leading space or capitalisation, merged.",
        "- **gloss** — dictionary translation for non-English tokens (from the report's offline glossary).",
        "- **baseline** — the layers at which the word is in the top-10 J-Lens readout of the character − default "
        "vector itself (any layer 0–62, word-like tokens, as in the steering runs): the steering vector's own "
        "readout rather than context-dependent content. Empty = not in the vector readout.",
    ]
    if experiment.replicate == "question":
        return "\n".join(common + [
            "- **Listed** when the word's series never moves against the trend along the ladder (ties allowed) and "
            "changes by at least 2× overall. One 3–4 point ladder cannot be tested for significance, so these "
            "lists describe each question; expect some noise among small counts.",
            "- **rho** — rank correlation of the series with strength order (−1 = strictly falling, +1 = strictly "
            "rising; ties pull it towards 0).",
            "- **log2FC** — log2 of (strongest strength) / (unsteered control). −1 = halved, +3 = eight times more.",
            "- **‰ by strength** — one number per strength, weakest to strongest: the word's count (or rank-weighted "
            "count) in thousandths of the question's grid (prompt positions × layers from the intervention layer "
            "to 62).",
            "",
        ])
    reps = result["replicates"]
    grids = ", ".join(f"{r['name']}: {r['positions']} × {r['layers']} = {r['grid']}" for r in reps)
    measure = ("the number of cells whose top-10 contains the word" if metric == "cells" else
               "the same count with each cell weighted by 1/rank (rank 1 = 1.0, rank 10 = 0.1)")
    fold = ("(average at the two strongest strengths) / (average at the two weakest)" if len(result["labels"]) > 5
            else "(strongest strength) / (unsteered control)")
    return "\n".join(common + [
        f"- **q** — false-discovery-rate adjusted p-value that the word changes monotonically with strength in "
        f"this direction, all {len(reps)} schemes combined. Below 0.05 = significant.",
        f"- **agree** — in how many of the {len(reps)} schemes the trend has this sign (all are required).",
        f"- **log2FC** — log2 of {fold}. −1 = halved, −3 = one eighth, +3 = eight times more.",
        f"- **<scheme> ‰ by strength** — one number per strength, weakest to strongest "
        f"({' '.join(result['labels'])}), for that scheme: {measure}, in thousandths of that scheme's grid "
        f"(prompt positions × layers from the first intervention layer to 62; {grids} cells).",
        "",
    ])


DISPLACEMENT_NOTE = ("How much of the readout the baseline words (the vector's own top-10 readout) take over. Each cell (one "
                     "prompt position at one layer) lists up to 10 distinct words; a percentage is the baseline "
                     "words' share of all those entries in that trial. Whatever they occupy is taken from every "
                     "other word, so a word can fade partly through crowding alone.\n")


def summary_section(experiment: Experiment, result: dict, level: int) -> list[str]:
    h = "#" * level
    by_question = result["mode"] == "question"
    reps, labels = result["replicates"], result["labels"]
    rows_shown = TABLE_ROWS[experiment.replicate]
    if by_question:
        lines = [f"\n{h} `{result['question']}` — {result['character']}\n",
                 f"{result['tested']} words tested. Baseline share of the readout: "
                 + ", ".join(f"{label} {100 * v:.0f}%" for label, v in zip(labels, result["baseline_share"][0])) + ".\n"]
    else:
        lines = [f"\n{h} `{result['character']}`\n",
                 f"{result['tested']} words reach {MIN_CELLS} cells in some trial and were tested.\n", DISPLACEMENT_NOTE,
                 "| scheme | " + " | ".join(labels) + " |", "|" + " --- |" * (1 + len(labels))]
        lines += [f"| {r['name']} | " + " | ".join(f"{100 * v:.0f}%" for v in share) + " |"
                  for r, share in zip(reps, result["baseline_share"])]
    for metric in METRICS:
        rows = result["metrics"][metric]
        lines.append(f"\n{h}# Metric: `{metric}`\n")
        for label, trend, qkey in (("Fading", "fades", "q_fade"), ("Appearing", "appears", "q_appear")):
            chosen = sorted((r for r in rows if r["trend"] == trend), key=lambda r: -abs(r["log2fc"]))
            lines.append(f"\n{h}## {label}: {len(chosen)} words ({sum(r['baseline'] for r in chosen)} baseline), "
                         f"top {min(rows_shown, len(chosen))} "
                         f"by |log2FC|\n")
            if not chosen:
                continue
            if by_question:
                lines.append("| word | gloss | baseline | rho | log2FC | ‰ by strength |")
                lines.append("|" + " --- |" * 6)
                for r in chosen[:rows_shown]:
                    lines.append(f"| `{r['word']}` | {r['gloss']} | {layer_ranges(r['vector_layers'])} "
                                 f"| {r['rho'][0]:+.2f} | {r['log2fc']:+.2f} | "
                                 + " ".join(f"{v:.1f}" for v in r["permille"][0]) + " |")
                continue
            lines.append(column_legend(experiment, result, metric) if label == "Fading"
                         else "Columns as in the fading table above.\n")
            per_scheme = len(reps) <= MAX_REPLICATE_COLUMNS
            series_head = [f"{r['name']} ‰ by strength" for r in reps] if per_scheme else ["mean ‰ by strength"]
            lines.append("| word | gloss | baseline | q | agree | log2FC | " + " | ".join(series_head) + " |")
            lines.append("|" + " --- |" * (6 + len(series_head)))
            for r in chosen[:rows_shown]:
                series = r["permille"] if per_scheme else [np.mean(r["permille"], axis=0).tolist()]
                lines.append(f"| `{r['word']}` | {r['gloss']} | {layer_ranges(r['vector_layers'])} "
                             f"| {r[qkey]:.1e} | {r['agree']}/{len(reps)} | {r['log2fc']:+.2f} | "
                             + " | ".join(" ".join(f"{v:.1f}" for v in s) for s in series) + " |")
        sanity = [r for r in rows if r["word"] in SANITY_WORDS and not by_question]
        if sanity:
            lines.append(f"\n{h}## Sanity check (words glossed \"to warn\")\n")
            lines.append("**trend** — `fades`/`appears` if it passed all criteria; **q fade** — adjusted p-value for a "
                         "decrease; **rho** — rank correlation of the word's per-strength values with strength, per "
                         "scheme (−1 = strictly falling, +1 = strictly rising).\n")
            lines.append("| word | trend | q fade | log2FC | rho |")
            lines.append("| --- | --- | --- | --- | --- |")
            lines += [f"| `{r['word']}` | {r['trend']} | {r['q_fade']:.1e} | {r['log2fc']:+.2f} | "
                      + " / ".join(f"{v:+.2f}" for v in r["rho"]) + " |" for r in sanity]
    return lines


def run(experiment: Experiment) -> None:
    trials, vocabulary, glossary = load_trials(experiment.bundle)
    lookup, words = build_groups(vocabulary)
    variants: dict[str, set[str]] = {}
    for text in vocabulary.values():
        variants.setdefault(word_key(text), set()).add(text)
    characters = list(dict.fromkeys(t.character for t in trials if t.character != "unsteered"))
    by_question = experiment.replicate == "question"

    out = OUT / experiment.name
    out.mkdir(parents=True, exist_ok=True)
    lines = [f"# J-Lens words that fade or appear with steering: `{experiment.name}`\n",
             f"{experiment.summary} Prompt positions only (through the newline before the answer), layers from the "
             f"first intervention layer to 62, ladders weakest → strongest.\n"]
    if by_question:
        lines += ["Each question is described on its own, from its single ladder.\n",
                  column_legend(experiment, None, None)]
    web_characters = []
    for character in characters:
        all_ladders = ladders(experiment, trials, character)
        units = [[lad] for lad in all_ladders] if by_question else [all_ladders]
        if by_question:
            lines.append(f"\n## `{character}`\n")
        results = []
        for unit in units:
            result = analyse(experiment, unit, character, lookup, words, variants, glossary)
            stem = f"{character}/{result['question']}" if by_question else character
            for metric in METRICS:
                write_csv(out / f"{stem}_{metric}.csv", result, metric)
            lines += summary_section(experiment, result, 3 if by_question else 2)
            results.append(result)
            print(experiment.name, character, result.get("question", ""),
                  {m: {t: sum(r["trend"] == t for r in result["metrics"][m]) for t in ("fades", "appears")}
                   for m in METRICS})
        web_characters.append({"character": character, "units": results})
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    web = {"name": experiment.name, "summary": experiment.summary, "replicate": experiment.replicate,
           "symbol": experiment.symbol, "characters": web_characters}
    (out / "web.json").write_text(json.dumps(web, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


def main() -> None:
    chosen = sys.argv[1:]
    for experiment in EXPERIMENTS:
        if not chosen or experiment.name in chosen:
            run(experiment)


if __name__ == "__main__":
    main()
