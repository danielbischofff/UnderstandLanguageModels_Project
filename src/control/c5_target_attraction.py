"""C5: when does a steered residual move to the steered role, and what predicts it?

For every trial of both 17-role Tylenol sweeps and both token segments (prompt
without position 0, and the generated answer), this measures how close each
layer-32 residual is to the role map: absolute proximity (max cosine, share of
tokens above 0.5), relative proximity to the target (its rank among 275 roles),
and the Ward cluster of the nearest role. Each role is then described by its
steering vector R - D (length, share along the Assistant Axis, distinctiveness),
and Spearman correlations ask which property predicts reaching the target.
Prompt trajectories also get a coarse type label (for the taxonomy in the paper).

    uv run --extra plots python src/control/c5_target_attraction.py
"""

from __future__ import annotations

import csv

import numpy as np
from scipy.stats import spearmanr

from common import NEAR, RESULTS, load_space, residuals, runs, save, trials, unit, content_mask


def segment_metrics(x, space, target, baseline_share=None):
    """Per-segment summary of where the residuals sit relative to the roles."""
    cos = unit(x) @ space.unit.T
    t = space.index(target)
    nearest = space.clusters[cos.argmax(1)]
    own_cluster = space.clusters[t]
    c_hat = unit(space.steering(target)) if target != "assistant" else None
    result = dict(
        tokens=len(x),
        maxcos=cos.max(1).mean(),
        near=(cos.max(1) > NEAR).mean(),
        rank_tgt=float(np.median((cos > cos[:, [t]]).sum(1) + 1)),
        cos_tgt=cos[:, t].mean(),
        margin_tgt=(cos[:, t] - np.median(cos, 1)).mean(),
        own=(nearest == own_cluster).mean(),
        clusters=np.bincount(nearest, minlength=6).tolist(),
        norm=float(np.median(np.linalg.norm(x, axis=1))),
    )
    if c_hat is not None:
        result["align_c"] = (unit(x) @ c_hat).mean()
    if baseline_share is not None:
        result["own_excess"] = result["own"] - baseline_share[own_cluster]
    return result, nearest


def trajectory_type(nearest, tokens, target_cluster, near):
    """Coarse label of a prompt trajectory (position 0 already removed)."""
    content = content_mask(["<|im_start|>"] + tokens)[1:]
    on_content = (nearest[content] == target_cluster).mean()
    on_template = (nearest[~content] == target_cluster).mean()
    if near < 0.1:
        kind = "off-map"  # nothing in the map is close; nearest labels are weak
    elif on_content >= 0.8 and on_template >= 0.8:
        kind = "locked"  # target cluster everywhere, template tokens included
    elif on_content >= 0.5:
        kind = "content-only"  # target on content words, back elsewhere on template
    else:
        kind = "baseline-like"
    return kind, on_content, on_template


def role_properties(space, roles):
    """What each steering vector R - D looks like, for the correlations."""
    axis = unit(space.default - space.mean)  # Assistant Axis of Lu et al.
    centered = unit(space.vectors - space.mean)
    props = {}
    for role in roles:
        i = space.index(role)
        c = space.steering(role)
        other = space.clusters != space.clusters[i]
        props[role] = dict(
            cluster=int(space.clusters[i]),
            norm_rd=float(np.linalg.norm(c)),
            axis_share=float(unit(c) @ -axis),  # cos(R - D, mu - D)
            isolation=float(1 - (centered[other] @ centered[i]).max()),
        )
    return props


def main() -> None:
    space = load_space()
    records, baselines = [], {}
    for method, run in runs().items():
        for trial in trials(method, run):
            segments = residuals(trial)
            if trial["character"] == "unsteered":
                for segment, (x, _) in segments.items():
                    x = x[1:] if segment == "prompt" else x
                    nearest = space.clusters[(unit(x) @ space.unit.T).argmax(1)]
                    baselines[(method, segment)] = np.bincount(nearest, minlength=6) / len(x)
            records.append((trial, segments))

    rows = []
    for trial, segments in records:
        target = "assistant" if trial["character"] == "unsteered" else trial["character"]
        row = {k: trial[k] for k in ("run", "method", "character", "layer", "coefficient", "judge")}
        for segment, (x, tokens) in segments.items():
            if segment == "prompt":
                x, tokens = x[1:], tokens[1:]  # position 0 is the attention sink
            metrics, nearest = segment_metrics(
                x, space, target, baselines.get((trial["run"], segment))
            )
            row.update({f"{segment}_{k}": v for k, v in metrics.items()})
            if segment == "prompt":
                kind, on_content, on_template = trajectory_type(
                    nearest, tokens, space.clusters[space.index(target)], metrics["near"]
                )
                row.update(prompt_type=kind, prompt_content_on_target=on_content,
                           prompt_template_on_target=on_template)
        rows.append(row)

    roles = sorted({r["character"] for r in rows} - {"unsteered"})
    props = role_properties(space, roles)
    correlations = {}
    for method in ("alpha", "band"):
        for layer in (29, 32):
            for coefficient in sorted({r["coefficient"] for r in rows if r["method"] == method}):
                group = [r for r in rows if r["method"] == method and r["layer"] == layer
                         and r["coefficient"] == coefficient]
                key = f"{method}-L{layer}-{coefficient:g}"
                correlations[key] = {
                    f"{segment}_{metric}~{prop}": dict(zip(("rho", "p"), spearmanr(
                        [r[f"{segment}_{metric}"] for r in group],
                        [props[r["character"]][prop] for r in group],
                    )))
                    for segment in ("prompt", "generated")
                    for metric in ("rank_tgt", "own_excess", "near")
                    for prop in ("norm_rd", "axis_share", "isolation")
                }
    types = {}
    for r in rows:
        types.setdefault(r["method"], {}).setdefault(r["prompt_type"], 0)
        types[r["method"]][r["prompt_type"]] += 1

    save("c5_target_attraction", dict(
        trials=len(rows), role_properties=props, prompt_types=types,
        baseline_cluster_share={f"{m}-{s}": v.tolist() for (m, s), v in baselines.items()},
        correlations=correlations,
    ))
    with (RESULTS / "c5_trials.csv").open("w", newline="") as handle:
        fields = list(dict.fromkeys(k for r in rows for k in r))
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    print(types)
    for key in ("alpha-L32-4", "band-L32-0.1"):
        print(key, {k: v for k, v in correlations[key].items() if k.startswith("prompt_rank")})


if __name__ == "__main__":
    main()
