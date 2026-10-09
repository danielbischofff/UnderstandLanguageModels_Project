"""Exploratory: the persona map at other depths, and whether the user's tone shows there.

Fits one role map per layer (same t-SNE settings as the final layer-32 map, own
Ward-6 clusters; layer 32 reuses the final map) and draws them side by side.
With a tone-layers run (configs/experiments/tsne-lens/tone-layers.yaml, which
keeps residuals at these layers), it also draws the casual vs formal Tylenol
trajectories per layer and tests, per layer over the 6 topics, the share of
user tokens whose nearest role sits in infant's cluster (casual vs formal /
third person, exact Wilcoxon), as C4 did at layer 32.
With --space rd every layer's map is fitted on the steering vectors R - D instead,
and residuals are projected as x - D (as in rd_map.py); the tone test keeps the
same fixed role set.
Outputs: writeup/figures/tsne/layers/ (not in the paper), results/layer_maps[_rd].json.

    uv run --extra plots python src/control/layer_maps.py [--space r|rd] [--jobs 4]
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from scipy.stats import wilcoxon

import figures as F
from common import MODEL, ROOT, content_mask, save, unit
from tsne_lens.inputs import load_trajectory
from tsne_lens.palette import map_colors
from tsne_lens.projection import RoleMap

plt = F.plt
LAYERS = (8, 16, 24, 32, 40, 48, 56)
VECTORS = ROOT / "models/assistant-axis-vectors/qwen-3-32b"
CACHE = ROOT / "outputs/plots/tsne"
OUT = F.OUT / "layers"
TONE_RUNS = ROOT / "runtime/outputs/experiments/tone-layers/runs"
MARKED = ("assistant", "demon", "infant")


def cache(layer: int, rd: bool = False) -> Path:
    if rd:
        return CACHE / ("roles-rd" if layer == 32 else f"roles-rd-L{layer}") / "role-map.pkl"
    return CACHE / ("roles" if layer == 32 else f"roles-L{layer}") / "role-map.pkl"


def layer_vectors(labels: list[str], layer: int):
    roles = np.stack([torch.load(VECTORS / f"role_vectors/{label}.pt", map_location="cpu",
                                 weights_only=True)[layer].float().numpy() for label in labels])
    default = torch.load(VECTORS / "default_vector.pt", map_location="cpu", weights_only=True)
    return roles, default[layer].float().numpy()


def fit(job: tuple[int, bool]) -> str:
    layer, rd = job
    path = cache(layer, rd)
    if path.exists():
        return f"layer {layer}: cached"
    final = RoleMap.load(cache(32)).config
    labels = list(RoleMap.load(cache(32)).labels)
    roles, default = layer_vectors(labels, layer)
    if rd:
        roles, default = roles - default, None
    role_map = RoleMap.fit(
        roles, labels, layer=layer, metric="cosine",
        perplexity=tuple(final["perplexities"]), dof=final["dof"], seed=final["seed"],
        n_clusters=6, subclusters=6, max_iter=final["max_iter"], tolerance=final["tolerance"],
        early_exaggeration=final["early_exaggeration"],
        early_exaggeration_iter=final["early_exaggeration_iter"],
        exaggeration_anneal_iter=final["exaggeration_anneal_iter"],
        exaggeration=final["exaggeration"], n_iter=final["n_iter"],
        relaxation_iter=final["relaxation_iter"], theta=final["theta"], default_vector=default,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    role_map.save(path)
    return f"layer {layer}: fitted"


def draw_map(ax, role_map, layer):
    roles = role_map.role_records()
    colors = map_colors(role_map)
    ax.scatter([r["x"] for r in roles], [r["y"] for r in roles], s=4,
               c=[colors[r["cluster"]] for r in roles], alpha=0.5, linewidths=0)
    for r in roles:
        if r["label"] in MARKED:
            ax.annotate(r["label"], (r["x"], r["y"]), xytext=(2, 2), textcoords="offset points",
                        fontsize=5.5, color=F.INK)
    star = next((p for p in role_map.config.get("reference_points", [])
                 if p["label"] == "default_vector"), None)
    if star:
        ax.scatter(star["x"], star["y"], marker="*", s=30, color=F.INK, zorder=5)
    q = role_map.config["quality"]
    ax.set_title(f"layer {layer} · recall {q['neighbor_recall']:.2f}", loc="left", fontsize=7)
    ax.set_xticks([]), ax.set_yticks([])
    for side in ax.spines.values():
        side.set_visible(False)


def tone_trials(run: Path) -> dict[str, Path]:
    manifest = json.loads((run / "manifest.json").read_text())
    return {t["question"]: run / t["path"] for t in manifest["trials"].values()
            if t["status"] == "complete"}


def tone_share(path: Path, layer: int, roles_unit, members, offset=None) -> float:
    """Share of the user's tokens whose nearest role is one of ``members`` (bool mask)."""
    x, rows = load_trajectory(path, layer=layer, model_id=MODEL, prompt_only=True)
    x = x[content_mask([r["token"] for r in rows])].astype(np.float64)
    if offset is not None:
        x = x - offset
    return float(members[(unit(x) @ roles_unit.T).argmax(1)].mean())


def paired(share: dict, topics: list[str]) -> dict:
    block = {}
    for other in ("formal", "third"):
        a = [share[f"{t}-casual"] for t in topics]
        b = [share[f"{t}-{other}"] for t in topics]
        block[f"casual_vs_{other}"] = dict(
            casual_mean=float(np.mean(a)), other_mean=float(np.mean(b)),
            topics_casual_higher=int(sum(x > y for x, y in zip(a, b))),
            p_exact=float(wilcoxon(a, b, method="exact").pvalue) if a != b else 1.0)
    return block


def tone_tests(maps: dict, trials: dict, offsets: dict) -> dict:
    """Per layer: (a) infant's own Ward cluster at that layer, which splits off at
    some depths; (b) a fixed set, the ten roles of infant's cluster at layer 32,
    so the measure means the same thing at every layer."""
    topics = sorted({q.rsplit("-", 1)[0] for q in trials})
    ref = RoleMap.load(cache(32))
    ref_clusters = np.asarray(ref.clusters)
    fixed = {label for label, c in zip(ref.labels, ref_clusters)
             if c == ref_clusters[list(ref.labels).index("infant")]}
    result = {"fixed_set": sorted(fixed)}
    for layer, role_map in maps.items():
        labels = list(role_map.labels)
        roles_unit = unit(role_map.vectors.astype(np.float64))
        clusters = np.asarray(role_map.clusters)
        own = clusters == clusters[labels.index("infant")]
        in_fixed = np.array([label in fixed for label in labels])
        block = dict(infant_cluster=role_map.cluster_names[int(clusters[labels.index("infant")])])
        for name, members in (("infant_cluster_share", own), ("fixed_set_share", in_fixed)):
            share = {q: tone_share(p, layer, roles_unit, members, offsets[layer])
                     for q, p in trials.items()}
            block[name] = dict(shares=share, **paired(share, topics))
        result[str(layer)] = block
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--space", choices=["r", "rd"], default="r")
    args = parser.parse_args()
    rd = args.space == "rd"
    suffix = "_rd" if rd else ""
    with ProcessPoolExecutor(args.jobs) as pool:
        for message in pool.map(fit, [(layer, rd) for layer in LAYERS]):
            print(message, flush=True)
    maps = {layer: RoleMap.load(cache(layer, rd)) for layer in LAYERS}
    labels = list(RoleMap.load(cache(32)).labels)
    offsets = {layer: layer_vectors(labels[:1], layer)[1].astype(np.float64) if rd else None
               for layer in LAYERS}
    OUT.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 4, figsize=(F.PAGE, 3.3))
    for ax, (layer, role_map) in zip(axes.flat, maps.items()):
        draw_map(ax, role_map, layer)
    axes.flat[-1].axis("off")
    fig.savefig(OUT / f"maps_by_layer{suffix}.pdf")
    plt.close(fig)

    result = dict(layers={str(layer): dict(
        quality={k: m.config["quality"][k] for k in
                 ("neighbor_recall", "trustworthiness", "global_distance_spearman",
                  "cluster_silhouette_original", "cluster_silhouette_2d")},
        clusters=m.cluster_names,
        cluster_of={label: m.cluster_names[int(m.clusters[list(m.labels).index(label)])]
                    for label in MARKED + ("toddler",)},
    ) for layer, m in maps.items()})

    runs = sorted(TONE_RUNS.iterdir()) if TONE_RUNS.exists() else []
    if runs:
        trials = tone_trials(runs[-1])
        result["tone"] = tone_tests(maps, trials, offsets)
        fig, axes = plt.subplots(2, 4, figsize=(F.PAGE, 3.3))
        for ax, (layer, role_map) in zip(axes.flat, maps.items()):
            draw_map(ax, role_map, layer)
            for question, cmap in (("tylenol-casual", "RdPu"), ("tylenol-formal", "Blues")):
                x, rows = load_trajectory(trials[question], layer=layer, model_id=MODEL,
                                          prompt_only=True)
                if rd:
                    x = x - offsets[layer]
                pts = role_map.project(x, rows)
                xy = np.array([[p["x"], p["y"]] for p in pts])
                xy += np.random.default_rng(0).normal(scale=3.0, size=xy.shape)
                ax.plot(xy[:, 0], xy[:, 1], color=plt.get_cmap(cmap)(0.6), lw=0.5, alpha=0.5)
                ax.scatter(xy[:, 0], xy[:, 1], c=np.arange(len(xy)), s=6, zorder=4,
                           cmap=plt.get_cmap(cmap), vmin=-len(xy) * 0.4, vmax=len(xy))
            t = result["tone"][str(layer)]["fixed_set_share"]["casual_vs_formal"]
            ax.set_title(f"L{layer}: {t['casual_mean']:.2f} / {t['other_mean']:.2f}",
                         loc="left", fontsize=6.5)
        axes.flat[-1].axis("off")
        fig.savefig(OUT / f"trajectories_by_layer{suffix}.pdf")
        plt.close(fig)
    else:
        print("no tone-layers run yet: maps only")

    save(f"layer_maps{suffix}", result)
    print(json.dumps(result, indent=1)[:3000])


if __name__ == "__main__":
    main()
