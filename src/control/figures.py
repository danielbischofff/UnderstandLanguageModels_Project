"""Paper figures for the 2D role-space lens, written to writeup/figures/tsne/.

Map figures reuse tsne_lens.plots.make_figure on the saved map
(outputs/plots/tsne/roles/role-map.pkl), so palette, labels and the default
star match roles.png; only size, fonts and legend placement change for print.
The other figures are matplotlib, read from src/control/results and the ribbon
JSON, with the same fixed cluster colours.

    uv run --extra plots python src/control/figures.py [figure ...]   # none = all
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys

# Fixed PDF dates: re-rendering an unchanged figure leaves the file byte-identical.
os.environ.setdefault("SOURCE_DATE_EPOCH", "0")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from common import NEAR, RESULTS, ROOT, load_space, residuals, runs, trials, unit  # noqa: E402

sys.path.insert(0, str(ROOT / "src" / "tsne-lens" / "scripts"))
from cluster_ribbons import display_token, tint_for  # noqa: E402
from tsne_lens.inputs import load_trajectory  # noqa: E402
from tsne_lens.palette import map_colors  # noqa: E402
import tsne_lens.plots as tsne_plots  # noqa: E402
from tsne_lens.plots import make_figure  # noqa: E402
from tsne_lens.projection import RoleMap  # noqa: E402

OUT = ROOT / "writeup/figures/tsne"
TSNE = ROOT / "outputs/plots/tsne"
INK, MUTED, GRID = "#1f1f1f", "#8a8983", "#dcdbd5"
COLUMN, PAGE = 3.15, 6.3  # ACL column and page widths, inches

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "savefig.bbox": "tight", "savefig.dpi": 300,
})


def print_layout(fig, width=1100, height=760):
    """Paper version of a make_figure plot: no title, larger text, legend below."""
    fig.layout.sliders, fig.layout.updatemenus = (), ()
    fig.update_layout(title=None, width=width, height=height, font=dict(size=15, family="Times New Roman"),
                      legend=dict(orientation="h", x=0, y=-0.12, font=dict(size=14)),
                      margin=dict(l=60, r=30, t=20, b=150))
    for annotation in fig.layout.annotations:
        if annotation.font and annotation.font.size:
            annotation.font.size = annotation.font.size + 3
    return fig


def trial_dir(method, character, layer=None, coefficient=None):
    run = runs()[method]
    for t in trials(method, run):
        if t["character"] == character and (character == "unsteered" or
                                            (t["layer"] == layer and t["coefficient"] == coefficient)):
            return t["path"]
    raise KeyError((method, character, layer, coefficient))


def roles_map_figure(role_map):
    print_layout(make_figure(role_map, [])).write_image(OUT / "roles_map.pdf")


def trajectories_figure(role_map):
    """Appendix: three prompt trajectories as small multiples on the frozen map."""
    picks = [("demon · α=4 · L32 (harmful)", trial_dir("alpha", "demon", 32, 4.0)),
             ("toddler · β=0.1 · L32 (harmful)", trial_dir("band", "toddler", 32, 0.1)),
             ("organizer · β=0.25 · L32 (harmful)", trial_dir("band", "organizer", 32, 0.25))]
    trajectories = []
    for name, path in picks:
        vectors, rows = load_trajectory(path, layer=32, model_id=role_map.config["model_id"], prompt_only=True)
        trajectories.append(dict(name=name, points=role_map.project(vectors, rows)))
    # Small multiples on the same frozen map: one trajectory per panel reads far
    # better than three overlaid ones that all start near the Assistant.
    roles = role_map.role_records()
    colors = map_colors(role_map)
    rx, ry = np.array([r["x"] for r in roles]), np.array([r["y"] for r in roles])
    rc = [colors[r["cluster"]] for r in roles]
    star = next(p for p in role_map.config["reference_points"] if p["label"] == "default_vector")
    fig, axes = plt.subplots(1, 3, figsize=(PAGE, 2.35), sharex=True, sharey=True)
    for ax, trajectory in zip(axes, trajectories):
        pts = trajectory["points"]
        x, y = np.array([p["x"] for p in pts]), np.array([p["y"] for p in pts])
        ax.scatter(rx, ry, s=7, c=rc, alpha=0.45, linewidths=0)
        for cluster in sorted(colors):
            members = [r for r in roles if r["cluster"] == cluster]
            ax.text(np.median([r["x"] for r in members]), np.median([r["y"] for r in members]) + 12,
                    str(cluster), fontsize=7, color=INK, ha="center")
        ax.scatter(star["x"], star["y"], marker="*", s=40, color=INK, zorder=4)
        ax.plot(x[1:], y[1:], color=INK, linewidth=0.5, alpha=0.6, zorder=2)
        sc = ax.scatter(x[1:], y[1:], c=np.arange(1, len(x)), cmap="viridis_r", s=9, marker="s",
                        linewidths=0, zorder=3)
        ax.scatter(x[0], y[0], s=40, facecolors="none", edgecolors=INK, linewidths=0.9, zorder=4)
        ax.scatter(x[-1], y[-1], s=30, marker="D", facecolors="none", edgecolors=INK, linewidths=0.9, zorder=4)
        ax.set_title(trajectory["name"], loc="left", fontsize=7)
        ax.set_xticks([]), ax.set_yticks([])
        for side in ax.spines.values():
            side.set_color(GRID)
    cbar = fig.colorbar(sc, ax=axes, fraction=0.015, pad=0.01)
    cbar.set_label("prompt token position", fontsize=7)
    cbar.ax.tick_params(labelsize=6)
    fig.savefig(OUT / "trajectories.pdf")
    plt.close(fig)


def tone_runs(role_map, offset=None):
    """Unsteered Tylenol prompt in its original (casual) and formal wording, from the C4 run.

    ``offset`` is subtracted from every residual first (the R-D map projects x - D)."""
    run = sorted((ROOT / "runtime/outputs/experiments/register-control/runs").iterdir())[-1]
    manifest = json.loads((run / "manifest.json").read_text())
    paths = {t["question"]: run / t["path"] for t in manifest["trials"].values()}
    out = []
    for question, label, cmap in (("tylenol-casual", "casual", "RdPu"),
                                  ("tylenol-formal", "formal", "Blues")):
        vectors, rows = load_trajectory(paths[question], layer=32,
                                        model_id=role_map.config["model_id"], prompt_only=True)
        if offset is not None:
            vectors = vectors - offset
        out.append((label, role_map.project(vectors, rows), cmap))
    return out


def trajectory_figure(role_map, runs, out=None, names=None):
    """One-column figure for the methods: prompt trajectories, position 0 included.

    ``runs`` is a list of (label, projected points, colormap); each path runs light
    (first token) to dark (last token) in its own hue, so two wordings stay apart.
    """
    from matplotlib.collections import LineCollection
    from matplotlib.colors import LinearSegmentedColormap

    roles = role_map.role_records()
    colors = map_colors(role_map)
    fig, ax = plt.subplots(figsize=(COLUMN, 2.45))
    ax.scatter([r["x"] for r in roles], [r["y"] for r in roles], s=6,
               c=[colors[r["cluster"]] for r in roles], alpha=0.3, linewidths=0)
    centers = {c: np.median([[r["x"], r["y"]] for r in roles if r["cluster"] == c], axis=0)
               for c in colors}
    if names is None:
        names = {0: ("0 · infant, fool,\nadolescent", (-8, 14)), 3: ("3 · assistant-like", (20, -58))}
    for c, (text, shift) in names.items():
        ax.text(*(centers[c] + shift), text, fontsize=6.5, color=INK, ha="center")
    # The default vector, or the assistant role on a map that has no default point.
    star = next((p for p in role_map.config.get("reference_points", []) if p["label"] == "default_vector"),
                next(r for r in roles if r["label"] == "assistant"))
    ax.scatter(star["x"], star["y"], marker="*", s=45, color=INK, zorder=6)
    bars = []
    for label, points, cmap in runs:
        # Skip the palest end of the ramp so the first tokens stay visible on white.
        ramp = LinearSegmentedColormap.from_list(cmap, plt.get_cmap(cmap)(np.linspace(0.3, 1.0, 256)))
        xy = np.array([[p["x"], p["y"]] for p in points])
        # Many tokens land on the same spot; a small fixed jitter shows how many.
        xy = xy + np.random.default_rng(0).normal(scale=3.0, size=xy.shape)
        segments = np.stack([xy[:-1], xy[1:]], axis=1)
        lines = LineCollection(segments, cmap=ramp, linewidths=0.8, alpha=0.55, zorder=3)
        lines.set_array(np.arange(len(segments)))
        ax.add_collection(lines)
        ax.scatter(xy[:, 0], xy[:, 1], c=np.arange(len(xy)), cmap=ramp, s=22, zorder=4,
                   edgecolors="white", linewidths=0.4)
        bars.append((label, lines))
    first = np.array([[runs[0][1][0]["x"], runs[0][1][0]["y"]]])[0]
    ax.scatter(*first, s=34, facecolors="none", edgecolors=INK, linewidths=1.0, zorder=5)
    ax.annotate("position 0 (sink)", first, xytext=(-28, -24), textcoords="offset points",
                fontsize=6.5, color=INK, ha="right", va="center",
                arrowprops=dict(arrowstyle="-", color=INK, lw=0.6, shrinkB=4))
    # One vertical ramp per wording, scaled in token positions.
    for i, (label, lines) in enumerate(bars):
        cax = ax.inset_axes([1.03 + 0.16 * i, 0.08, 0.035, 0.80])
        cbar = fig.colorbar(lines, cax=cax)
        cbar.solids.set_alpha(1)
        cbar.outline.set_visible(False)
        cax.set_title(label, fontsize=6.5, color=INK, pad=3)
        cax.tick_params(labelsize=6, length=2)
        if i < len(bars) - 1:
            cax.set_yticklabels([])
    ax.text(1.03 + 0.16 * len(bars) - 0.01, 0.48, "token position", transform=ax.transAxes,
            rotation=90, va="center", fontsize=6.5, color=INK)
    ax.set_xticks([]), ax.set_yticks([])
    for side in ax.spines.values():
        side.set_visible(False)
    fig.savefig(out or OUT / "trajectory.pdf")
    plt.close(fig)


def ribbon_axes(ax, role_map, found, gaps, xticks=True, fontsize=7):
    """Draw ribbon rows (label, record) on ``ax``: nearest cluster per prompt token, one row per trial."""
    colors = map_colors(role_map)
    tokens = [display_token(t).replace("⏎", "\\n") for t in found[0][1]["tokens"]]
    for row, (label, rec) in enumerate(found):
        y = row + gaps[row]
        for x, (cluster, sim) in enumerate(zip(rec["clusters"], rec["similarities"])):
            if x == 0:  # attention sink: shown, but not read
                face, alpha = "#d9d9d6", 1.0
            elif cluster is None:
                face, alpha = "white", 1.0
            else:
                face, alpha = colors[int(cluster)], tint_for(sim)
            ax.add_patch(plt.Rectangle((x + 0.06, y + 0.08), 0.88, 0.84, facecolor=face, alpha=alpha,
                                       edgecolor=GRID, linewidth=0.3))
        if rec["outcome"] in (1, 2):
            ax.text(len(tokens) + 0.4, y + 0.5, {1: "ignorant", 2: "harmful"}[rec["outcome"]],
                    va="center", fontsize=fontsize, color=INK)
    ax.set_xlim(0, len(tokens) + 4)
    ax.set_ylim(len(found) + gaps[len(found) - 1], 0)
    ax.set_yticks([row + gaps[row] + 0.5 for row in range(len(found))], [label for label, _ in found],
                  fontsize=fontsize)
    # Row labels take the colour of the steered role's cluster, as in the legend.
    cluster_of = dict(zip(role_map.labels, role_map.clusters))
    for tick, (_, rec) in zip(ax.get_yticklabels(), found):
        if rec["character"] in cluster_of:
            tick.set_color(colors[int(cluster_of[rec["character"]])])
    if xticks:
        ax.set_xticks(np.arange(len(tokens)) + 0.5, tokens, rotation=90, fontsize=fontsize - 1)
    else:
        ax.set_xticks([])
    ax.tick_params(length=0)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)


def ribbon_legend(target, role_map, **kwargs):
    colors = map_colors(role_map)
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[c]) for c in sorted(colors)]
    names = [f"{c} · {role_map.cluster_names[c]}" for c in sorted(colors)]
    target.legend(handles + [plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor=GRID),
                             plt.Rectangle((0, 0), 1, 1, facecolor="#d9d9d6")],
                  names + ["no role above 0.5", "position 0 (sink)"], ncol=4, frameon=False,
                  handlelength=1, columnspacing=1.0, **kwargs)


def draw_ribbon(role_map, found, gaps, out, row_height=0.26):
    fig, ax = plt.subplots(figsize=(PAGE, row_height * len(found) + 1.2))
    ribbon_axes(ax, role_map, found, gaps)
    ribbon_legend(ax, role_map, loc="lower center", bbox_to_anchor=(0.45, 1.0), fontsize=6.5)
    fig.savefig(out)
    plt.close(fig)


def ribbon_record(data, run, character, layers=None, coefficient=None):
    return next(r for r in data["records"] if r["run"] == run and r["character"] == character
                and (layers is None or (r["layers"] == layers and r["coefficient"] == coefficient)))


def ribbon_figure(role_map):
    data = json.loads((TSNE / "roles-cluster-ribbons/ribbons.json").read_text())
    rows = [("alpha", "unsteered", None, None, "unsteered"),
            ("alpha", "demon", [29], 4.0, "demon · α=4 · L29"),
            ("alpha", "demon", [32], 4.0, "demon · α=4 · L32"),
            ("alpha", "toddler", [32], 4.0, "toddler · α=4 · L32"),
            ("band", "demon", [32], 0.1, "demon · β=0.1 · L32"),
            ("band", "toddler", [29], 0.1, "toddler · β=0.1 · L29"),
            ("band", "toddler", [32], 0.1, "toddler · β=0.1 · L32"),
            ("band", "toddler", [32], 0.25, "toddler · β=0.25 · L32"),
            ("band", "organizer", [32], 0.25, "organizer · β=0.25 · L32")]
    found = [(label, ribbon_record(data, run, character, layers, coefficient))
             for run, character, layers, coefficient, label in rows]
    # A small gap after the unsteered row and after the alpha block.
    gaps = {0: 0.0, 1: 0.35, 2: 0.35, 3: 0.35, 4: 0.7, 5: 0.7, 6: 0.7, 7: 0.7, 8: 0.7}
    draw_ribbon(role_map, found, gaps, OUT / "ribbon_compact.pdf")


def ribbon_settings_figures(role_map):
    """Appendix, one page: all 17 steered personas at three settings (layer 32), grouped
    by the target's cluster, unsteered row on top of each panel."""
    data = json.loads((TSNE / "roles-cluster-ribbons/ribbons.json").read_text())
    cluster_of = dict(zip(role_map.labels, role_map.clusters))
    panels = []
    for run, coefficient, title in (("alpha", 2.5, "alpha steering, α = 2.5"),
                                    ("alpha", 4.0, "alpha steering, α = 4"),
                                    ("band", 0.1, "band capping, β = 0.1")):
        records = [r for r in data["records"] if r["run"] == run and r["layers"] == [32]
                   and r["coefficient"] == coefficient]
        records.sort(key=lambda r: (int(cluster_of[r["character"]]), r["character"]))
        found = [("unsteered", ribbon_record(data, run, "unsteered"))]
        found += [(f"{int(cluster_of[r['character']])} · {r['character']}", r) for r in records]
        gaps, gap, previous = {0: 0.0}, 0.3, None
        for i, (_, r) in enumerate(found[1:], start=1):
            cluster = int(cluster_of[r["character"]])
            if previous is not None and cluster != previous:
                gap += 0.3
            gaps[i], previous = gap, cluster
        panels.append((title, found, gaps))
    heights = [len(f) + g[len(f) - 1] for _, f, g in panels]
    fig, axes = plt.subplots(3, 1, figsize=(PAGE, 8.2), gridspec_kw=dict(height_ratios=heights, hspace=0.12))
    for i, (ax, (title, found, gaps)) in enumerate(zip(axes, panels)):
        ribbon_axes(ax, role_map, found, gaps, xticks=i == len(panels) - 1, fontsize=6)
        ax.set_title(title, loc="left", fontsize=7)
    fig.subplots_adjust(top=0.94)
    ribbon_legend(fig, role_map, loc="upper center", bbox_to_anchor=(0.5, 1.0), fontsize=6)
    fig.savefig(OUT / "ribbon_settings_L32.pdf")
    plt.close(fig)


def rank_figure(role_map):
    """One point per steered role (alpha): its steering-vector length against how
    near the target role gets, median and range over the 6 alpha trials."""
    props = json.loads((RESULTS / "c5_target_attraction.json").read_text())["role_properties"]
    colors = map_colors(role_map)
    with (RESULTS / "c5_trials.csv").open() as handle:
        rows = [r for r in csv.DictReader(handle) if r["method"] == "alpha"]
    fig, ax = plt.subplots(figsize=(COLUMN, 2.3))
    offsets = {"organizer": (4, -2), "mathematician": (4, 2), "doctor": (4, -6), "demon": (2, 8),
               "toddler": (-2, -11), "perfectionist": (4, 3)}
    for role in sorted({r["character"] for r in rows}):
        ranks = [float(r["prompt_rank_tgt"]) for r in rows if r["character"] == role]
        x, c = props[role]["norm_rd"], colors[props[role]["cluster"]]
        ax.plot([x, x], [min(ranks), max(ranks)], color=c, linewidth=1.0, alpha=0.6)
        harmful = any(r["judge"] == "2" for r in rows if r["character"] == role)
        ax.scatter(x, np.median(ranks), s=26, color=c, edgecolors=INK if harmful else "white",
                   linewidths=0.9 if harmful else 0.4, zorder=3)
        if role in offsets:
            dx, dy = offsets[role]
            ax.annotate(role, (x, np.median(ranks)), xytext=(dx, dy), textcoords="offset points",
                        fontsize=6.5, color=INK, ha="left" if dx > 0 else "right", va="center")
    ax.set_yscale("log")
    ax.invert_yaxis()
    ax.set_yticks([1, 3, 10, 30, 100], ["1", "3", "10", "30", "100"])
    ax.set_xlabel("length of the steering vector ‖R − D‖")
    ax.set_ylabel("rank of the target role\n(of 275; 1 = nearest)")
    ax.grid(axis="y", color=GRID, linewidth=0.4)
    fig.savefig(OUT / "rank_vs_norm.pdf")
    plt.close(fig)


def tone_figure():
    data = json.loads((RESULTS / "c4_register.json").read_text())
    prompts = next(v for k, v in data.items() if k.startswith("0001_"))["prompts"]
    tones = [("casual", "casual"), ("formal", "formal"), ("third", "third person")]
    topics = sorted({q.rsplit("-", 1)[0] for q in prompts})
    fig, ax = plt.subplots(figsize=(COLUMN, 2.1))
    ends = []
    for topic in topics:
        y = [prompts[f"{topic}-{t}"]["prompt"]["cluster0"] for t, _ in tones]
        ax.plot(range(3), y, color=MUTED, linewidth=0.8, marker="o", markersize=3)
        ends.append([y[-1], topic])
    ends.sort()
    for k in range(1, len(ends)):  # keep end labels at least 0.05 apart
        ends[k][0] = max(ends[k][0], ends[k - 1][0] + 0.05)
    for y, topic in ends:
        ax.text(2.08, y, topic, va="center", fontsize=6, color=MUTED)
    means = [np.mean([prompts[f"{t}-{tone}"]["prompt"]["cluster0"] for t in topics]) for tone, _ in tones]
    ax.plot(range(3), means, color=INK, linewidth=1.8, marker="s", markersize=4, label="mean of 6 topics")
    ax.set_xticks(range(3), [label for _, label in tones])
    ax.set_xlim(-0.2, 2.5)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("user tokens nearest the\nchildlike/impulsive cluster")
    ax.grid(axis="y", color=GRID, linewidth=0.4)
    ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(1.0, 1.0))
    fig.savefig(OUT / "tone_control.pdf")
    plt.close(fig)


def offrails_figure():
    with (RESULTS / "c14_offrails_trials.csv").open() as handle:
        rows = [r for r in csv.DictReader(handle) if r["run"].startswith("mini")]
    signatures = [("prompt_own_excess", 1, "prompt: share in the target's cluster"),
                  ("prompt_axis", -1, "prompt: drift down the Assistant Axis"),
                  ("answer_axis", -1, "answer: drift down the Assistant Axis")]
    fig, axes = plt.subplots(1, 2, figsize=(PAGE, 1.35), sharey=True)
    for ax, (code, title) in zip(axes, (("1", "in character (98 of 768)"), ("2", "broken (146 of 768)"))):
        groups = {}
        for r in rows:
            groups.setdefault((r["run"], r["character"], r["setting"]), []).append(r)
        informative = [g for g in groups.values() if 0 < sum(r["judge"] == code for r in g) < len(g)]
        for i, (key, sign, label) in enumerate(signatures):
            y = np.array([r["judge"] == code for r in rows])
            x = sign * np.array([float(r[key]) if r[key] else np.nan for r in rows])
            keep = ~np.isnan(x)
            overall = roc_auc_score(y[keep], x[keep])
            aucs, weights = [], []
            for g in informative:
                gy = np.array([r["judge"] == code for r in g])
                gx = sign * np.array([float(r[key]) if r[key] else np.nan for r in g])
                ok = ~np.isnan(gx)
                if 0 < gy[ok].sum() < ok.sum():
                    aucs.append(roc_auc_score(gy[ok], gx[ok]))
                    weights.append(gy[ok].sum() * (~gy[ok]).sum())
            within = np.average(aucs, weights=weights)
            ax.plot([within, overall], [i, i], color=GRID, linewidth=1.2, zorder=1)
            ax.scatter(overall, i, s=22, color=INK, zorder=2, label="all trials" if i == 0 else None)
            ax.scatter(within, i, s=22, facecolors="white", edgecolors=INK, zorder=2,
                       label="questions of one persona and setting" if i == 0 else None)
        ax.axvline(0.5, color=MUTED, linewidth=0.6, linestyle="--")
        ax.set_xlim(0.15, 0.95)
        ax.set_title(title, loc="left")
        ax.set_xlabel("AUC (0.5 = chance)")
        ax.set_yticks(range(len(signatures)), [s[2] for s in signatures])
    axes[0].invert_yaxis()  # shared y: inverting both would cancel out
    fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, ncol=2, fontsize=7,
               loc="upper center", bbox_to_anchor=(0.6, -0.12))
    fig.savefig(OUT / "offrails_auc.pdf")
    plt.close(fig)


def null_figure(space):
    mu = unit(space.mean)
    rng = np.random.default_rng(0)
    real, null = [], []
    for method, run in runs().items():
        for t in trials(method, run):
            if method == "band" and t["character"] == "unsteered":
                continue
            seg = residuals(t)
            for x in (seg["prompt"][0][1:], seg["generated"][0]):
                x = unit(x)
                a = x @ mu
                r = rng.standard_normal(x.shape)
                r = unit(r - np.outer(r @ mu, mu))
                w = a[:, None] * mu + np.sqrt(np.clip(1 - a**2, 0, None))[:, None] * r
                real.append((x @ space.unit.T).max(1))
                null.append((w @ space.unit.T).max(1))
    real, null = np.concatenate(real), np.concatenate(null)
    fig, ax = plt.subplots(figsize=(COLUMN, 1.9))
    bins = np.linspace(0, 1, 51)
    ax.hist(null, bins=bins, color=GRID, label="null: same alignment with mean role")
    ax.hist(real, bins=bins, histtype="step", color=INK, linewidth=1, label="real residuals")
    ax.axvline(NEAR, color=MUTED, linestyle="--", linewidth=0.6)
    ax.set_xlabel("max cosine to any of 275 roles")
    ax.set_ylabel("tokens")
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=6.5)
    fig.savefig(OUT / "maxcos_null.pdf")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    role_map = RoleMap.load(TSNE / "roles/role-map.pkl")
    # Display name only: the medoid-based "fool, hoarder, adolescent" undersells the
    # cluster that holds infant and toddler, which our experiments revolve around.
    for cluster, name in role_map.cluster_names.items():
        if name.startswith("fool"):
            role_map.cluster_names[cluster] = "infant, fool, adolescent"
    # Label the same three roles on the map: swap the medoid hoarder for infant.
    medoid_labels = tsne_plots.labeled_indices

    def display_labels(vectors, labels, clusters, metric="cosine"):
        swap = {labels.index("hoarder"): labels.index("infant")}
        return sorted(swap.get(i, i) for i in medoid_labels(vectors, labels, clusters, metric))

    tsne_plots.labeled_indices = display_labels
    figures = {
        "roles_map": lambda: roles_map_figure(role_map),
        "trajectory": lambda: trajectory_figure(role_map, tone_runs(role_map)),
        "trajectories": lambda: trajectories_figure(role_map),
        "ribbon_compact": lambda: ribbon_figure(role_map),
        "ribbon_settings": lambda: ribbon_settings_figures(role_map),
        "rank_vs_norm": lambda: rank_figure(role_map),
        "tone_control": tone_figure,
        "offrails_auc": offrails_figure,
        "maxcos_null": lambda: null_figure(load_space()),
        "cluster_ribbons": copy_cluster_ribbons,
    }
    parser = argparse.ArgumentParser(description="Render paper figures; none named = all.")
    parser.add_argument("names", nargs="*", metavar="figure", help=", ".join(figures))
    names = parser.parse_args().names or list(figures)
    unknown = sorted(set(names) - set(figures))
    if unknown:
        parser.error(f"unknown figure(s) {unknown}; choose from {', '.join(figures)}")
    for name in names:
        figures[name]()
        print("wrote", name)


def copy_cluster_ribbons():
    """Existing per-cluster ribbon PNGs, reused as they are (appendix)."""
    for png in sorted((TSNE / "roles-cluster-ribbons").glob("cluster-*-ribbons.png")):
        shutil.copy(png, OUT / png.name)


if __name__ == "__main__":
    main()
