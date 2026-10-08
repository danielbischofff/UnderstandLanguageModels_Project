"""Paper figures for the 2D role-space lens, written to writeup/figures/tsne/.

Map figures reuse tsne_lens.plots.make_figure on the saved map
(outputs/plots/tsne/roles/role-map.pkl), so palette, labels and the default
star match roles.png; only size, fonts and legend placement change for print.
The other figures are matplotlib, read from src/control/results and the ribbon
JSON, with the same fixed cluster colours.

    uv run --extra plots python src/control/figures.py
"""

from __future__ import annotations

import csv
import json
import shutil
import sys

import matplotlib

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


def map_figures(role_map):
    print_layout(make_figure(role_map, [])).write_image(OUT / "roles_map.pdf")
    picks = [("unsteered", trial_dir("alpha", "unsteered")),
             ("demon · alpha 4 · L32 (harmful)", trial_dir("alpha", "demon", 32, 4.0)),
             ("toddler · band 0.1 · L32 (harmful)", trial_dir("band", "toddler", 32, 0.1))]
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
        sc = ax.scatter(x[1:], y[1:], c=np.arange(1, len(x)), cmap="viridis", s=9, marker="s",
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
    return trajectories


def ribbon_figure(role_map):
    data = json.loads((TSNE / "roles-cluster-ribbons/ribbons.json").read_text())
    colors = map_colors(role_map)
    rows = [("alpha", "unsteered", None, None, "unsteered"),
            ("alpha", "demon", [29], 4.0, "demon · α=4 · L29"),
            ("alpha", "demon", [32], 4.0, "demon · α=4 · L32"),
            ("alpha", "toddler", [32], 4.0, "toddler · α=4 · L32"),
            ("band", "demon", [32], 0.1, "demon · β=0.1 · L32"),
            ("band", "toddler", [29], 0.1, "toddler · β=0.1 · L29"),
            ("band", "toddler", [32], 0.1, "toddler · β=0.1 · L32"),
            ("band", "toddler", [32], 0.25, "toddler · β=0.25 · L32"),
            ("band", "organizer", [32], 0.25, "organizer · β=0.25 · L32")]
    found = []
    for run, character, layers, coefficient, label in rows:
        rec = next(r for r in data["records"] if r["run"] == run and r["character"] == character
                   and (layers is None or (r["layers"] == layers and r["coefficient"] == coefficient)))
        found.append((label, rec))
    tokens = [display_token(t).replace("⏎", "\\n") for t in found[0][1]["tokens"]]
    fig, ax = plt.subplots(figsize=(PAGE, 0.24 * len(found) + 0.9))
    for y, (label, rec) in enumerate(found):
        for x, (cluster, sim) in enumerate(zip(rec["clusters"], rec["similarities"])):
            if x == 0:  # attention sink: shown, but not read
                face, alpha = "#d9d9d6", 1.0
            elif cluster is None:
                face, alpha = "white", 1.0
            else:
                face, alpha = colors[int(cluster)], tint_for(sim)
            ax.add_patch(plt.Rectangle((x + 0.06, y + 0.08), 0.88, 0.84, facecolor=face, alpha=alpha,
                                       edgecolor=GRID, linewidth=0.3))
        if rec["outcome"] == 2:
            ax.text(len(tokens) + 0.4, y + 0.5, "harmful", va="center", fontsize=6.5, color=INK)
    ax.set_xlim(0, len(tokens) + 4)
    ax.set_ylim(len(found), 0)
    ax.set_yticks(np.arange(len(found)) + 0.5, [label for label, _ in found])
    ax.set_xticks(np.arange(len(tokens)) + 0.5, tokens, rotation=90, fontsize=5.5)
    ax.tick_params(length=0)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[c]) for c in sorted(colors)]
    names = [f"{c} · {role_map.cluster_names[c].split(',')[0]}" for c in sorted(colors)]
    ax.legend(handles + [plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor=GRID)],
              names + ["no role > 0.5"], ncol=7, loc="lower center", bbox_to_anchor=(0.45, 1.0),
              frameon=False, handlelength=1, columnspacing=0.8)
    fig.savefig(OUT / "ribbon_compact.pdf")
    plt.close(fig)


def rank_figure(role_map):
    props = json.loads((RESULTS / "c5_target_attraction.json").read_text())["role_properties"]
    colors = map_colors(role_map)
    with (RESULTS / "c5_trials.csv").open() as handle:
        rows = [r for r in csv.DictReader(handle) if r["method"] != "unsteered"]
    fig, ax = plt.subplots(figsize=(COLUMN, 2.4))
    for method, marker in (("alpha", "o"), ("band", "^")):
        group = [r for r in rows if r["method"] == method]
        x = [props[r["character"]]["norm_rd"] for r in group]
        y = [float(r["prompt_rank_tgt"]) for r in group]
        c = [colors[props[r["character"]]["cluster"]] for r in group]
        ax.scatter(x, y, s=12, c=c, marker=marker, alpha=0.75, linewidths=0, label=method)
        bad = [i for i, r in enumerate(group) if r["judge"] == "2"]
        ax.scatter([x[i] for i in bad], [y[i] for i in bad], s=46, marker=marker,
                   facecolors="none", edgecolors=INK, linewidths=0.8)
    ax.set_yscale("log")
    ax.invert_yaxis()
    ax.set_xlabel("‖R − D‖, length of the steering vector")
    ax.set_ylabel("target rank among 275 roles\n(median over prompt tokens, 1 = nearest)")
    ax.grid(axis="y", color=GRID, linewidth=0.4)
    handles = [plt.Line2D([], [], marker=m, linestyle="", color=MUTED, label=l)
               for m, l in (("o", "alpha"), ("^", "band"))]
    handles.append(plt.Line2D([], [], marker="o", linestyle="", markerfacecolor="none", markeredgecolor=INK,
                              label="harmful"))
    ax.legend(handles=handles, frameon=False, loc="lower right")
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
    signatures = [("prompt_align_mu", -1, "prompt: alignment with mean role"),
                  ("prompt_own_excess", 1, "prompt: target-cluster share"),
                  ("prompt_axis", -1, "prompt: drift down Assistant Axis"),
                  ("prompt_near", -1, "prompt: tokens near any role"),
                  ("prompt_norm_ratio", 1, "prompt: residual norm"),
                  ("answer_axis", -1, "answer: drift down Assistant Axis")]
    fig, axes = plt.subplots(1, 2, figsize=(PAGE, 1.9), sharey=True)
    for ax, (code, title) in zip(axes, (("1", "in character (98 trials)"), ("2", "broken (146 trials)"))):
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
                       label="within role × setting" if i == 0 else None)
        ax.axvline(0.5, color=MUTED, linewidth=0.6, linestyle="--")
        ax.set_xlim(0.15, 0.95)
        ax.set_title(title, loc="left")
        ax.set_xlabel("AUC (0.5 = chance)")
        ax.set_yticks(range(len(signatures)), [s[2] for s in signatures])
        ax.invert_yaxis()
    axes[1].legend(frameon=False, loc="lower left", fontsize=6.5)
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
    space = load_space()
    for name, func in (("roles map + trajectories", lambda: map_figures(role_map)),
                       ("compact ribbon", lambda: ribbon_figure(role_map)),
                       ("rank vs ||R-D||", lambda: rank_figure(role_map)),
                       ("tone control", tone_figure),
                       ("off-rails AUC", offrails_figure),
                       ("max-cos null", lambda: null_figure(space))):
        func()
        print("wrote", name)
    # Existing per-cluster ribbons reused as they are (appendix).
    for png in sorted((TSNE / "roles-cluster-ribbons").glob("cluster-*-ribbons.png")):
        shutil.copy(png, OUT / png.name)
    print("copied cluster ribbons to", OUT)


if __name__ == "__main__":
    main()
