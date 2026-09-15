"""Colour every token by the Ward cluster its high-dimensional neighbours support.

The 2D projection of a whole experiment is too crowded to read: 103 trials x 47
tokens of arrows over a 275-role scatter. This script skips the 2D positions and
asks a discrete question per residual instead — *is it actually close to any role
vector, and if so whose cluster?* — then draws one trial as one horizontal ribbon
of token colours. Trials whose steered character shares a Ward cluster are
stacked into a single figure, so one plot covers one cluster's steering sweep
across every run, injection layer and coefficient.

A cell is coloured only when at least one of the token's ten nearest role vectors
has cosine similarity above ``--min-similarity`` (0.5 by default); otherwise it
is white, because the nearest role is not near in any useful sense. Among the
neighbours that do clear the bar, the cluster holding most of them wins the cell,
and the tint is the strongest similarity inside that cluster. So the ribbon shows
evidence of proximity, not merely the least distant of six remote options.

Neighbours are cosine, taken in the original hidden space against the role
vectors stored in the map, never against the embedding. Cluster colours come from
``tsne_lens.palette``, which pins them to cluster membership, so a cluster is the
same colour here as on the roles map and across refits.

Example:

    uv run --extra plots python src/tsne-lens/scripts/cluster_ribbons.py \
      --map outputs/plots/tsne/roles/role-map.pkl \
      --run alpha=runtime/outputs/experiments/17roles-tylenol/tylenol-8000mg-roles-alpha/runs/0001_... \
      --run band=runtime/outputs/experiments/17roles-tylenol/tylenol-8000mg-roles-band/runs/0001_... \
      --prompt-only --output outputs/plots/tsne/roles-cluster-ribbons
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from collections import Counter
from html import escape
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tsne_lens.inputs import load_trajectory  # noqa: E402
from tsne_lens.palette import map_colors  # noqa: E402
from tsne_lens.projection import RoleMap  # noqa: E402

# Reserved status colours; never a cluster colour.
OUTCOME_STYLE = {
    2: ("#C3271E", "harmful (outcome 2)"),
    1: ("#B26B00", "ignorant (outcome 1)"),
}
SURFACE = "#FCFCFB"
BLANK = "#FFFFFF"
INK = "#0B0B0B"
INK_SOFT = "#52514E"
INK_MUTED = "#8A8983"
GRID = "#DCDBD5"

# A token clears the bar at MIN_SIMILARITY and saturates at TINT_CEILING; the top
# of the scale is fixed rather than per-figure so two ribbons are comparable.
MIN_SIMILARITY = 0.5
TINT_CEILING = 0.9
MIN_ALPHA, MAX_ALPHA = 0.28, 1.0


def parse_labelled_path(value: str) -> tuple[str, Path]:
    """Accept ``label=path`` or a bare path whose parent directory names it."""
    label, separator, raw = value.partition("=")
    if not separator:
        path = Path(value)
        # runs/<run-id> -> the sweep directory two levels up is the useful name.
        return (path.parent.parent.name or path.name), path
    if not label.strip() or not raw.strip():
        raise argparse.ArgumentTypeError(f"expected label=path, got {value!r}")
    return label.strip(), Path(raw.strip())


def display_token(token: str) -> str:
    """Render a byte-level BPE token as something readable on an axis."""
    if not token:
        return "?"
    if token.startswith("<|") and token.endswith("|>"):
        return token[2:-2]
    text = token.replace("Ċ", "⏎").replace("Ġ", "")
    return text or "␣" * token.count("Ġ") or token


def tint_for(similarity: float, threshold: float = MIN_SIMILARITY) -> float:
    """Opacity for a cell that cleared the bar, from the bar up to the ceiling."""
    span = (similarity - threshold) / max(TINT_CEILING - threshold, 1e-9)
    return MIN_ALPHA + (MAX_ALPHA - MIN_ALPHA) * float(np.clip(span, 0.0, 1.0))


def read_outcomes(path: Path) -> tuple[dict[str, int], dict]:
    if not path.is_file():
        return {}, {}
    payload = json.loads(path.read_text())
    outcomes = {str(k): int(v) for k, v in payload.get("outcomes", {}).items()}
    return outcomes, payload.get("rubric", {})


def neighbour_table(vectors: np.ndarray, role_map: RoleMap, k: int):
    """Top-k cosine neighbours per residual: their clusters and similarities."""
    from sklearn.metrics import pairwise_distances

    distances = pairwise_distances(vectors, role_map.vectors, metric="cosine")
    k = min(k, distances.shape[1])
    nearest = np.argsort(distances, axis=1)[:, :k]
    clusters, similarities, neighbours = [], [], []
    for row, indices in zip(distances, nearest):
        clusters.append([int(role_map.clusters[j]) for j in indices])
        similarities.append([round(float(1.0 - row[j]), 5) for j in indices])
        neighbours.append(
            [
                {
                    "label": role_map.labels[j],
                    "cluster": int(role_map.clusters[j]),
                    "similarity": float(1.0 - row[j]),
                }
                for j in indices
            ]
        )
    return clusters, similarities, neighbours


def cell_for(clusters, similarities, threshold: float):
    """One ribbon cell: the cluster the evidence supports, and how strong it is.

    Only neighbours above the threshold count. A token with none of them stays
    white — nothing in the map is genuinely close to it — and keeps its best
    similarity for the record. Otherwise the cluster holding most of the
    qualifying neighbours wins (a tie goes to whichever owns the closest one) and
    the cell is tinted by the strongest similarity inside that cluster.
    """
    qualifying = [(c, s) for c, s in zip(clusters, similarities) if s > threshold]
    if not qualifying:
        return None, float(max(similarities, default=0.0)), 0.0
    counts = Counter(cluster for cluster, _ in qualifying)
    top = max(counts.values())
    winner = next(cluster for cluster, _ in qualifying if counts[cluster] == top)
    similarity = max(s for cluster, s in qualifying if cluster == winner)
    return winner, float(similarity), counts[winner] / len(clusters)


def assign_cells(record: dict, threshold: float) -> None:
    """Apply the plotting rule to one record's stored neighbour table, in place."""
    if "neighbor_clusters" not in record or "neighbor_similarities" not in record:
        raise ValueError(
            "record has no neighbour table; regenerate ribbons.json with this "
            "version of the script before replotting with --from-json"
        )
    cells = [
        cell_for(clusters, similarities, threshold)
        for clusters, similarities in zip(
            record["neighbor_clusters"], record["neighbor_similarities"]
        )
    ]
    record["clusters"] = [cluster for cluster, _, _ in cells]
    record["similarities"] = [similarity for _, similarity, _ in cells]
    record["purities"] = [purity for _, _, purity in cells]


def collect_run(label: str, run: Path, role_map: RoleMap, args) -> list[dict]:
    manifest = json.loads((run / "manifest.json").read_text())
    trials = sorted(manifest.get("trials", {}).values(), key=lambda t: t["index"])
    complete = [t for t in trials if t["status"] == "complete"]
    if not complete:
        raise ValueError(f"no completed trials in {run}")
    outcomes, rubric = read_outcomes(
        Path(args.outcomes[label]) if label in args.outcomes else run / "judging" / "outcomes.json"
    )
    records = []
    for trial in complete:
        source = (run / trial["path"]).resolve()
        if not source.is_relative_to(run.resolve()):
            raise ValueError("trial path escapes the run directory")
        if not (source / "activations" / "manifest.json").is_file():
            raise ValueError(
                f"missing activations in {source}; results.zip does not contain tensors"
            )
        vectors, rows = load_trajectory(
            source,
            layer=role_map.config["layer"],
            model_id=role_map.config["model_id"],
            point=args.point,
            batch_index=args.batch_index,
            generated_only=args.generated_only,
            prompt_only=args.prompt_only,
        )
        clusters, similarities, neighbours = neighbour_table(
            vectors, role_map, args.neighbors
        )
        trial_hash = source.name.split("_", 1)[-1]
        steering = trial.get("steering") or {}
        records.append(
            dict(
                run=label,
                run_path=str(run.resolve()),
                trial=source.name,
                trial_hash=trial_hash,
                character=str(trial["character"]),
                steering=steering or None,
                method=steering.get("method"),
                coefficient=steering.get("coefficient"),
                layers=steering.get("layers"),
                outcome=outcomes.get(trial_hash),
                rubric=rubric,
                tokens=[r["token"] for r in rows],
                positions=[r["position"] for r in rows],
                # The neighbour table, not the cells: it is what lets --from-json
                # redraw at a different --min-similarity without the tensors.
                neighbor_clusters=clusters,
                neighbor_similarities=similarities,
                neighbors=neighbours if args.store_neighbors else None,
            )
        )
        print(f"  {label}/{source.name}: {len(rows)} tokens", flush=True)
    return records


# What the trial's coefficient means, so a row label never implies "strength"
# for a sweep whose knob is a band half-width.
COEFFICIENT_SYMBOL = {"additive": "α", "character_band": "w"}


def row_label(record: dict) -> str:
    if record["steering"] is None:
        return f"{record['run']} · no steering"
    layers = "+".join(str(x) for x in (record["layers"] or []))
    symbol = COEFFICIENT_SYMBOL.get(record.get("method"), "c")
    return f"{record['run']} · L{layers} · {symbol}={record['coefficient']:g}"


def sort_key(record: dict, run_order: list[str]) -> tuple:
    return (
        record["character"],
        run_order.index(record["run"]),
        tuple(record["layers"] or []),
        record["coefficient"] if record["coefficient"] is not None else -1.0,
    )


def label_ink(color: str, alpha: float) -> str:
    """Ink for the cluster digit printed on a cell, by the tint's luminance."""
    import matplotlib

    blended = [
        channel * alpha + surface * (1 - alpha)
        for channel, surface in zip(
            matplotlib.colors.to_rgb(color), matplotlib.colors.to_rgb(SURFACE)
        )
    ]
    luminance = sum(w * c for w, c in zip((0.2126, 0.7152, 0.0722), blended))
    return "#FFFFFF" if luminance < 0.5 else INK


def draw_cluster(cluster: int, records: list[dict], baselines: list[dict], role_map,
                 colors: dict[int, str], args, output: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PatchCollection
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch, Rectangle

    rows = baselines + records
    tokens = rows[0]["tokens"]
    n_tokens, n_rows = len(tokens), len(rows)

    cell_w, cell_h = 0.235, 0.235
    left_margin, right_margin = 3.05, 2.05
    top_margin, bottom_margin = 2.05, 3.15
    legend_y, key_y = 1.12, 0.32
    fig_w = left_margin + n_tokens * cell_w + right_margin
    fig_h = top_margin + n_rows * cell_h + bottom_margin
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor=SURFACE)
    ax = fig.add_axes(
        [
            left_margin / fig_w,
            bottom_margin / fig_h,
            (n_tokens * cell_w) / fig_w,
            (n_rows * cell_h) / fig_h,
        ]
    )
    ax.set_facecolor(SURFACE)
    ax.set_xlim(0, n_tokens)
    ax.set_ylim(n_rows, 0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)

    patches, rgba, blanks, labels = [], [], [], []
    gap = 0.11
    for i, record in enumerate(rows):
        start = 0
        for j in range(n_tokens + 1):
            if j < n_tokens and record["clusters"][j] == record["clusters"][start]:
                continue
            # [start, j) is one run of a single cell colour: gap only at its ends.
            owner = record["clusters"][start]
            for t in range(start, j):
                x0 = t + (gap if t == start else 0.0)
                x1 = t + 1 - (gap if t == j - 1 else 0.0)
                patch = Rectangle((x0, i + 0.09), x1 - x0, 0.82)
                if owner is None:
                    blanks.append(patch)
                    continue
                patches.append(patch)
                rgba.append(
                    matplotlib.colors.to_rgba(
                        colors[owner], tint_for(record["similarities"][t], args.min_similarity)
                    )
                )
            if owner is not None and j - start >= 2:
                middle = (start + j) // 2
                alpha = tint_for(record["similarities"][middle], args.min_similarity)
                labels.append(
                    ((start + j) / 2, i + 0.5, str(owner), label_ink(colors[owner], alpha))
                )
            start = j
    # Blank cells keep the ribbon continuous where the evidence runs out.
    ax.add_collection(
        PatchCollection(blanks, facecolor=BLANK, edgecolor=GRID, linewidth=0.4)
    )
    collection = PatchCollection(patches, match_original=False, linewidth=0)
    collection.set_facecolor(rgba)
    ax.add_collection(collection)
    for x, y, text, ink in labels:
        ax.text(x, y, text, ha="center", va="center", fontsize=6.5,
                color=ink, alpha=0.92, zorder=3)

    # Chat-template boundaries, kept recessive.
    for j, token in enumerate(tokens):
        if token in ("<|im_start|>", "<|im_end|>", "<think>", "</think>"):
            ax.axvline(j, color=GRID, linewidth=0.8, zorder=0)

    ax.set_xticks(np.arange(n_tokens) + 0.5)
    ax.set_xticklabels([display_token(t) for t in tokens], rotation=90,
                       fontsize=6.8, color=INK_SOFT)
    ax.set_yticks([])
    ax.set_xlabel(
        f"{'prompt' if args.prompt_only else 'generated' if args.generated_only else 'all'}"
        " tokens, in order",
        fontsize=8.5, color=INK_SOFT, labelpad=6,
    )

    # Left gutter: group label per character, detail per row. Right gutter: how
    # much of the row is coloured at all, and how much of it is the steered
    # character's own cluster.
    x_detail = -0.25
    x_char = x_detail - 1.70 / cell_w
    x_mark = n_tokens + 0.6
    x_share = n_tokens + 1.2 + 0.55 / cell_w
    x_lit = x_share + 0.62 / cell_w
    groups = {}
    for i, record in enumerate(rows):
        groups.setdefault(record["character"], []).append(i)
        ax.text(x_detail, i + 0.5, row_label(record), ha="right", va="center",
                fontsize=7.2, color=INK_SOFT, family="DejaVu Sans Mono", clip_on=False)
        style = OUTCOME_STYLE.get(record["outcome"])
        if style:
            ax.plot([x_mark], [i + 0.5], marker="o", markersize=6.5,
                    color=style[0], markeredgecolor=SURFACE, markeredgewidth=0.9,
                    clip_on=False, zorder=5)
        own = role_map.label_cluster.get(record["character"])
        if own is not None:
            share = float(np.mean([c == own for c in record["clusters"]]))
            ax.text(x_share, i + 0.5, f"{share:.0%}", ha="right", va="center",
                    fontsize=7.0, color=INK_MUTED, family="DejaVu Sans Mono",
                    clip_on=False)
        lit = float(np.mean([c is not None for c in record["clusters"]]))
        ax.text(x_lit, i + 0.5, f"{lit:.0%}", ha="right", va="center",
                fontsize=7.0, color=INK_MUTED, family="DejaVu Sans Mono",
                clip_on=False)
    for character, indices in groups.items():
        ax.text(x_char, (min(indices) + max(indices)) / 2 + 0.5, character,
                ha="right", va="center", fontsize=9.8, color=INK, fontweight="bold",
                clip_on=False)
        if min(indices) > 0:
            ax.plot([x_char + 0.3, x_lit], [min(indices)] * 2, color=GRID,
                    linewidth=1.0, clip_on=False, zorder=0)
    ax.text(x_share, -0.5, "own", ha="right", va="bottom", fontsize=7.2,
            color=INK_MUTED, clip_on=False)
    ax.text(x_lit, -0.5, "near", ha="right", va="bottom", fontsize=7.2,
            color=INK_MUTED, clip_on=False)
    ax.text(x_mark, -0.5, "judge", ha="center", va="bottom", fontsize=7.2,
            color=INK_MUTED, clip_on=False)

    name = role_map.cluster_names[cluster]
    fig.text(left_margin / fig_w, 1 - 0.55 / fig_h,
             f"Neighbourhood clusters per token — steering cluster {cluster}: {name}",
             fontsize=15, color=INK, fontweight="bold", va="top")
    segment = "prompt" if args.prompt_only else "generated" if args.generated_only else "all"
    subtitle = [
        f"Each cell is one residual stream. It is white unless at least one of its top-{args.neighbors} "
        f"cosine neighbours among the {len(role_map.labels)} role vectors clears "
        f"{args.min_similarity:.0%} similarity in the original {role_map.vectors.shape[1]}-d space — "
        "white means no role in the map is close to that residual at all, whichever is least far.",
        f"A coloured cell takes the Ward-{role_map.config['n_clusters']} cluster holding most of those "
        "qualifying neighbours, tinted by the strongest similarity inside it; cluster colours are fixed "
        "by membership, so they match the roles map.",
        f"Layer {role_map.config['layer']}, {args.point.replace('_', '-')}, {segment} tokens. "
        f"One row per trial, grouped by the steered character; every character here belongs to cluster {cluster}.",
    ]
    wrapped = []
    for line in subtitle:
        wrapped.extend(textwrap.wrap(line, width=int((fig_w - left_margin - 0.4) / 0.0635)))
    for n, line in enumerate(wrapped):
        fig.text(left_margin / fig_w, 1 - (1.00 + 0.21 * n) / fig_h, line,
                 fontsize=9.0, color=INK_SOFT, va="top")

    cluster_handles = [
        Patch(facecolor=colors[c], edgecolor="none",
              label=f"{c} · {role_map.cluster_names[c]}")
        for c in sorted(role_map.cluster_names)
    ] + [
        Patch(facecolor=BLANK, edgecolor=GRID, linewidth=0.8,
              label=f"no neighbour above {args.min_similarity:.0%}")
    ]
    legend = fig.legend(
        handles=cluster_handles, loc="lower left", ncol=3, frameon=False,
        fontsize=8.6, handlelength=1.6, handleheight=1.0, columnspacing=2.2,
        title="Ward cluster supported by the token's qualifying neighbours   (the digit printed on each run of two or more cells repeats it)",
        title_fontsize=9.0,
        bbox_to_anchor=(left_margin / fig_w, legend_y / fig_h),
    )
    legend.get_title().set_color(INK)
    legend._legend_box.align = "left"
    for text in legend.get_texts():
        text.set_color(INK_SOFT)

    # Tint key: the same hue at four similarities, against the white cell.
    key_x = left_margin / fig_w
    fig.text(key_x, (key_y + 0.30) / fig_h,
             "Tint = strongest cosine similarity to a role in that cluster",
             fontsize=9.0, color=INK, va="bottom")
    steps = [None, args.min_similarity, 0.6, 0.7, 0.8, 0.9]
    for n, similarity in enumerate(steps):
        fig.add_artist(
            Rectangle((key_x + n * 0.62 / fig_w, key_y / fig_h), 0.34 / fig_w, 0.18 / fig_h,
                      transform=fig.transFigure,
                      facecolor=BLANK if similarity is None else colors[cluster],
                      alpha=1.0 if similarity is None else tint_for(similarity, args.min_similarity),
                      edgecolor=GRID if similarity is None else "none",
                      linewidth=0.6)
        )
        fig.text(key_x + (n * 0.62 + 0.39) / fig_w, (key_y + 0.09) / fig_h,
                 "—" if similarity is None else f"{similarity:.0%}",
                 fontsize=8.4, color=INK_SOFT, va="center")

    marker_x = key_x + 4.9 / fig_w
    fig.text(marker_x, (key_y + 0.30) / fig_h, "Judge outcome (right gutter)",
             fontsize=9.0, color=INK, va="bottom")
    present = [c for c in sorted(OUTCOME_STYLE, reverse=True)
               if any(r["outcome"] == c for r in rows)]
    for n, code in enumerate(present or [2]):
        color, text = OUTCOME_STYLE[code]
        fig.add_artist(
            Line2D([marker_x + 0.09 / fig_w], [(key_y + 0.09 - 0.26 * n) / fig_h],
                   transform=fig.transFigure, marker="o", linestyle="none",
                   markersize=6.5, color=color, markeredgecolor=SURFACE,
                   markeredgewidth=0.9)
        )
        label = text if present else f"{text} — none in this cluster"
        fig.text(marker_x + 0.30 / fig_w, (key_y + 0.09 - 0.26 * n) / fig_h, label,
                 fontsize=8.4, color=INK_SOFT, va="center")
    fig.text(marker_x + 3.4 / fig_w, (key_y + 0.30) / fig_h,
             "own / near (right gutter)", fontsize=9.0, color=INK, va="bottom")
    fig.text(marker_x + 3.4 / fig_w, (key_y + 0.09) / fig_h,
             "share of this row's tokens coloured with the steered character's\n"
             f"own cluster, and share coloured at all (a neighbour above {args.min_similarity:.0%})",
             fontsize=8.4, color=INK_SOFT, va="center", linespacing=1.6)

    destination = output / f"cluster-{cluster}-ribbons.png"
    fig.savefig(destination, dpi=args.dpi, facecolor=SURFACE)
    fig.savefig(destination.with_suffix(".svg"), facecolor=SURFACE)
    plt.close(fig)
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--map", type=Path, required=True, help="role-map.pkl from tsne-lens")
    parser.add_argument("--run", type=parse_labelled_path, action="append", required=True,
                        metavar="LABEL=DIR", help="experiment run directory; repeatable")
    parser.add_argument("--outcomes", type=parse_labelled_path, action="append", default=[],
                        metavar="LABEL=FILE",
                        help="judging outcomes.json override for a run "
                             "(default: <run>/judging/outcomes.json)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--neighbors", type=int, default=10)
    parser.add_argument("--min-similarity", type=float, default=MIN_SIMILARITY,
                        help="cosine similarity a neighbour must clear for its cluster "
                             "to colour the cell; below it the cell is white "
                             f"(default: {MIN_SIMILARITY})")
    parser.add_argument("--point", default="post_intervention",
                        choices=("post_intervention", "pre_intervention"))
    parser.add_argument("--batch-index", type=int, default=0)
    parser.add_argument("--prompt-only", action="store_true")
    parser.add_argument("--generated-only", action="store_true")
    parser.add_argument("--store-neighbors", action="store_true",
                        help="keep the labelled neighbour lists in ribbons.json")
    parser.add_argument("--from-json", type=Path,
                        help="replot from a previous ribbons.json instead of reading tensors")
    parser.add_argument("--dpi", type=int, default=170)
    args = parser.parse_args(argv)
    if args.neighbors < 1:
        parser.error("--neighbors must be positive")
    if not -1.0 < args.min_similarity < 1.0:
        parser.error("--min-similarity must lie in (-1, 1)")
    if args.prompt_only and args.generated_only:
        parser.error("select only one of --prompt-only and --generated-only")
    args.outcomes = dict(args.outcomes)
    run_order = [label for label, _ in args.run]
    if len(set(run_order)) != len(run_order):
        parser.error("run labels must be unique")
    unknown = set(args.outcomes) - set(run_order)
    if unknown:
        parser.error(f"--outcomes labels without a matching --run: {sorted(unknown)}")

    role_map = RoleMap.load(args.map)
    role_map.label_cluster = {
        label: int(cluster) for label, cluster in zip(role_map.labels, role_map.clusters)
    }
    # One colour per cluster, fixed by membership rather than by Ward's numbering.
    colors = map_colors(role_map)
    args.output.mkdir(parents=True, exist_ok=True)

    if args.from_json:
        payload = json.loads(args.from_json.read_text())
        records = payload["records"]
        # The neighbour table is the record; --min-similarity stays free to change.
        for key in ("neighbors", "point", "prompt_only", "generated_only"):
            if key in payload.get("settings", {}):
                setattr(args, key, payload["settings"][key])
    else:
        records = []
        for label, run in args.run:
            print(f"Reading {label} ← {run}", flush=True)
            records.extend(collect_run(label, run, role_map, args))
    for record in records:
        assign_cells(record, args.min_similarity)

    token_sets = {tuple(r["tokens"]) for r in records}
    if len(token_sets) != 1:
        raise ValueError("trials do not share one token sequence; ribbons would not align")

    baselines = [r for r in records if r["steering"] is None]
    steered = [r for r in records if r["steering"] is not None]
    unknown_characters = sorted(
        {r["character"] for r in steered} - set(role_map.label_cluster)
    )
    if unknown_characters:
        raise ValueError(f"steered characters absent from the role map: {unknown_characters}")

    by_cluster: dict[int, list[dict]] = {}
    for record in steered:
        by_cluster.setdefault(role_map.label_cluster[record["character"]], []).append(record)

    figures = []
    for cluster in sorted(by_cluster):
        group = sorted(by_cluster[cluster], key=lambda r: sort_key(r, run_order))
        destination = draw_cluster(cluster, group, baselines, role_map, colors, args,
                                   args.output)
        characters = sorted({r["character"] for r in group})
        figures.append(
            dict(cluster=cluster, name=role_map.cluster_names[cluster],
                 color=colors[cluster], characters=characters,
                 rows=len(group) + len(baselines),
                 harmful=sum(1 for r in group if r["outcome"] == 2),
                 near=round(float(np.mean([c is not None for r in group
                                           for c in r["clusters"]])), 4),
                 file=destination.name)
        )
        print(f"Wrote {destination}", flush=True)

    settings = dict(neighbors=args.neighbors, min_similarity=args.min_similarity,
                    point=args.point,
                    prompt_only=args.prompt_only, generated_only=args.generated_only,
                    layer=role_map.config["layer"], model_id=role_map.config["model_id"],
                    metric="cosine", map=str(args.map.resolve()),
                    cluster_names={str(k): v for k, v in role_map.cluster_names.items()},
                    cluster_colors={str(k): v for k, v in colors.items()})
    (args.output / "ribbons.json").write_text(
        json.dumps(dict(settings=settings, figures=figures, records=records), indent=1) + "\n"
    )

    items = "\n".join(
        f'<li><a href="{escape(f["file"])}">cluster {f["cluster"]} — {escape(f["name"])}</a>'
        f' · {escape(", ".join(f["characters"]))} · {f["rows"]} rows'
        f' · {f["harmful"]} harmful · {f["near"]:.0%} of cells coloured</li>'
        for f in figures
    )
    (args.output / "index.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        "<title>Neighbourhood-cluster ribbons</title>"
        '<body style="font:14px/1.5 system-ui;margin:2rem;max-width:60rem">'
        "<h1>Neighbourhood-cluster ribbons</h1>"
        f"<p>Top-{args.neighbors} cosine neighbours in the layer "
        f"{role_map.config['layer']} role space, {escape(args.point)}, "
        f"{'prompt' if args.prompt_only else 'generated' if args.generated_only else 'all'}"
        f" tokens. A cell is coloured only where a neighbour clears "
        f"{args.min_similarity:.0%} similarity; otherwise it is white. "
        f"Runs: {escape(', '.join(run_order))}.</p>"
        f"<ul>{items}</ul>"
        '<p><a href="ribbons.json">ribbons.json</a> — the top-k neighbour table, '
        "plus the cell each token was given, for every trial.</p>"
        "</body></html>",
        encoding="utf-8",
    )
    print(f"Index: {args.output / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
