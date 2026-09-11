"""Checkpoint PNGs and a GIF with representative labels and adaptive axes."""

import json
from io import BytesIO
from pathlib import Path

import numpy as np
from openTSNE import TSNEEmbedding

from .labels import PINNED_ROLES, label_offsets, labeled_indices


def default_positions(embedding, frames, vector, pca_position):
    """Use true PCA at frame zero, then independent fixed-layout projections."""
    if vector is None:
        return None
    from openTSNE import affinity

    params = (
        {"perplexities": [5]}
        if isinstance(embedding.affinities, affinity.Multiscale)
        else {"perplexity": 5}
    )
    if pca_position is None:
        raise ValueError(
            "the reference needs its role-fitted PCA position for frame zero"
        )
    result = [np.asarray(pca_position)]
    for xy in frames[1:]:
        reference = TSNEEmbedding(
            xy.copy(),
            embedding.affinities,
            negative_gradient_method="bh",
            theta=0,
            n_jobs=1,
        )
        partial = reference.prepare_partial(
            np.asarray(vector)[None],
            initialization="median",
            k=min(25, len(xy)),
            **params,
        )
        partial.optimize(
            250,
            learning_rate=0.1,
            exaggeration=1.5,
            momentum=0.8,
            max_grad_norm=0.25,
            inplace=True,
        )
        result.append(np.asarray(partial)[0].copy())
    return np.asarray(result)


def export_training(
    embedding,
    labels,
    history,
    frames,
    output,
    *,
    clusters=None,
    default_vector=None,
    default_pca=None,
    vectors=None,
    metric="cosine",
    adaptive_scale=True,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    np.savez_compressed(
        output / "checkpoints.npz", coordinates=frames, labels=np.asarray(labels)
    )
    default_xy = default_positions(embedding, frames, default_vector, default_pca)
    extent = float(np.max(np.abs(frames)))
    if default_xy is not None:
        extent = max(extent, float(np.max(np.abs(default_xy))))
        np.save(output / "default-vector-positions.npy", default_xy)
    extent = max(extent * 1.12, 1e-4)
    colors = (
        "#aab3bf"
        if clusters is None
        else np.asarray(
            ["#87aade", "#d6b681", "#93b6a1", "#cc9aba", "#9bc2c9", "#b6a3d2"]
        )[np.asarray(clusters) % 6]
    )
    selected = (
        labeled_indices(vectors, labels, clusters, metric)
        if vectors is not None and clusters is not None
        else [i for i, name in enumerate(labels) if name in PINNED_ROLES]
    )

    def annotate_roles(ax, xy, reference, span):
        positions = [xy[i] for i in selected]
        names = [labels[i] for i in selected]
        if reference is not None:
            positions.append(reference)
            names.append("default_vector")
            ax.scatter(
                *reference,
                s=105,
                c="#111111",
                marker="*",
                zorder=5,
                edgecolors="white",
                linewidths=0.8,
                label="default_vector",
            )
        offsets = (
            label_offsets(positions, names, span, width=640, height=640)
            if names
            else []
        )
        for position, name, offset in zip(positions, names, offsets):
            ax.annotate(
                name,
                position,
                xytext=offset * 0.72,
                textcoords="offset points",
                color="#333",
                fontsize=8,
                zorder=6,
                arrowprops=dict(arrowstyle="-", color="#888", lw=0.5),
                ha="center",
                va="center",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1),
            )

    images = []
    fig, ax = plt.subplots(figsize=(10, 8), dpi=105)
    for index, (xy, record) in enumerate(zip(frames, history)):
        ax.clear()
        points = xy if default_xy is None else np.vstack([xy, default_xy[index]])
        if adaptive_scale:
            low, high = points.min(axis=0), points.max(axis=0)
            center = (low + high) / 2
            radius = max(float((high - low).max()) * 0.62, 1e-12)
        else:
            center, radius = np.zeros(2), extent
        ax.scatter(xy[:, 0], xy[:, 1], s=19, c=colors, alpha=0.65, linewidths=0)
        annotate_roles(
            ax,
            xy,
            None if default_xy is None else default_xy[index],
            [2 * radius, 2 * radius],
        )
        ps = "/".join(f"{p:g}" for p in record["perplexities"])
        ax.set_title(
            f"Iteration {record['iteration']} · {record['phase']} · perplexity {ps} · exaggeration {record['exaggeration']:g}\n"
            f"Ordinary KL (current affinity): {record['kl']:.7f}"
        )
        ax.set(
            xlim=(center[0] - radius, center[0] + radius),
            ylim=(center[1] - radius, center[1] + radius),
            xlabel="t-SNE 1",
            ylabel="t-SNE 2",
        )
        ax.set_aspect("equal")
        ax.grid(alpha=0.15)
        if default_xy is not None:
            ax.legend(loc="upper left", fontsize=8)
        fig.texts.clear()
        fig.text(
            0.5,
            0.02,
            (
                "Adaptive axes per frame (equal x/y scale)"
                if adaptive_scale
                else "Fixed axes across frames"
            )
            + "; fixed affinities.\nDefault: true PCA at frame 0; independent insertion afterward.",
            ha="center",
            fontsize=8,
        )
        fig.tight_layout(rect=(0, 0.03, 1, 1))
        with BytesIO() as buffer:
            fig.savefig(buffer, format="png")
            buffer.seek(0)
            with Image.open(buffer) as im:
                images.append(im.convert("RGB").copy())
    plt.close(fig)
    images[0].save(
        output / "training.gif",
        save_all=True,
        append_images=images[1:],
        loop=0,
        duration=[1000] + [300] * max(0, len(images) - 2) + [1800],
    )
    for im in images:
        im.close()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(
        [r["iteration"] for r in history],
        [r["kl"] for r in history],
        ".-",
        label="P=" + str(history[0]["perplexities"]),
    )
    ax.set(
        xlabel="Optimization iterations (not epochs)",
        ylabel="Ordinary KL for current affinity",
        title="Convergence with fixed affinities",
    )
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / "kl-history.png")
    plt.close(fig)
