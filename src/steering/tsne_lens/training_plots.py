"""Checkpoint PNGs and a GIF with fixed role highlights and a shared axis range."""

import json
from pathlib import Path

import numpy as np
from openTSNE import TSNEEmbedding

CORE_COLORS = {
    "echo": "#d62728",
    "rogue": "#8b5cf6",
    "influencer": "#ed9500",
    "assistant": "#009e73",
    "default_vector": "#111111",
}


def default_positions(embedding, frames, vector):
    """Project a held-out reference into each saved layout, never modifying roles."""
    if vector is None:
        return None
    from openTSNE import affinity

    params = (
        {"perplexities": [5]}
        if isinstance(embedding.affinities, affinity.Multiscale)
        else {"perplexity": 5}
    )
    result = []
    for xy in frames:
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
            250, learning_rate=0.1, exaggeration=1.5, max_grad_norm=0.25, inplace=True
        )
        result.append(np.asarray(partial)[0].copy())
    return np.asarray(result)


def export_training(
    embedding, labels, history, frames, output, *, clusters=None, default_vector=None
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
    default_xy = default_positions(embedding, frames, default_vector)
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
    paths = []
    fig, ax = plt.subplots(figsize=(10, 8), dpi=105)
    for index, (xy, record) in enumerate(zip(frames, history)):
        ax.clear()
        ax.scatter(xy[:, 0], xy[:, 1], s=19, c=colors, alpha=0.65, linewidths=0)
        for name, color in CORE_COLORS.items():
            if name == "default_vector":
                if default_xy is None:
                    continue
                position = default_xy[index]
            elif name in labels:
                position = xy[labels.index(name)]
            else:
                continue
            ax.scatter(
                *position,
                s=105,
                c=color,
                edgecolors="white",
                linewidths=1.2,
                zorder=5,
                marker="*" if name == "default_vector" else "o",
                label=name,
            )
            offset = (8, -18) if name == "default_vector" else (8, 10)
            ax.annotate(
                name,
                position,
                xytext=offset,
                textcoords="offset points",
                color=color,
                fontsize=10,
                zorder=6,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1),
            )
        ps = "/".join(f"{p:g}" for p in record["perplexities"])
        ax.set_title(
            f"Iteration {record['iteration']} · {record['phase']} · perplexity {ps} · exaggeration {record['exaggeration']:g}\n"
            f"Ordinary KL (current affinity): {record['kl']:.7f}"
        )
        ax.set(
            xlim=(-extent, extent),
            ylim=(-extent, extent),
            xlabel="t-SNE 1",
            ylabel="t-SNE 2",
        )
        ax.set_aspect("equal")
        ax.grid(alpha=0.15)
        ax.legend(loc="upper left", fontsize=8)
        fig.texts.clear()
        fig.text(
            0.5,
            0.02,
            "Fixed axes across frames; default_vector is held out and projected. KL resets at affinity changes.",
            ha="center",
            fontsize=8,
        )
        fig.tight_layout(rect=(0, 0.03, 1, 1))
        path = output / f"frame-{index:03d}-iter-{record['iteration']:05d}.png"
        fig.savefig(path)
        paths.append(path)
    plt.close(fig)
    images = []
    for path in paths:
        with Image.open(path) as im:
            images.append(im.convert("RGB").copy())
    images[0].save(
        output / "training.gif",
        save_all=True,
        append_images=images[1:],
        loop=0,
        duration=[1000] + [300] * max(0, len(images) - 2) + [1800],
    )
    for im in images:
        im.close()
    # The true tiny-scale initializer is shown separately with its own axes.
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(frames[0, :, 0], frames[0, :, 1], s=16, c=colors)
    for name, color in CORE_COLORS.items():
        if name in labels or (name == "default_vector" and default_xy is not None):
            p = (
                default_xy[0]
                if name == "default_vector"
                else frames[0, labels.index(name)]
            )
            ax.scatter(*p, c=color, s=60)
            ax.annotate(name, p, fontsize=9, color=color)
    ax.set_title(f"Actual PCA initialization · KL {history[0]['kl']:.7f} · zoomed axes")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(output / "pca-initialization.png")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 4))
    # Never draw a KL improvement line across an affinity change.
    start = 0
    for i in range(1, len(history) + 1):
        if (
            i == len(history)
            or history[i]["perplexities"] != history[start]["perplexities"]
        ):
            section = history[start:i]
            ax.plot(
                [r["iteration"] for r in section],
                [r["kl"] for r in section],
                ".-",
                label="P=" + str(section[0]["perplexities"]),
            )
            start = i
    ax.set(
        xlabel="Optimization iterations (not epochs)",
        ylabel="Ordinary KL for current affinity",
        title="Convergence within each affinity stage",
    )
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / "kl-history.png")
    plt.close(fig)
