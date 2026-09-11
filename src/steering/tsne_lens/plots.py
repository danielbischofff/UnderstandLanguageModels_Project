"""Offline Plotly overview with a token cursor, plus optional static PNG."""

import json
from html import escape
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.colors import qualitative

from .labels import label_offsets, labeled_indices


def make_figure(role_map, trajectories):
    fig = go.Figure()
    roles = role_map.role_records()
    last_step = max((len(t["points"]) - 1 for t in trajectories), default=1)
    for cluster in sorted(set(role_map.clusters)):
        group = [r for r in roles if r["cluster"] == cluster]
        color = (
            "#9ca3af"
            if cluster == -1
            else qualitative.Dark24[int(cluster) % len(qualitative.Dark24)]
        )
        label = role_map.cluster_names[int(cluster)]
        legend_label = f"{int(cluster) + 1}. {label}" if cluster >= 0 else label
        fig.add_trace(
            go.Scatter(
                x=[r["x"] for r in group],
                y=[r["y"] for r in group],
                mode="markers",
                name=escape(legend_label),
                marker=dict(size=10, color=color, opacity=0.8),
                text=[escape(r["label"]) for r in group],
                hovertemplate="<b>%{text}</b><extra>%{fullData.name}</extra>",
            )
        )
        if cluster != -1:
            fig.add_annotation(
                x=float(np.median([r["x"] for r in group])),
                y=float(np.median([r["y"] for r in group])),
                text=str(int(cluster) + 1),
                showarrow=False,
                yshift=18,
                bgcolor="rgba(255,255,255,0.8)",
                font=dict(size=11),
            )
    cursors = []
    selected = labeled_indices(
        role_map.vectors, role_map.labels, role_map.clusters, role_map.config["metric"]
    )
    labeled = [roles[i] for i in selected]
    for point in role_map.config.get("reference_points", []):
        if point["label"] == "default_vector":
            labeled.append(point)
            fig.add_trace(
                go.Scatter(
                    x=[point["x"]],
                    y=[point["y"]],
                    mode="markers",
                    name="default_vector",
                    marker=dict(
                        color="#111111",
                        size=14,
                        symbol="star",
                        line=dict(color="white", width=1),
                    ),
                    hovertemplate="default_vector (held-out reference)<extra></extra>",
                )
            )
    if labeled:
        xy = np.asarray([[r["x"], r["y"]] for r in labeled])
        span = np.ptp(np.asarray(role_map.embedding), axis=0) * 1.15
        # Equal-scale axes: estimate the same pixel scale in both dimensions.
        scale = max(span[0] / 760, span[1] / 640)
        offsets = label_offsets(
            xy, [r["label"] for r in labeled], [scale * 760, scale * 640]
        )
        for point, (dx, dy) in zip(labeled, offsets):
            fig.add_annotation(
                x=point["x"],
                y=point["y"],
                text=escape(point["label"]),
                ax=float(dx),
                ay=float(-dy),
                xanchor="center",
                yanchor="middle",
                showarrow=True,
                arrowhead=0,
                arrowwidth=0.6,
                arrowcolor="#888",
                bgcolor="rgba(255,255,255,0.85)",
                borderpad=1,
                font=dict(size=11, color="#333"),
            )
    for i, trajectory in enumerate(trajectories):
        rows, name = trajectory["points"], escape(trajectory["name"])
        color = qualitative.Dark24[i % len(qualitative.Dark24)]
        hover = []
        for row in rows:
            neighbors = "<br>".join(
                f"{escape(n['label'])}: {n['distance']:.4f}" for n in row["neighbors"]
            )
            hover.append(
                f"<b>{name}</b><br>step {row['step']} · position {row['position']}"
                f" · {escape(str(row['segment']))}<br>input token: {escape(str(row['token']))}"
                f" (ID {row['token_id']})<br>residual norm: {row['norm']:.3f}"
                f"<br>Nearest roles ({role_map.config['metric']} distance):<br>{neighbors}"
            )
        fig.add_trace(
            go.Scatter(
                x=[r["x"] for r in rows],
                y=[r["y"] for r in rows],
                mode="lines+markers",
                name=name,
                legendgroup=name,
                line=dict(color=color, width=1.5),
                marker=dict(
                    size=6,
                    color=list(range(len(rows))),
                    cmin=0,
                    cmax=max(1, last_step),
                    colorscale="Viridis",
                    showscale=i == 0,
                    colorbar=dict(title="Token order", x=1.02),
                ),
                text=hover,
                hovertemplate="%{text}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[rows[0]["x"], rows[-1]["x"]],
                y=[rows[0]["y"], rows[-1]["y"]],
                mode="markers+text",
                marker=dict(
                    symbol=["circle-open", "diamond-open"], size=14, color=color
                ),
                text=["start", "end"],
                textposition="top center",
                showlegend=False,
                legendgroup=name,
                hoverinfo="skip",
            )
        )
        cursors.append((len(fig.data), rows, hover))
        fig.add_trace(
            go.Scatter(
                x=[rows[0]["x"]],
                y=[rows[0]["y"]],
                mode="markers",
                marker=dict(
                    size=17, color=color, symbol="circle-open", line=dict(width=3)
                ),
                text=[hover[0]],
                hovertemplate="%{text}<extra></extra>",
                showlegend=False,
                legendgroup=name,
            )
        )
    if cursors:
        # Only the cursors change: reference anchors and complete paths remain fixed.
        frames = []
        for step in range(max(len(rows) for _, rows, _ in cursors)):
            frames.append(
                go.Frame(
                    name=str(step),
                    traces=[i for i, _, _ in cursors],
                    data=[
                        go.Scatter(
                            x=[rows[step]["x"]] if step < len(rows) else [],
                            y=[rows[step]["y"]] if step < len(rows) else [],
                            text=[hover[step]] if step < len(rows) else [],
                        )
                        for _, rows, hover in cursors
                    ],
                )
            )
        fig.frames = frames
        fig.update_layout(
            sliders=[
                dict(
                    currentvalue=dict(prefix="Trajectory row: "),
                    steps=[
                        dict(
                            label=f.name,
                            method="animate",
                            args=[
                                [f.name],
                                dict(
                                    mode="immediate",
                                    frame=dict(duration=0, redraw=False),
                                    transition=dict(duration=0),
                                ),
                            ],
                        )
                        for f in frames
                    ],
                )
            ],
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    direction="left",
                    x=0,
                    xanchor="left",
                    y=-0.15,
                    buttons=[
                        dict(
                            label="Play",
                            method="animate",
                            args=[
                                None,
                                dict(
                                    fromcurrent=True,
                                    frame=dict(duration=160, redraw=False),
                                    transition=dict(duration=0),
                                ),
                            ],
                        ),
                        dict(
                            label="Pause",
                            method="animate",
                            args=[
                                [None],
                                dict(
                                    mode="immediate",
                                    frame=dict(duration=0, redraw=False),
                                    transition=dict(duration=0),
                                ),
                            ],
                        ),
                    ],
                )
            ],
        )
    fig.update_layout(
        template="plotly_white",
        width=1300,
        height=900,
        title=(
            f"Role space · {escape(role_map.config['model_id'])} · layer {role_map.config['layer']}"
            + (
                f" · Ward {role_map.config['n_clusters']}"
                if role_map.config.get("clustering") == "ward"
                else ""
            )
            + f"<br><sup>Fixed role t-SNE · {role_map.config['metric']} · clusters in original space; "
            "2D gaps and path lengths are qualitative</sup>"
        ),
        xaxis_title="t-SNE 1",
        yaxis_title="t-SNE 2",
        yaxis=dict(scaleanchor="x", scaleratio=1),
        legend=dict(x=1.14, y=1),
        margin=dict(b=130, r=350),
    )
    return fig


def write_outputs(role_map, trajectories, output: Path, formats=("html",)):
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": role_map.config,
        "roles": role_map.role_records(),
        "trajectories": trajectories,
    }
    (output / "projection.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    fig = make_figure(role_map, trajectories)
    if "html" in formats:
        fig.write_html(output / "roles.html", include_plotlyjs=True, auto_play=False)
    if "png" in formats:
        # Keep the PNG an overview without browser playback controls.
        fig.layout.sliders = ()
        fig.layout.updatemenus = ()
        try:
            fig.write_image(output / "roles.png", scale=2)
        except (RuntimeError, ValueError) as error:
            raise RuntimeError(
                "PNG export needs Kaleido and Chrome/Chromium. Install Chrome with "
                "`uv run --extra plots plotly_get_chrome -y`. JSON and any requested "
                "HTML were saved; rerun with --map and --formats png."
            ) from error
    if "subcluster_ids" in role_map.config and not np.array_equal(
        role_map.clusters, role_map.config["subcluster_ids"]
    ):
        from dataclasses import replace

        finer = replace(
            role_map,
            clusters=np.asarray(role_map.config["subcluster_ids"]),
            cluster_names={
                int(k): v for k, v in role_map.config["subcluster_names"].items()
            },
            config={
                **role_map.config,
                "n_clusters": role_map.config["subcluster_count"],
            },
        )
        subfig = make_figure(finer, trajectories)
        if "html" in formats:
            subfig.write_html(
                output / "subclusters.html", include_plotlyjs=True, auto_play=False
            )
        if "png" in formats:
            subfig.layout.sliders = ()
            subfig.layout.updatemenus = ()
            subfig.write_image(output / "subclusters.png", scale=2)
    return fig
