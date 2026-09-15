"""Contrastive association timelines: one interactive figure per matrix cell.

Both measures share one x axis - the token positions of the prompt and the
generated answer - and each contrastive idea keeps one colour across both
panels. Positions where a measure jumps or drops relative to the series' own
variation are labelled with the token that was being processed, so a reader can
see *where* the point of view moved rather than only that it did.

The colours are the categorical slots in `quantitive_analysis.contrastive.PALETTE`,
assigned in fixed order and never cycled. Three of them sit below 3:1 contrast
on a light surface, so every series also carries a direct end label and the
run's CSV table stands in for the colours entirely.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any

import numpy as np

from .contrastive import COSINE_MEASURES, PRIMARY_COSINE, PRIMARY_RANK, RANK_MEASURES
from .quant import COSINE_ONLY, RunArchive, TrialSeries

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#7a7873"
GRID = "#e9e8e4"

MEASURE_TITLES = {
    "centroid_cosine": "cosine to set centroid (top-k, probability weighted)",
    "weighted_max_cosine": "cosine to nearest set word (top-k, probability weighted)",
    "top1_cosine": "cosine of the top word to the set centroid",
    "max_cosine": "largest top-k to set-word cosine",
    "median_rank": "median rank of the set's words",
    "mean_rank": "mean rank of the set's words",
    "min_rank": "best rank in the set",
    "mean_reciprocal_rank": "mean reciprocal rank of the set's words",
}


@dataclass(frozen=True)
class PlotOptions:
    """Which measures to draw and how eagerly to label a change.

    ``change_z`` scales the labelling threshold in robust standard deviations of
    the series' own step-to-step change; ``min_change_fraction`` additionally
    requires the step to cover that fraction of the series' range, which keeps a
    flat timeline from being labelled everywhere. Raise either to label less.
    """

    cosine_measure: str = PRIMARY_COSINE
    rank_measure: str = PRIMARY_RANK
    change_z: float = 3.0
    min_change_fraction: float = 0.08
    max_annotations: int = 6
    height: int = 820

    def __post_init__(self) -> None:
        if self.cosine_measure not in COSINE_MEASURES:
            raise ValueError(f"unknown cosine measure {self.cosine_measure!r}")
        if self.rank_measure not in RANK_MEASURES:
            raise ValueError(f"unknown rank measure {self.rank_measure!r}")
        if self.change_z < 0 or self.min_change_fraction < 0:
            raise ValueError("change thresholds must be nonnegative")
        if self.max_annotations < 0:
            raise ValueError("max_annotations must be nonnegative")


def _column(values: Sequence[float | None], length: int) -> np.ndarray:
    column = np.full(length, np.nan)
    for index, value in enumerate(values[:length]):
        if value is not None:
            column[index] = float(value)
    return column


def significant_changes(
    values: np.ndarray, options: PlotOptions, *, logarithmic: bool = False
) -> list[int]:
    """Positions whose step away from the previous one stands out in this series.

    The scale is the median absolute deviation of the steps, so a few large
    moves do not raise the bar for the rest. Rank series are compared in orders
    of magnitude, where a change from rank 40000 to 400 is one step of two.
    """

    series = np.log10(np.where(values > 0, values, np.nan)) if logarithmic else np.asarray(values, float)
    steps = np.diff(series)
    finite = steps[np.isfinite(steps)]
    if finite.size < 4 or options.max_annotations == 0:
        return []
    deviation = float(np.median(np.abs(finite - np.median(finite))))
    span = float(np.nanmax(series) - np.nanmin(series)) if np.isfinite(series).any() else 0.0
    threshold = max(options.change_z * 1.4826 * deviation, options.min_change_fraction * span)
    if threshold <= 0:
        return []
    candidates = [
        (abs(float(step)), index + 1)
        for index, step in enumerate(steps)
        if np.isfinite(step) and abs(float(step)) >= threshold
    ]
    # Largest move first, earliest position breaking a tie.
    candidates.sort(key=lambda item: (-item[0], item[1]))
    return sorted(index for _, index in candidates[: options.max_annotations])


def _token_label(text: str) -> str:
    cleaned = text.replace("\n", "\\n").replace("\r", "").strip() or text.replace("\n", "\\n")
    return escape(cleaned[:18]) if cleaned else "·"


def _change_text(token: str, previous: float, current: float, *, ranks: bool) -> str:
    arrow = "▲" if (current > previous) == (not ranks) else "▼"
    if ranks:
        detail = f"{previous:,.0f}→{current:,.0f}"
    else:
        detail = f"{current - previous:+.3f}"
    return f"{_token_label(token)}<br>{arrow} {detail}"


def build_figure(
    trial: TrialSeries,
    layer_index: int,
    *,
    experiment: str,
    run: str,
    options: PlotOptions = PlotOptions(),
    table: str | None = None,
) -> Any:
    """One figure: cosine association above, rank association below."""

    try:
        from plotly import graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as error:  # pragma: no cover - optional dependency
        raise ImportError(
            "plotting needs plotly; install the project's 'plots' extra "
            "(uv sync --extra plots)"
        ) from error

    layer = trial.layers[layer_index]
    positions = len(trial.tokens)
    tokens = [str(token.get("text", "")) for token in trial.tokens]
    x = list(range(positions))
    rank_available = any(
        np.isfinite(_column(item.measures.get(options.rank_measure, ()), positions)).any()
        for item in layer.sets
    )
    rows = 2 if rank_available else 1
    figure = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.09,
        subplot_titles=(
            [MEASURE_TITLES[options.cosine_measure], MEASURE_TITLES[options.rank_measure]]
            if rank_available
            else [MEASURE_TITLES[options.cosine_measure]]
        ),
    )

    # The token names ride on an invisible trace, so the unified hover box names
    # the token being processed once instead of repeating it per series. Its y
    # sits inside the drawn range so it cannot stretch the axis.
    drawn = np.concatenate(
        [_column(item.measures.get(options.cosine_measure, ()), positions) for item in layer.sets]
    )
    baseline = float(np.nanmedian(drawn)) if np.isfinite(drawn).any() else 0.0
    figure.add_trace(
        go.Scatter(
            x=x,
            y=[baseline] * positions,
            mode="markers",
            marker={"opacity": 0, "size": 1},
            text=[
                f"{_token_label(token)}  ({trial.tokens[index].get('segment', '')})"
                for index, token in enumerate(tokens)
            ],
            hoverinfo="text",
            showlegend=False,
            name="token",
        ),
        row=1,
        col=1,
    )

    annotations: list[dict[str, Any]] = []
    rank_extent: list[float] = []
    for item in layer.sets:
        for row, measure, ranks in (
            (1, options.cosine_measure, False),
            (2, options.rank_measure, True),
        ):
            if row > rows:
                continue
            values = _column(item.measures.get(measure, ()), positions)
            if not np.isfinite(values).any():
                continue
            if ranks:
                rank_extent.extend(values[np.isfinite(values) & (values > 0)].tolist())
            figure.add_trace(
                go.Scatter(
                    x=x,
                    y=[None if not np.isfinite(value) else float(value) for value in values],
                    name=item.label,
                    legendgroup=item.id,
                    showlegend=row == 1,
                    mode="lines",
                    connectgaps=False,
                    line={"color": item.color, "width": 2},
                    hovertemplate="%{y:.4g}<extra>" + escape(item.label) + "</extra>",
                ),
                row=row,
                col=1,
            )
            finite = np.flatnonzero(np.isfinite(values))
            if finite.size and row == 1:
                last = int(finite[-1])
                figure.add_trace(
                    go.Scatter(
                        x=[last],
                        y=[float(values[last])],
                        mode="markers+text",
                        marker={"color": item.color, "size": 8},
                        text=[f" {escape(item.label)}"],
                        textposition="middle right",
                        textfont={"color": INK_SECONDARY, "size": 11},
                        showlegend=False,
                        legendgroup=item.id,
                        hoverinfo="skip",
                    ),
                    row=row,
                    col=1,
                )
            for index in significant_changes(values, options, logarithmic=ranks):
                rising = values[index] > values[index - 1]
                annotations.append(
                    {
                        "x": index,
                        "y": float(values[index]),
                        "xref": "x" if row == 1 else f"x{row}",
                        "yref": "y" if row == 1 else f"y{row}",
                        "text": _change_text(
                            tokens[index], float(values[index - 1]), float(values[index]), ranks=ranks
                        ),
                        "showarrow": True,
                        "arrowhead": 6,
                        "arrowsize": 0.7,
                        "arrowwidth": 1,
                        "arrowcolor": INK_MUTED,
                        "ax": 0,
                        "ay": -30 if rising else 30,
                        "align": "center",
                        "font": {"size": 10, "color": INK},
                        "bgcolor": "rgba(252,252,251,0.82)",
                        "bordercolor": GRID,
                        "borderwidth": 1,
                        "borderpad": 2,
                    }
                )

    if 0 < trial.prompt_length < positions:
        figure.add_vline(
            x=trial.prompt_length - 0.5,
            line={"color": INK_MUTED, "width": 1, "dash": "dot"},
        )
        annotations.append(
            {
                "x": trial.prompt_length - 0.5,
                "y": 1.0,
                "xref": "x",
                "yref": "paper",
                "text": "answer begins",
                "showarrow": False,
                "xanchor": "left",
                "yanchor": "bottom",
                "font": {"size": 10, "color": INK_MUTED},
            }
        )

    words = ", ".join(
        f"{item.label} ({len(item.words)} words)" for item in layer.sets
    )
    caption = (
        f"{escape(experiment)} · run {escape(run)} · layer {layer.layer} · "
        f"top-{layer.top_k} readout · {escape(layer.source)}"
    )
    if layer.source == COSINE_ONLY:
        caption += " · ranks need retained activations"
    footer = f"sets: {escape(words)}"
    if table is not None:
        footer += f" · table: {escape(table)}"
    annotations.append(
        {
            "x": 0.0,
            "y": -0.12 if rows == 2 else -0.24,
            "xref": "paper",
            "yref": "paper",
            "text": footer,
            "showarrow": False,
            "xanchor": "left",
            "font": {"size": 10, "color": INK_MUTED},
        }
    )

    figure.update_layout(
        title={
            "text": (
                f"{escape(trial.question['id'])} — {escape(trial.label)}"
                f"<br><span style='font-size:12px;color:{INK_SECONDARY}'>{caption}</span>"
            ),
            "font": {"size": 17, "color": INK},
            "x": 0,
            "xanchor": "left",
            "y": 0.975,
            "yanchor": "top",
        },
        annotations=list(figure.layout.annotations) + annotations,
        height=options.height,
        hovermode="x unified",
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font={"family": "system-ui, -apple-system, Segoe UI, sans-serif", "color": INK, "size": 12},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.03,
            "x": 0,
            "title": {"text": ""},
            "bgcolor": "rgba(0,0,0,0)",
        },
        margin={"l": 76, "r": 136, "t": 152, "b": 104},
    )
    figure.update_xaxes(
        gridcolor=GRID,
        zeroline=False,
        linecolor=GRID,
        ticks="outside",
        tickcolor=GRID,
        range=[-1, positions + max(6.0, 0.16 * positions)],
    )
    figure.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=GRID, ticks="outside", tickcolor=GRID)
    figure.update_xaxes(title_text="token position (prompt, then generated answer)", row=rows, col=1)
    figure.update_yaxes(title_text="cosine", row=1, col=1)
    if rows == 2:
        # A reversed log axis needs its range in exponents: plotly's own
        # reversed autorange picks nonsense bounds for rank data.
        best, worst = min(rank_extent), max(rank_extent)
        figure.update_yaxes(
            title_text="rank (better is higher)",
            type="log",
            dtick=1,
            range=[np.log10(worst * 2.0), np.log10(max(0.7, best / 2.0))],
            row=2,
            col=1,
        )
    for note in figure.layout.annotations[: rows]:
        note.font = {"size": 12, "color": INK_SECONDARY}
        note.x = 0
        note.xanchor = "left"
    return figure


def figure_filename(trial: TrialSeries, layer: int) -> str:
    question = "".join(
        character if character.isalnum() or character in "-_" else "-"
        for character in str(trial.question["id"])
    )
    return f"{question}__{trial.key}__layer-{layer:02d}.html"


def write_figures(
    archive: RunArchive,
    series: Sequence[TrialSeries],
    destination: Path,
    *,
    options: PlotOptions = PlotOptions(),
    table: str | None = None,
) -> list[Path]:
    """One self-contained page per trial and layer, sharing one plotly bundle."""

    destination.mkdir(parents=True, exist_ok=True)
    experiment = str(archive.experiment.get("name", "experiment"))
    written: list[Path] = []
    for trial in series:
        for index, layer in enumerate(trial.layers):
            figure = build_figure(
                trial,
                index,
                experiment=experiment,
                run=archive.run,
                options=options,
                table=table,
            )
            path = destination / figure_filename(trial, layer.layer)
            figure.write_html(
                path,
                include_plotlyjs="directory",
                full_html=True,
                config={"displaylogo": False, "responsive": True},
            )
            written.append(path)
    return written


def write_index(
    archive: RunArchive,
    series: Sequence[TrialSeries],
    destination: Path,
    *,
    table: str | None = None,
    document: str | None = None,
) -> Path:
    """A plain index so a run's figures are navigable without a server."""

    experiment = escape(str(archive.experiment.get("name", "experiment")))
    grouped: dict[str, list[tuple[TrialSeries, int, str]]] = {}
    for trial in series:
        for layer in trial.layers:
            grouped.setdefault(str(trial.question["id"]), []).append(
                (trial, layer.layer, layer.source)
            )
    sections = []
    for question, entries in grouped.items():
        rows = "".join(
            f'<li><a href="{escape(figure_filename(trial, layer))}">{escape(trial.label)}'
            f"</a> <span class=meta>layer {layer} · {escape(source)}</span></li>"
            for trial, layer, source in entries
        )
        text = escape(next(t.question.get("text", "") for t, _, _ in entries))
        sections.append(f"<h2>{escape(question)}</h2><p class=meta>{text}</p><ul>{rows}</ul>")
    downloads = [
        f'<a href="{escape(name)}">{escape(name)}</a>'
        for name in (table, document)
        if name is not None
    ]
    page = (
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{experiment} · contrastive association</title>"
        f"<style>body{{font:15px/1.5 system-ui,sans-serif;color:{INK};background:{SURFACE};"
        "max-width:920px;margin:40px auto;padding:0 16px}"
        f"h1{{font-size:22px}}h2{{font-size:16px;margin-top:28px}}.meta{{color:{INK_SECONDARY};font-size:13px}}"
        "li{margin:4px 0}a{color:#1c5cab}</style>"
        f"<h1>{experiment} · contrastive association</h1>"
        f"<p class=meta>Run {escape(archive.run)}. Cosine and rank association of every "
        "scored token position to each curated contrastive set."
        + (f" Data: {' · '.join(downloads)}." if downloads else "")
        + "</p>"
        + "".join(sections)
        + "</html>"
    )
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / "index.html"
    path.write_text(page, encoding="utf-8")
    return path
