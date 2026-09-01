"""Thin adapter from recorded steering activations to the existing J-Lens code."""

from __future__ import annotations

import importlib
import sys
from collections.abc import Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
from tokenizers import Tokenizer

from .capture import ActivationArtifact


@contextmanager
def _analysis_imports(source_dir: Path):
    """Import the existing analysis modules without copying their algorithms."""

    source = str(source_dir.resolve())
    sys.path.insert(0, source)
    try:
        readout = importlib.import_module("readout")
        unembed = importlib.import_module("unembed")
        run_analysis = importlib.import_module("run_analysis")
        yield readout, unembed, run_analysis
    finally:
        if sys.path and sys.path[0] == source:
            sys.path.pop(0)


def _selected_event_metadata(
    artifact: ActivationArtifact,
    *,
    layer: int | None,
    phase: str,
    forward_index: int | None,
    points: Sequence[str],
) -> tuple[int, int, list[dict[str, Any]]]:
    events = [
        event
        for event in artifact.manifest["events"]
        if event["phase"] == phase and event["point"] in points
    ]
    available_layers = sorted({int(event["layer"]) for event in events})
    if layer is None:
        if len(available_layers) != 1:
            raise ValueError(
                "--layer is required when the selected capture has zero or multiple "
                f"matching layers; available={available_layers}"
            )
        layer = available_layers[0]
    events = [event for event in events if int(event["layer"]) == layer]
    available_forwards = sorted({int(event["forward_index"]) for event in events})
    if forward_index is None:
        if not available_forwards:
            raise ValueError(
                f"no captured events match layer={layer}, phase={phase!r}, points={list(points)}"
            )
        # Prefill normally has one call. For decode, the first call is the most
        # useful quick default and explicit --forward-index remains available.
        forward_index = available_forwards[0]
    selected = [
        event for event in events if int(event["forward_index"]) == forward_index
    ]
    by_point = {event["point"]: event for event in selected}
    missing = [point for point in points if point not in by_point]
    if missing:
        raise ValueError(
            f"capture has no {missing} event at layer={layer}, phase={phase!r}, "
            f"forward_index={forward_index}; available forwards={available_forwards}"
        )
    return layer, forward_index, [by_point[point] for point in points]


def quick_j_lens_tokens(
    artifact_path: str | Path,
    *,
    analysis_source: str | Path,
    data_dir: str | Path,
    lens_path: str | Path,
    layer: int | None = None,
    phase: str = "prefill",
    forward_index: int | None = None,
    token_index: int = -1,
    batch_index: int = 0,
    points: Sequence[str] = ("pre_intervention", "post_intervention"),
    top_k_count: int = 10,
) -> dict[str, Any]:
    """Read selected captured residuals with the repository's J-Lens utilities.

    By default this reads the final prompt position before intervention, after
    intervention, and the ``post - pre`` change. It performs no lens fitting or
    interpretability implementation of its own.
    """

    if top_k_count <= 0:
        raise ValueError("top_k_count must be positive")
    if phase not in {"prefill", "decode"}:
        raise ValueError("phase must be 'prefill' or 'decode'")
    if not points or len(points) != len(set(points)):
        raise ValueError("points must be nonempty and unique")
    unknown_points = set(points) - {"pre_intervention", "post_intervention"}
    if unknown_points:
        raise ValueError(f"unknown capture points: {sorted(unknown_points)}")
    artifact = ActivationArtifact.open(artifact_path)
    layer, forward_index, events = _selected_event_metadata(
        artifact,
        layer=layer,
        phase=phase,
        forward_index=forward_index,
        points=points,
    )
    rows: list[np.ndarray] = []
    labels: list[str] = []
    selected_token: dict[str, Any] | None = None
    for event in events:
        tensor = artifact.load_tensor(event["tensor_key"])
        try:
            row = tensor[batch_index, token_index].float().numpy()
        except IndexError as error:
            raise IndexError(
                f"batch/token index ({batch_index}, {token_index}) is invalid for "
                f"captured shape {list(tensor.shape)}"
            ) from error
        rows.append(row)
        labels.append(str(event["point"]))
        if selected_token is None:
            token_ids = event.get("input_token_ids")
            cache_positions = event.get("cache_positions")
            selected_token = {
                "input_token_id": (
                    token_ids[batch_index][token_index]
                    if token_ids is not None
                    else None
                ),
                "cache_position": (
                    cache_positions[token_index]
                    if cache_positions is not None
                    else None
                ),
            }
    if labels == ["pre_intervention", "post_intervention"]:
        rows.append(rows[1] - rows[0])
        labels.append("post_minus_pre")

    matrix = np.stack(rows).astype(np.float32, copy=False)
    with _analysis_imports(Path(analysis_source)) as (
        readout_module,
        unembed_module,
        analysis_module,
    ):
        lens = readout_module.LensTransport(lens_path)
        if layer not in lens.source_layers:
            raise ValueError(
                f"J-Lens has no fitted Jacobian for layer {layer}; "
                f"available={lens.source_layers}"
            )
        transported = lens.transport(layer, matrix)
        unembedding = unembed_module.Unembedding(data_dir)
        logits = unembedding.logits(transported)
        tokenizer = Tokenizer.from_file(str(Path(data_dir) / "tokenizer.json"))
        texts, wordlike = analysis_module.decode_vocab(
            tokenizer, unembedding.vocab_size
        )
        ids, scores, probabilities = unembed_module.top_k(logits, top_k_count)
        word_ids, word_scores, word_probabilities = unembed_module.top_k(
            logits, top_k_count, allowed=wordlike
        )

    def pack(row_index: int, token_ids, token_scores, token_probabilities):
        return [
            {
                "id": int(token_id),
                "text": texts[int(token_id)],
                "logit": float(score),
                "probability": float(probability),
            }
            for token_id, score, probability in zip(
                token_ids[row_index],
                token_scores[row_index],
                token_probabilities[row_index],
            )
        ]

    return {
        "schema_version": 1,
        "source": {
            "activation_artifact": str(Path(artifact_path)),
            "lens": str(Path(lens_path)),
            "analysis_source": str(Path(analysis_source)),
            "lens_prompts_fitted": lens.n_prompts,
        },
        "selection": {
            "layer": layer,
            "phase": phase,
            "forward_index": forward_index,
            "batch_index": batch_index,
            "token_index": token_index,
            **(selected_token or {}),
        },
        "readouts": [
            {
                "label": label,
                "top": pack(index, ids, scores, probabilities),
                "top_wordlike": pack(index, word_ids, word_scores, word_probabilities),
            }
            for index, label in enumerate(labels)
        ],
    }
