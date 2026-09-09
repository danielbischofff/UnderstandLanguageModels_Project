"""Local role tensors, portable NPZ matrices, and steering capture artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from ..capture import ActivationArtifact
from ..vector_store import _extract_tensor


def matrix(values, name: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or min(values.shape) == 0 or not np.isfinite(values).all():
        raise ValueError(f"{name} must be a nonempty finite [rows, hidden] matrix")
    return values


def _check_metadata(data, layer, model_id):
    if "layer" in data and int(data["layer"]) != layer:
        raise ValueError("input layer does not match the role map")
    if "model_id" in data and str(data["model_id"]) != model_id:
        raise ValueError("input model_id does not match the role map")


def load_roles(path: Path, layer: int, model_id: str):
    """Read role_vectors/*.pt [layers, hidden] or NPZ vectors + labels."""
    if layer < 0:
        raise ValueError("layer must be zero-based and nonnegative")
    if path.is_dir():
        files = sorted(path.glob("*.pt"))
        if not files:
            raise ValueError(f"no role .pt files in {path}")
        vectors = [
            _extract_tensor(torch.load(p, map_location="cpu", weights_only=True), p)[
                layer
            ]
            .float()
            .numpy()
            for p in files
        ]
        labels = [p.stem for p in files]
    else:
        with np.load(path, allow_pickle=False) as data:
            _check_metadata(data, layer, model_id)
            vectors = data["vectors"]
            if vectors.ndim == 3:  # [roles, layers, hidden]
                vectors = vectors[:, layer, :]
            labels = data["labels"].tolist()
    vectors = matrix(vectors, "roles")
    if (
        not isinstance(labels, list)
        or len(labels) != len(vectors)
        or any(not isinstance(label, str) or not label.strip() for label in labels)
        or len(set(labels)) != len(labels)
    ):
        raise ValueError("labels must be unique nonempty strings, one per role")
    return vectors, labels


def load_trajectory(
    path: Path,
    *,
    layer: int,
    model_id: str,
    point: str = "post_intervention",
    batch_index: int = 0,
    generated_only: bool = False,
):
    """Return [tokens, hidden] and aligned records; never join different batches."""
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            _check_metadata(data, layer, model_id)
            if "point" in data and str(data["point"]) != point:
                raise ValueError("input capture point does not match --point")
            vectors = matrix(data["vectors"], "trajectory")
            columns = {
                key: data[key].tolist()
                for key in ("token_ids", "tokens", "positions", "segments")
                if key in data
            }
        if any(
            not isinstance(v, list) or len(v) != len(vectors) for v in columns.values()
        ):
            raise ValueError("trajectory metadata must have one value per vector")
        columns.setdefault("positions", list(range(len(vectors))))
        columns.setdefault("token_ids", [None] * len(vectors))
        columns.setdefault("tokens", [""] * len(vectors))
        columns.setdefault("segments", ["unknown"] * len(vectors))
        if any(
            s not in {"prompt", "generated", "unknown"} for s in columns["segments"]
        ):
            raise ValueError("segments must be prompt, generated, or unknown")
        rows = [
            {
                "step": i,
                "position": columns["positions"][i],
                "token_id": columns["token_ids"][i],
                "token": columns["tokens"][i],
                "segment": columns["segments"][i],
                "point": point,
            }
            for i in range(len(vectors))
        ]
    else:
        candidate = path / "activations" if (path / "activations").is_dir() else path
        artifact = ActivationArtifact.open(candidate)
        coordinate = artifact.manifest.get("coordinate_system", {})
        if coordinate.get("residual_stream") != "post_block_post_mlp":
            raise ValueError("artifact must contain post-block/post-MLP residuals")
        metadata_path = artifact.directory.parent / "metadata.json"
        metadata = (
            json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        )
        _check_metadata(metadata, layer, model_id)
        tokens = metadata.get("tokens", {})
        prompt_ids = tokens.get("prompt_ids")
        all_ids = (prompt_ids or []) + tokens.get("generated_ids", [])
        all_text = (tokens.get("prompt_tokens") or [""] * len(prompt_ids or [])) + (
            tokens.get("generated_tokens")
            or [""] * len(tokens.get("generated_ids", []))
        )
        events = sorted(
            (
                e
                for e in artifact.manifest["events"]
                if e["layer"] == layer and e["point"] == point
            ),
            key=lambda e: e["forward_index"],
        )
        vectors, rows = [], []
        for event in events:
            tensor = artifact.load_tensor(event["tensor_key"])
            if tensor.ndim != 3 or not 0 <= batch_index < tensor.shape[0]:
                raise ValueError(
                    "capture must have [batch, positions, hidden] and a valid batch index"
                )
            values = tensor[batch_index].float().numpy()
            ids = event.get("input_token_ids")
            ids = ids[batch_index] if ids is not None else [None] * len(values)
            positions = event.get("cache_positions")
            if positions is None:
                # An all-position prefill also represents the experiment's full causal replay.
                positions = (
                    list(range(len(values)))
                    if (
                        event["phase"] == "prefill"
                        and event["position_policy"] == "all"
                    )
                    else [None] * len(values)
                )
            if len(ids) != len(values) or len(positions) != len(values):
                raise ValueError("capture token alignment does not match tensor")
            for value, token_id, position in zip(values, ids, positions):
                segment = "generated" if event["phase"] == "decode" else "prompt"
                if position is not None and prompt_ids is not None and batch_index == 0:
                    segment = "generated" if position >= len(prompt_ids) else "prompt"
                text = ""
                if (
                    batch_index == 0
                    and position is not None
                    and 0 <= position < len(all_ids)
                    and all_ids[position] == token_id
                ):
                    text = all_text[position]
                rows.append(
                    {
                        "step": len(rows),
                        "position": position,
                        "token_id": token_id,
                        "token": text,
                        "segment": segment,
                        "point": point,
                        "forward_index": event["forward_index"],
                        "phase": event["phase"],
                        "batch_index": batch_index,
                    }
                )
                vectors.append(value)
        if not rows:
            raise ValueError(f"no capture events at layer {layer}, point {point}")
        vectors = matrix(vectors, "trajectory")
    if generated_only:
        if any(r["segment"] == "unknown" for r in rows):
            raise ValueError("--generated-only needs segment metadata")
        mask = np.array([r["segment"] == "generated" for r in rows])
        vectors, rows = vectors[mask], [r for r, keep in zip(rows, mask) if keep]
        if not rows:
            raise ValueError("no generated-token residuals in this input")
    return vectors, rows
