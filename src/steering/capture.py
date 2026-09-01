"""Framework-neutral activation capture records for downstream analysis."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from safetensors import safe_open

# Capture boundaries at one selected layer. ``pre_intervention`` is the block
# output immediately before the first ordered addition/cap at that layer;
# ``post_intervention`` is the result immediately after the last operation.
# Capturing both therefore gives the most useful direct before/after pair.
CapturePoint = Literal["pre_intervention", "post_intervention"]

# Token-position policy within each model forward call (this does not select
# layers):
# - ``all`` stores every sequence position: the full prompt during prefill and
#   every input position supplied by subsequent cached decode calls.
# - ``last`` stores only the final sequence position of each call: normally the
#   last prompt token during prefill and the current token during each decode.
# - ``generated_only`` skips prompt prefill and stores cached decode calls. The
#   first sampled token has no decode activation until it is fed back as input
#   to predict the second token; the final sampled token may never be fed back.
CapturePositions = Literal["all", "last", "generated_only"]

# ``model`` preserves the observed activation dtype. Explicit dtypes trade
# storage/precision independently of the model's inference dtype.
CaptureDType = Literal["model", "float32", "float16", "bfloat16"]


@dataclass(frozen=True)
class ActivationCaptureConfig:
    """Select residual activations to retain for later interpretability work.

    Captures are always taken at transformer-block output, corresponding to the
    post-block/post-MLP residual stream used by the steering hooks.

    ``layers`` chooses zero-based transformer layers. ``points`` chooses the
    boundary around the ordered interventions at each layer. ``positions``
    chooses token positions within each forward call, not model layers.
    """

    layers: tuple[int, ...]
    points: tuple[CapturePoint, ...] = ("pre_intervention", "post_intervention")
    positions: CapturePositions = "all"
    dtype: CaptureDType = "model"

    def __init__(
        self,
        layers: list[int] | tuple[int, ...],
        points: list[CapturePoint] | tuple[CapturePoint, ...] = (
            "pre_intervention",
            "post_intervention",
        ),
        positions: CapturePositions = "all",
        dtype: CaptureDType = "model",
    ) -> None:
        normalized_layers = tuple(int(layer) for layer in layers)
        normalized_points = tuple(points)
        if not normalized_layers:
            raise ValueError("activation capture layers must not be empty")
        if len(normalized_layers) != len(set(normalized_layers)):
            raise ValueError("activation capture layers must not contain duplicates")
        if not normalized_points or len(normalized_points) != len(
            set(normalized_points)
        ):
            raise ValueError("activation capture points must be nonempty and unique")
        if any(
            point not in {"pre_intervention", "post_intervention"}
            for point in normalized_points
        ):
            raise ValueError("unknown activation capture point")
        if positions not in {"all", "last", "generated_only"}:
            raise ValueError("unknown activation capture position policy")
        if dtype not in {"model", "float32", "float16", "bfloat16"}:
            raise ValueError("unknown activation capture dtype")
        object.__setattr__(self, "layers", normalized_layers)
        object.__setattr__(self, "points", normalized_points)
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "dtype", dtype)

    def to_dict(self) -> dict[str, Any]:
        return {
            "layers": list(self.layers),
            "points": list(self.points),
            "positions": self.positions,
            "dtype": self.dtype,
            "hook_point": "transformer_block_output",
            "residual_stream": "post_block_post_mlp",
        }


@dataclass(frozen=True)
class ActivationCaptureEvent:
    """One layer/forward/point tensor with token-alignment metadata."""

    tensor: torch.Tensor
    layer: int
    point: CapturePoint
    phase: Literal["prefill", "decode"]
    forward_index: int
    input_token_ids: list[list[int]] | None
    cache_positions: list[int] | None
    position_policy: CapturePositions

    def manifest_entry(self, tensor_key: str) -> dict[str, Any]:
        return {
            "tensor_key": tensor_key,
            "layer": self.layer,
            "point": self.point,
            "phase": self.phase,
            "forward_index": self.forward_index,
            "input_token_ids": self.input_token_ids,
            "cache_positions": self.cache_positions,
            "position_policy": self.position_policy,
            "shape": list(self.tensor.shape),
            "dtype": str(self.tensor.dtype),
        }


def capture_dtype(value: CaptureDType, model_dtype: torch.dtype) -> torch.dtype:
    if value == "model":
        return model_dtype
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[value]


@dataclass(frozen=True)
class ActivationArtifact:
    """Lazy reader for a recorded activation directory."""

    directory: Path
    manifest: dict[str, Any]

    @classmethod
    def open(cls, path: str | Path) -> ActivationArtifact:
        candidate = Path(path)
        manifest_path = (
            candidate if candidate.is_file() else candidate / "manifest.json"
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != 1:
            raise ValueError(
                f"unsupported activation schema version: {manifest.get('schema_version')!r}"
            )
        tensor_path = manifest_path.parent / manifest["tensor_file"]
        if not tensor_path.is_file():
            raise FileNotFoundError(tensor_path)
        return cls(manifest_path.parent, manifest)

    @property
    def tensor_path(self) -> Path:
        return self.directory / self.manifest["tensor_file"]

    def load_tensor(self, tensor_key: str) -> torch.Tensor:
        """Load one CPU tensor without deserializing unrelated capture events."""

        with safe_open(self.tensor_path, framework="pt", device="cpu") as handle:
            if tensor_key not in set(handle.keys()):
                raise KeyError(tensor_key)
            return handle.get_tensor(tensor_key)

    def events(self):
        """Yield `(event_metadata, tensor)` pairs in recorded forward order."""

        for event in self.manifest["events"]:
            yield event, self.load_tensor(event["tensor_key"])
