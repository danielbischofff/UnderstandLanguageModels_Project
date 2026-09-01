"""Intervention configuration and inspectable residual-stream mathematics."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import torch

from .recipes import VectorRecipe

ScalingMode = Literal["raw", "unit"]
# Token positions modified within each forward call (never a layer selector):
# ``all`` modifies the full prefill sequence and all cached decode inputs;
# ``last`` modifies only the final sequence position in each call; and
# ``generated_only`` leaves prefill/first-token logits unchanged, then modifies
# positions in cached decoding calls.
PositionPolicy = Literal["all", "last", "generated_only"]
BoundType = Literal["lower", "upper", "interval"]
Ordering = Literal["add_then_cap", "cap_then_add"]


def _normalize_scaling(value: str) -> str:
    return value.replace("-", "_")


def _validate_layers(
    layers: Sequence[int] | None, *, required: bool
) -> tuple[int, ...] | None:
    if layers is None:
        if required:
            raise ValueError("capping layers must be specified explicitly")
        return None
    result = tuple(int(layer) for layer in layers)
    if not result:
        raise ValueError("layers must not be empty")
    if len(result) != len(set(result)):
        raise ValueError("layers must not contain duplicates")
    return result


@dataclass(frozen=True)
class SteeringRequest:
    """One additive vector intervention.

    If ``layers`` is omitted, inference resolves it to
    ``model.config.num_hidden_layers // 2``.
    """

    vector: VectorRecipe
    coefficient: float | Mapping[int, float] = 1.0
    layers: tuple[int, ...] | None = None
    scaling: ScalingMode = "raw"
    positions: PositionPolicy = "all"
    preserve_original_activation_norm: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.vector, VectorRecipe):
            raise TypeError("vector must be a VectorRecipe")
        object.__setattr__(
            self, "layers", _validate_layers(self.layers, required=False)
        )
        scaling = _normalize_scaling(self.scaling)
        if scaling not in {"raw", "unit"}:
            raise ValueError(f"unknown scaling mode: {self.scaling!r}")
        object.__setattr__(self, "scaling", scaling)
        if self.positions not in {"all", "last", "generated_only"}:
            raise ValueError(f"unknown token-position policy: {self.positions!r}")
        values = (
            self.coefficient.values()
            if isinstance(self.coefficient, Mapping)
            else [self.coefficient]
        )
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("coefficients must be finite")

    def coefficient_for(self, layer: int) -> float:
        if not isinstance(self.coefficient, Mapping):
            return float(self.coefficient)
        if layer in self.coefficient:
            return float(self.coefficient[layer])
        if (
            str(layer) in self.coefficient
        ):  # YAML/JSON keys often deserialize as strings
            return float(self.coefficient[str(layer)])  # type: ignore[index]
        raise ValueError(f"missing coefficient for layer {layer}")


Threshold = (
    float
    | tuple[float, float]
    | Mapping[int, float]
    | Mapping[int, tuple[float, float]]
)


@dataclass(frozen=True)
class CappingRequest:
    """Projection cap along a unit direction; layers are always explicit."""

    direction: VectorRecipe
    bound: BoundType
    layers: tuple[int, ...]
    threshold: Threshold
    positions: PositionPolicy = "all"

    def __post_init__(self) -> None:
        if not isinstance(self.direction, VectorRecipe):
            raise TypeError("direction must be a VectorRecipe")
        object.__setattr__(self, "layers", _validate_layers(self.layers, required=True))
        if self.bound not in {"lower", "upper", "interval"}:
            raise ValueError("bound must be 'lower', 'upper', or 'interval'")
        if self.positions not in {"all", "last", "generated_only"}:
            raise ValueError(f"unknown token-position policy: {self.positions!r}")

    def threshold_for(self, layer: int) -> float | tuple[float, float]:
        value: Any = self.threshold
        if isinstance(value, Mapping):
            if layer in value:
                value = value[layer]
            elif str(layer) in value:
                value = value[str(layer)]
            else:
                raise ValueError(f"missing cap threshold for layer {layer}")
        if self.bound == "interval":
            if not isinstance(value, (tuple, list)) or len(value) != 2:
                raise ValueError("interval threshold must be a (lower, upper) pair")
            lower, upper = float(value[0]), float(value[1])
            if not all(math.isfinite(item) for item in (lower, upper)) or lower > upper:
                raise ValueError("invalid interval threshold")
            return lower, upper
        result = float(value)
        if not math.isfinite(result):
            raise ValueError("cap threshold must be finite")
        return result


@dataclass(frozen=True)
class SteeringPlan:
    additive: tuple[SteeringRequest, ...] = field(default_factory=tuple)
    caps: tuple[CappingRequest, ...] = field(default_factory=tuple)
    ordering: Ordering = "add_then_cap"

    def __init__(
        self,
        additive: Sequence[SteeringRequest] = (),
        caps: Sequence[CappingRequest] = (),
        ordering: Ordering = "add_then_cap",
    ) -> None:
        if ordering not in {"add_then_cap", "cap_then_add"}:
            raise ValueError("ordering must be 'add_then_cap' or 'cap_then_add'")
        if not all(isinstance(item, SteeringRequest) for item in additive):
            raise TypeError("additive entries must be SteeringRequest instances")
        if not all(isinstance(item, CappingRequest) for item in caps):
            raise TypeError("cap entries must be CappingRequest instances")
        object.__setattr__(self, "additive", tuple(additive))
        object.__setattr__(self, "caps", tuple(caps))
        object.__setattr__(self, "ordering", ordering)


@dataclass(frozen=True)
class ResolvedIntervention:
    operation: Literal["add", "cap"]
    layer: int
    direction: torch.Tensor
    positions: PositionPolicy
    recipe_hash: str
    delta: torch.Tensor | None = None
    preserve_original_activation_norm: bool = False
    bound: BoundType | None = None
    threshold: float | tuple[float, float] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


def unit_direction(vector: torch.Tensor) -> torch.Tensor:
    norm = torch.linalg.vector_norm(vector.to(torch.float32))
    if not torch.isfinite(norm) or float(norm) == 0.0:
        raise ValueError("direction must have a finite nonzero norm")
    return vector.to(torch.float32) / norm


def _selected(
    activations: torch.Tensor, positions: PositionPolicy
) -> tuple[Any, torch.Tensor]:
    if activations.ndim < 2:
        raise ValueError("expected activations with a sequence dimension")
    if positions == "last":
        index = (..., slice(-1, None), slice(None))
        return index, activations[index]
    index = (..., slice(None), slice(None))
    return index, activations


def apply_addition(
    activations: torch.Tensor,
    delta: torch.Tensor,
    *,
    positions: PositionPolicy = "all",
    preserve_original_activation_norm: bool = False,
) -> torch.Tensor:
    """Apply h' = h + delta at selected positions."""

    index, selected = _selected(activations, positions)
    local_delta = delta.to(device=selected.device, dtype=selected.dtype)
    changed = selected + local_delta
    if preserve_original_activation_norm:
        original_norm = torch.linalg.vector_norm(
            selected.to(torch.float32), dim=-1, keepdim=True
        )
        changed_norm = torch.linalg.vector_norm(
            changed.to(torch.float32), dim=-1, keepdim=True
        )
        scale = torch.where(
            changed_norm > 0,
            original_norm / changed_norm.clamp_min(torch.finfo(torch.float32).eps),
            torch.zeros_like(changed_norm),
        ).to(changed.dtype)
        changed = changed * scale
    if positions == "last":
        result = activations.clone()
        result[index] = changed
        return result
    return changed


def apply_cap(
    activations: torch.Tensor,
    direction: torch.Tensor,
    *,
    bound: BoundType,
    threshold: float | tuple[float, float],
    positions: PositionPolicy = "all",
) -> torch.Tensor:
    """Clamp scalar projections along a unit direction."""

    index, selected = _selected(activations, positions)
    unit = unit_direction(direction).to(device=selected.device, dtype=selected.dtype)
    projection = torch.einsum("...d,d->...", selected, unit)
    if bound == "lower":
        target = projection.clamp_min(float(threshold))
    elif bound == "upper":
        target = projection.clamp_max(float(threshold))
    elif bound == "interval":
        lower, upper = threshold  # type: ignore[misc]
        target = projection.clamp(min=float(lower), max=float(upper))
    else:
        raise ValueError(f"unknown cap bound: {bound}")
    changed = selected + (target - projection).unsqueeze(-1) * unit
    if positions == "last":
        result = activations.clone()
        result[index] = changed
        return result
    return changed
