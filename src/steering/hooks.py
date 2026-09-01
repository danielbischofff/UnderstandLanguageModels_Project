"""Exception-safe post-transformer-block forward hooks."""

from __future__ import annotations

import copy
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import TracebackType
from typing import Any, Self

import torch

from .interventions import (
    ResolvedIntervention,
    apply_addition,
    apply_cap,
    unit_direction,
)

BLOCK_PATHS = (
    "model.layers",
    "model.model.layers",
    "transformer.h",
    "language_model.layers",
    "gpt_neox.layers",
)


def locate_transformer_blocks(model: torch.nn.Module) -> tuple[Any, str]:
    for path in BLOCK_PATHS:
        current: Any = model
        for part in path.split("."):
            if not hasattr(current, part):
                break
            current = getattr(current, part)
        else:
            if hasattr(current, "__len__") and hasattr(current, "__getitem__"):
                return current, path
    raise ValueError(
        "Could not locate transformer blocks; checked " + ", ".join(BLOCK_PATHS)
    )


def _cache_has_tokens(cache: Any) -> bool:
    if cache is None:
        return False
    get_length = getattr(cache, "get_seq_length", None)
    if callable(get_length):
        try:
            return int(get_length()) > 0
        except (TypeError, ValueError):
            pass
    if isinstance(cache, (tuple, list)):
        if not cache:
            return False
        first = cache[0]
        if isinstance(first, (tuple, list)) and first and torch.is_tensor(first[0]):
            return first[0].shape[-2] > 0
    return True


class ForwardPhaseTracker:
    """Track prompt prefill versus cached decoding from explicit cache metadata."""

    def __init__(self) -> None:
        self.is_decode = False

    def pre_hook(
        self, module: torch.nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> None:
        del module, args
        cache_position = kwargs.get("cache_position")
        if torch.is_tensor(cache_position) and cache_position.numel():
            self.is_decode = bool(torch.min(cache_position).item() > 0)
            return
        self.is_decode = _cache_has_tokens(kwargs.get("past_key_values"))


def _extract_primary(output: Any) -> tuple[torch.Tensor | None, Any]:
    if torch.is_tensor(output):
        return output, ("tensor", None)
    if isinstance(output, tuple) and output and torch.is_tensor(output[0]):
        return output[0], ("tuple", None)
    if isinstance(output, list) and output and torch.is_tensor(output[0]):
        return output[0], ("list", None)
    if isinstance(output, Mapping):
        for key, value in output.items():
            if torch.is_tensor(value):
                return value, ("mapping", key)
    return None, ("unsupported", None)


def _replace_primary(output: Any, tensor: torch.Tensor, descriptor: Any) -> Any:
    kind, key = descriptor
    if kind == "tensor":
        return tensor
    if kind == "tuple":
        if hasattr(output, "_fields") and hasattr(output, "_replace"):
            return output._replace(**{output._fields[0]: tensor})
        return (tensor, *output[1:])
    if kind == "list":
        return [tensor, *output[1:]]
    if kind == "mapping":
        result = copy.copy(output)
        result[key] = tensor
        return result
    return output


@dataclass
class ProjectionDiagnostic:
    layer: int
    operation: str
    recipe_hash: str
    phase: str
    pre: list[float]
    post: list[float]


class SteeringContext:
    """Register grouped layer hooks and always remove them on context exit."""

    def __init__(
        self,
        model: torch.nn.Module,
        interventions: Iterable[ResolvedIntervention],
        *,
        collect_diagnostics: bool = False,
    ) -> None:
        self.model = model
        self.interventions = tuple(interventions)
        self.collect_diagnostics = collect_diagnostics
        self.diagnostics: list[ProjectionDiagnostic] = []
        self._handles: list[Any] = []
        self._tracker = ForwardPhaseTracker()

    def _hook(self, layer: int, operations: tuple[ResolvedIntervention, ...]):
        def hook(module: torch.nn.Module, args: tuple[Any, ...], output: Any) -> Any:
            del module, args
            tensor, descriptor = _extract_primary(output)
            if tensor is None:
                raise TypeError(
                    f"Unsupported transformer block output at layer {layer}: {type(output).__name__}"
                )
            changed = tensor
            for operation in operations:
                if (
                    operation.positions == "generated_only"
                    and not self._tracker.is_decode
                ):
                    continue
                effective_positions = (
                    "all"
                    if operation.positions == "generated_only"
                    else operation.positions
                )
                direction = operation.direction.to(
                    device=changed.device, dtype=changed.dtype
                )
                pre_projection: torch.Tensor | None = None
                if self.collect_diagnostics:
                    unit = unit_direction(direction).to(changed)
                    pre_projection = torch.einsum("...d,d->...", changed, unit)
                if operation.operation == "add":
                    assert operation.delta is not None
                    changed = apply_addition(
                        changed,
                        operation.delta,
                        positions=effective_positions,
                        preserve_original_activation_norm=operation.preserve_original_activation_norm,
                    )
                else:
                    assert (
                        operation.bound is not None and operation.threshold is not None
                    )
                    changed = apply_cap(
                        changed,
                        direction,
                        bound=operation.bound,
                        threshold=operation.threshold,
                        positions=effective_positions,
                    )
                if self.collect_diagnostics and pre_projection is not None:
                    unit = unit_direction(direction).to(changed)
                    post_projection = torch.einsum("...d,d->...", changed, unit)
                    self.diagnostics.append(
                        ProjectionDiagnostic(
                            layer=layer,
                            operation=operation.operation,
                            recipe_hash=operation.recipe_hash,
                            phase="decode" if self._tracker.is_decode else "prefill",
                            pre=pre_projection.detach()
                            .to(torch.float32)
                            .flatten()
                            .cpu()
                            .tolist(),
                            post=post_projection.detach()
                            .to(torch.float32)
                            .flatten()
                            .cpu()
                            .tolist(),
                        )
                    )
            return _replace_primary(output, changed, descriptor)

        return hook

    def __enter__(self) -> Self:
        if not self.interventions:
            return self
        blocks, _ = locate_transformer_blocks(self.model)
        grouped: dict[int, list[ResolvedIntervention]] = {}
        for intervention in self.interventions:
            if not 0 <= intervention.layer < len(blocks):
                raise IndexError(
                    f"layer {intervention.layer} out of range for {len(blocks)} transformer blocks"
                )
            grouped.setdefault(intervention.layer, []).append(intervention)
        try:
            self._handles.append(
                self.model.register_forward_pre_hook(
                    self._tracker.pre_hook, with_kwargs=True
                )
            )
            for layer, operations in grouped.items():
                self._handles.append(
                    blocks[layer].register_forward_hook(
                        self._hook(layer, tuple(operations))
                    )
                )
        except Exception:
            self._remove_hooks()
            raise
        return self

    def _remove_hooks(self) -> None:
        for handle in reversed(self._handles):
            handle.remove()
        self._handles.clear()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        del exc_type, exc, traceback
        self._remove_hooks()
        return False
