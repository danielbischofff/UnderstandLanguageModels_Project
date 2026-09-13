"""Exception-safe post-transformer-block forward hooks."""

from __future__ import annotations

import copy
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import TracebackType
from typing import Any, Self

import torch

from .capture import (
    ActivationCaptureConfig,
    ActivationCaptureEvent,
    CapturePoint,
    capture_dtype,
)
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
        self.forward_index = -1
        self.input_token_ids: list[list[int]] | None = None
        self.cache_positions: list[int] | None = None

    def pre_hook(
        self, module: torch.nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> None:
        del module
        self.forward_index += 1
        input_ids = kwargs.get("input_ids")
        if input_ids is None and args and torch.is_tensor(args[0]):
            candidate = args[0]
            if candidate.ndim == 2 and candidate.dtype in {
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
                torch.uint8,
            }:
                input_ids = candidate
        self.input_token_ids = (
            input_ids.detach().cpu().tolist() if torch.is_tensor(input_ids) else None
        )
        cache_position = kwargs.get("cache_position")
        self.cache_positions = (
            cache_position.detach().to(torch.int64).flatten().cpu().tolist()
            if torch.is_tensor(cache_position)
            else None
        )
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
        activation_capture: ActivationCaptureConfig | None = None,
        replay_prompt_length: int | None = None,
        user_query_start: int | None = None,
    ) -> None:
        self.model = model
        self.replay_prompt_length = replay_prompt_length
        # First token of the final user message; required by "from_user_query".
        self.user_query_start = user_query_start
        self.interventions = tuple(interventions)
        self.collect_diagnostics = collect_diagnostics
        self.activation_capture = activation_capture
        self.diagnostics: list[ProjectionDiagnostic] = []
        self.activation_events: list[ActivationCaptureEvent] = []
        self._handles: list[Any] = []
        self._tracker = ForwardPhaseTracker()

    def _capture_tensor(
        self, tensor: torch.Tensor, layer: int, point: CapturePoint
    ) -> None:
        config = self.activation_capture
        if config is None or layer not in config.layers or point not in config.points:
            return
        if config.positions == "generated_only" and not self._tracker.is_decode:
            return
        selected = tensor
        input_token_ids = self._tracker.input_token_ids
        cache_positions = self._tracker.cache_positions
        if config.positions == "last":
            selected = tensor[..., -1:, :]
            if input_token_ids is not None:
                input_token_ids = [row[-1:] for row in input_token_ids]
            if cache_positions is not None:
                cache_positions = cache_positions[-1:]
        stored = (
            selected.detach()
            .to(device="cpu", dtype=capture_dtype(config.dtype, selected.dtype))
            .contiguous()
        )
        self.activation_events.append(
            ActivationCaptureEvent(
                tensor=stored,
                layer=layer,
                point=point,
                phase="decode" if self._tracker.is_decode else "prefill",
                forward_index=self._tracker.forward_index,
                input_token_ids=input_token_ids,
                cache_positions=cache_positions,
                position_policy=config.positions,
            )
        )

    def _hook(self, layer: int, operations: tuple[ResolvedIntervention, ...]):
        def hook(module: torch.nn.Module, args: tuple[Any, ...], output: Any) -> Any:
            del module, args
            tensor, descriptor = _extract_primary(output)
            if tensor is None:
                raise TypeError(
                    f"Unsupported transformer block output at layer {layer}: {type(output).__name__}"
                )
            changed = tensor
            self._capture_tensor(changed, layer, "pre_intervention")
            for operation in operations:
                if (
                    operation.positions == "generated_only"
                    and not self._tracker.is_decode
                    and self.replay_prompt_length is None
                ):
                    continue
                effective_positions = (
                    "all"
                    if operation.positions == "generated_only"
                    else operation.positions
                )
                # Replay the positions changed during prefill + cached decoding.
                # "last" includes the final prompt token and every response token;
                # "generated_only" starts with the first response token.
                replay_start = None
                if operation.positions == "from_user_query":
                    if self.user_query_start is None:
                        raise ValueError("from_user_query steering requires user_query_start")
                    # Cached decode calls lie entirely after the query.
                    effective_positions = "all"
                    if not self._tracker.is_decode:
                        replay_start = self.user_query_start
                elif self.replay_prompt_length is not None and operation.positions != "all":
                    replay_start = max(0, self.replay_prompt_length - (operation.positions == "last"))
                    effective_positions = "all"
                selected = changed if replay_start is None else changed[..., replay_start:, :]
                direction = operation.direction.to(
                    device=changed.device, dtype=changed.dtype
                )
                pre_projection: torch.Tensor | None = None
                if self.collect_diagnostics:
                    unit = unit_direction(direction).to(changed)
                    pre_projection = torch.einsum("...d,d->...", changed, unit)
                if operation.operation == "add":
                    assert operation.delta is not None
                    updated = apply_addition(
                        selected,
                        operation.delta,
                        positions=effective_positions,
                        preserve_original_activation_norm=operation.preserve_original_activation_norm,
                    )
                else:
                    assert (
                        operation.bound is not None and operation.threshold is not None
                    )
                    updated = apply_cap(
                        selected,
                        direction,
                        bound=operation.bound,
                        threshold=operation.threshold,
                        positions=effective_positions,
                    )
                if replay_start is None:
                    changed = updated
                else:
                    changed = changed.clone()
                    changed[..., replay_start:, :] = updated
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
            self._capture_tensor(changed, layer, "post_intervention")
            return _replace_primary(output, changed, descriptor)

        return hook

    def __enter__(self) -> Self:
        if not self.interventions and self.activation_capture is None:
            return self
        blocks, _ = locate_transformer_blocks(self.model)
        grouped: dict[int, list[ResolvedIntervention]] = {}
        for intervention in self.interventions:
            if not 0 <= intervention.layer < len(blocks):
                raise IndexError(
                    f"layer {intervention.layer} out of range for {len(blocks)} transformer blocks"
                )
            grouped.setdefault(intervention.layer, []).append(intervention)
        if self.activation_capture is not None:
            for layer in self.activation_capture.layers:
                if not 0 <= layer < len(blocks):
                    raise IndexError(
                        f"activation capture layer {layer} out of range for {len(blocks)} transformer blocks"
                    )
                grouped.setdefault(layer, [])
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
