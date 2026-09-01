"""Streaming calibration of residual norms and direction projections."""

from __future__ import annotations

import json
import math
import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from .hooks import _extract_primary, locate_transformer_blocks
from .recipes import VectorRecipe
from .vector_store import VectorStore


@dataclass
class RunningStats:
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def update(self, values: torch.Tensor) -> None:
        values = values.detach().to(torch.float64).flatten()
        if not values.numel():
            return
        batch_count = int(values.numel())
        batch_mean = float(values.mean())
        batch_m2 = float(torch.sum((values - batch_mean) ** 2))
        if self.count == 0:
            self.count, self.mean, self.m2 = batch_count, batch_mean, batch_m2
            return
        total = self.count + batch_count
        delta = batch_mean - self.mean
        self.m2 += batch_m2 + delta * delta * self.count * batch_count / total
        self.mean += delta * batch_count / total
        self.count = total

    @property
    def variance(self) -> float:
        return self.m2 / self.count if self.count else float("nan")

    @property
    def std(self) -> float:
        return math.sqrt(self.variance)

    def to_dict(self) -> dict[str, Any]:
        return {"count": self.count, "mean": self.mean, "std": self.std}


@dataclass
class Reservoir:
    capacity: int
    seed: int
    seen: int = 0
    samples: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.capacity <= 0:
            raise ValueError("reservoir capacity must be positive")
        self._random = random.Random(self.seed)

    def update(self, values: torch.Tensor) -> None:
        for value in values.detach().to(torch.float32).flatten().cpu().tolist():
            self.seen += 1
            if len(self.samples) < self.capacity:
                self.samples.append(float(value))
                continue
            replacement = self._random.randrange(self.seen)
            if replacement < self.capacity:
                self.samples[replacement] = float(value)

    def quantiles(self, requested: Sequence[float]) -> dict[str, float]:
        if not self.samples:
            return {}
        values = torch.tensor(self.samples, dtype=torch.float64)
        return {
            f"{quantile:g}": float(torch.quantile(values, float(quantile)))
            for quantile in requested
        }


@dataclass
class DirectionAccumulator:
    stats: RunningStats
    reservoir: Reservoir


@dataclass
class LayerAccumulator:
    residual_norm: RunningStats
    directions: dict[str, DirectionAccumulator]


def _assistant_encoded(
    tokenizer: Any,
    messages: Sequence[Mapping[str, str]],
    *,
    enable_thinking: bool | None = None,
) -> tuple[Any, torch.Tensor]:
    """Tokenize a completed chat and return its assistant-token mask."""

    try:
        template_options = (
            {"enable_thinking": enable_thinking} if enable_thinking is not None else {}
        )
        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_tensors="pt",
            return_dict=True,
            return_assistant_tokens_mask=True,
            **template_options,
        )
        mask = encoded.get("assistant_masks")
        if mask is None:
            mask = encoded.get("assistant_mask")
        if mask is not None and torch.as_tensor(mask).any():
            return encoded, torch.as_tensor(mask, dtype=torch.bool)
    except (TypeError, ValueError):
        pass
    assistant_indices = [
        index
        for index, message in enumerate(messages)
        if message["role"] == "assistant"
    ]
    if assistant_indices != [len(messages) - 1]:
        raise ValueError(
            "The tokenizer chat template does not expose assistant masks; fallback requires exactly one final assistant message"
        )
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        return_tensors="pt",
        return_dict=True,
        **({"enable_thinking": enable_thinking} if enable_thinking is not None else {}),
    )
    prefix = tokenizer.apply_chat_template(
        messages[:-1],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        **({"enable_thinking": enable_thinking} if enable_thinking is not None else {}),
    )
    full_length = encoded["input_ids"].shape[-1]
    prefix_length = prefix["input_ids"].shape[-1]
    if prefix_length >= full_length:
        raise ValueError("unable to identify assistant response token span")
    mask = torch.zeros_like(encoded["input_ids"], dtype=torch.bool)
    mask[..., prefix_length:] = True
    return encoded, mask


class CalibrationRunner:
    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        vector_store: VectorStore,
        *,
        model_id: str,
        model_revision: str | None = None,
        tokenizer_revision: str | None = None,
        enable_thinking: bool = False,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.vector_store = vector_store
        self.model_id = model_id
        self.model_revision = model_revision
        self.tokenizer_revision = tokenizer_revision
        self.enable_thinking = enable_thinking

    def run(
        self,
        conversations: Iterable[Sequence[Mapping[str, str]]],
        *,
        layers: Sequence[int],
        directions: Sequence[VectorRecipe] = (),
        dataset: Mapping[str, Any],
        quantiles: Sequence[float] = (0.01, 0.25, 0.5, 0.75, 0.95, 0.99),
        reservoir_capacity: int = 100_000,
        seed: int = 0,
        max_conversations: int | None = None,
    ) -> dict[str, Any]:
        selected_layers = tuple(int(layer) for layer in layers)
        if not selected_layers:
            raise ValueError("calibration layers must not be empty")
        if any(not 0 <= value <= 1 for value in quantiles):
            raise ValueError("quantiles must be between zero and one")
        blocks, block_path = locate_transformer_blocks(self.model)
        if any(not 0 <= layer < len(blocks) for layer in selected_layers):
            raise IndexError("calibration layer out of range")
        materialized = [
            self.vector_store.materialize(recipe, model_config=self.model.config)
            for recipe in directions
        ]
        accumulators: dict[int, LayerAccumulator] = {}
        for layer in selected_layers:
            direction_data = {
                vector.recipe_hash: DirectionAccumulator(
                    RunningStats(),
                    Reservoir(reservoir_capacity, seed + layer * 1009 + index),
                )
                for index, vector in enumerate(materialized)
            }
            accumulators[layer] = LayerAccumulator(RunningStats(), direction_data)

        current_mask: torch.Tensor | None = None

        def make_hook(layer: int):
            def hook(
                module: torch.nn.Module, args: tuple[Any, ...], output: Any
            ) -> None:
                del module, args
                tensor, _ = _extract_primary(output)
                if tensor is None:
                    raise TypeError(f"unsupported block output at layer {layer}")
                assert current_mask is not None
                mask = current_mask.to(tensor.device)
                selected = tensor[mask].to(torch.float32)
                norms = torch.linalg.vector_norm(selected, dim=-1)
                accumulators[layer].residual_norm.update(norms)
                for vector in materialized:
                    unit = vector.tensor[layer].to(torch.float32)
                    unit = unit / torch.linalg.vector_norm(unit)
                    projections = selected @ unit.to(selected.device)
                    accumulator = accumulators[layer].directions[vector.recipe_hash]
                    accumulator.stats.update(projections)
                    accumulator.reservoir.update(projections)

            return hook

        handles = [
            blocks[layer].register_forward_hook(make_hook(layer))
            for layer in selected_layers
        ]
        conversation_count = 0
        response_count = 0
        self.model.eval()
        try:
            with torch.inference_mode():
                for messages in conversations:
                    if (
                        max_conversations is not None
                        and conversation_count >= max_conversations
                    ):
                        break
                    encoded, mask = _assistant_encoded(
                        self.tokenizer,
                        messages,
                        enable_thinking=(
                            self.enable_thinking
                            if self.model_id.lower().startswith("qwen/")
                            else None
                        ),
                    )
                    attention = encoded.get("attention_mask")
                    if attention is not None:
                        mask &= attention.to(dtype=torch.bool)
                    current_mask = mask
                    device = self.model.get_input_embeddings().weight.device
                    inputs = {
                        key: value.to(device)
                        for key, value in encoded.items()
                        if key in {"input_ids", "attention_mask", "token_type_ids"}
                    }
                    self.model(**inputs, use_cache=False)
                    conversation_count += 1
                    response_count += sum(
                        message["role"] == "assistant" for message in messages
                    )
        finally:
            for handle in handles:
                handle.remove()

        layer_records: dict[str, Any] = {}
        for layer, accumulator in accumulators.items():
            layer_records[str(layer)] = {
                "residual_norm": accumulator.residual_norm.to_dict(),
                "directions": {
                    vector.recipe_hash: {
                        **accumulator.directions[vector.recipe_hash].stats.to_dict(),
                        "quantiles": accumulator.directions[
                            vector.recipe_hash
                        ].reservoir.quantiles(quantiles),
                        "quantile_sample_count": len(
                            accumulator.directions[vector.recipe_hash].reservoir.samples
                        ),
                        "quantiles_exact": accumulator.directions[
                            vector.recipe_hash
                        ].reservoir.seen
                        <= reservoir_capacity,
                        "recipe": vector.recipe.to_dict(),
                    }
                    for vector in materialized
                },
            }
        token_counts = [
            record["residual_norm"]["count"] for record in layer_records.values()
        ]
        return {
            "schema_version": 1,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "tokenizer_revision": self.tokenizer_revision,
            "enable_thinking": self.enable_thinking,
            "model_dtype": str(next(self.model.parameters()).dtype),
            "accumulation_dtype": "torch.float64",
            "block_path": block_path,
            "token_policy": "assistant_response_non_padding",
            "dataset": dict(dataset),
            "paper_scale_exact": dataset.get("id") == "lmsys/lmsys-chat-1m",
            "reproduction_approximation": dataset.get("id") != "lmsys/lmsys-chat-1m",
            "seed": seed,
            "reservoir_capacity": reservoir_capacity,
            "requested_quantiles": list(quantiles),
            "counts": {
                "conversations": conversation_count,
                "responses": response_count,
                "tokens": token_counts[0] if token_counts else 0,
            },
            "directions": [vector.metadata() for vector in materialized],
            "layers": layer_records,
        }


def iter_jsonl_conversations(
    path: str | Path, *, messages_column: str = "messages"
) -> Iterable[Sequence[Mapping[str, str]]]:
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            messages = record.get(messages_column)
            if not isinstance(messages, list):
                raise TypeError(
                    f"{path}:{line_number} has no list column {messages_column!r}"
                )
            yield messages


def save_calibration(record: Mapping[str, Any], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(output)
