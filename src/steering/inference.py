"""High-level Hugging Face generation with persona-vector interventions."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed,
)

from .capture import ActivationCaptureConfig, ActivationCaptureEvent
from .hooks import SteeringContext
from .interventions import (
    CappingRequest,
    ResolvedIntervention,
    SteeringPlan,
    SteeringRequest,
    unit_direction,
)
from .vector_store import VectorStore

DEFAULT_MODEL_ID = "Qwen/Qwen3-32B-FP8"


@dataclass(frozen=True)
class GenerationResponse:
    text: str
    generated_token_ids: list[int]
    metadata: dict[str, Any]
    projections: list[dict[str, Any]]
    activations: list[ActivationCaptureEvent] = field(default_factory=list)


def _load_calibration(
    value: str | Path | Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    if value is None:
        raise ValueError("this intervention requires calibration metadata")
    if isinstance(value, Mapping):
        return value
    with Path(value).open(encoding="utf-8") as handle:
        return json.load(handle)


def _layer_calibration(
    calibration: Mapping[str, Any], layer: int, recipe_hash: str
) -> Mapping[str, Any]:
    try:
        layer_data = calibration["layers"][str(layer)]
    except KeyError as error:
        raise ValueError(f"calibration has no statistics for layer {layer}") from error
    directions = layer_data.get("directions", {})
    if recipe_hash not in directions:
        raise ValueError(
            f"calibration layer {layer} has no direction matching recipe hash {recipe_hash}"
        )
    return {"layer": layer_data, "direction": directions[recipe_hash]}


def _calibration_layer(calibration: Mapping[str, Any], layer: int) -> Mapping[str, Any]:
    try:
        return calibration["layers"][str(layer)]
    except KeyError as error:
        raise ValueError(f"calibration has no statistics for layer {layer}") from error


def _calibration_provenance(
    source: str | Path | Mapping[str, Any] | None,
    calibration: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if calibration is None:
        return None
    return {
        "source": str(source) if not isinstance(source, Mapping) else "inline",
        "dataset": calibration.get("dataset"),
        "model_id": calibration.get("model_id"),
        "model_revision": calibration.get("model_revision"),
        "token_policy": calibration.get("token_policy"),
        "reproduction_approximation": calibration.get("reproduction_approximation"),
    }


def _resolve_quantile(data: Mapping[str, Any], quantile: float) -> float:
    quantiles = data.get("quantiles", {})
    candidates = (str(quantile), f"{quantile:g}", f"{quantile:.6f}")
    for key in candidates:
        if key in quantiles:
            return float(quantiles[key])
    numeric = {float(key): float(value) for key, value in quantiles.items()}
    for key, value in numeric.items():
        if abs(key - quantile) < 1e-12:
            return value
    raise ValueError(
        f"quantile {quantile} was not computed; available: {sorted(numeric)}"
    )


class InferenceRunner:
    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        *,
        vector_store: VectorStore,
        model_id: str = DEFAULT_MODEL_ID,
        model_revision: str | None = None,
        tokenizer_revision: str | None = None,
        enable_thinking: bool = False,
        quantization: str | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.vector_store = vector_store
        self.model_id = model_id
        self.model_revision = model_revision
        self.tokenizer_revision = tokenizer_revision
        self.enable_thinking = enable_thinking
        self.quantization = quantization

    @classmethod
    def from_pretrained(
        cls,
        model_id: str = DEFAULT_MODEL_ID,
        *,
        model_revision: str | None = None,
        tokenizer_revision: str | None = None,
        device_map: str | Mapping[str, Any] | None = "auto",
        dtype: torch.dtype | str = torch.bfloat16,
        quantization: str | None = None,
        enable_thinking: bool = False,
        trust_remote_code: bool = False,
        vector_store: VectorStore | None = None,
        **model_kwargs: Any,
    ) -> InferenceRunner:
        quantization_config = None
        if quantization is not None:
            if quantization == "4bit":
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
            elif quantization == "8bit":
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            else:
                raise ValueError("quantization must be None, '4bit', or '8bit'")
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            revision=tokenizer_revision or model_revision,
            trust_remote_code=trust_remote_code,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=model_revision,
            device_map=device_map,
            torch_dtype=dtype,
            quantization_config=quantization_config,
            trust_remote_code=trust_remote_code,
            **model_kwargs,
        )
        store = vector_store or VectorStore(model_id=model_id)
        return cls(
            model,
            tokenizer,
            vector_store=store,
            model_id=model_id,
            model_revision=model_revision,
            tokenizer_revision=tokenizer_revision,
            enable_thinking=enable_thinking,
            quantization=quantization,
        )

    def _layers(self, request: SteeringRequest) -> tuple[int, ...]:
        total = int(self.model.config.num_hidden_layers)
        layers = request.layers or (total // 2,)
        for layer in layers:
            if not 0 <= layer < total:
                raise IndexError(f"layer {layer} out of range for {total} layers")
        return layers

    def _resolve_additive(self, request: SteeringRequest) -> list[ResolvedIntervention]:
        materialized = self.vector_store.materialize(
            request.vector, model_config=self.model.config
        )
        calibration = (
            _load_calibration(request.calibration)
            if request.scaling in {"average_residual_norm", "projection_std"}
            else None
        )
        result = []
        for layer in self._layers(request):
            coefficient = request.coefficient_for(layer)
            if coefficient == 0.0:
                continue
            vector = materialized.tensor[layer].to(torch.float32)
            unit = unit_direction(vector)
            if request.scaling == "raw":
                delta = coefficient * vector
                scale_value = 1.0
            elif request.scaling == "unit":
                delta = coefficient * unit
                scale_value = 1.0
            else:
                assert calibration is not None
                if request.scaling == "average_residual_norm":
                    layer_stats = _calibration_layer(calibration, layer)
                    scale_value = float(layer_stats["residual_norm"]["mean"])
                else:
                    stats = _layer_calibration(
                        calibration, layer, materialized.recipe_hash
                    )
                    scale_value = float(stats["direction"]["std"])
                delta = coefficient * scale_value * unit
            result.append(
                ResolvedIntervention(
                    operation="add",
                    layer=layer,
                    direction=vector,
                    delta=delta,
                    positions=request.positions,
                    recipe_hash=materialized.recipe_hash,
                    preserve_original_activation_norm=request.preserve_original_activation_norm,
                    metadata={
                        **materialized.metadata(),
                        "coefficient": coefficient,
                        "scaling": request.scaling,
                        "calibration_scale": scale_value,
                        "positions": request.positions,
                        "preserve_original_activation_norm": request.preserve_original_activation_norm,
                        "non_paper_norm_preservation": request.preserve_original_activation_norm,
                        "calibration": _calibration_provenance(
                            request.calibration, calibration
                        ),
                    },
                )
            )
        return result

    def _resolve_cap(self, request: CappingRequest) -> list[ResolvedIntervention]:
        materialized = self.vector_store.materialize(
            request.direction, model_config=self.model.config
        )
        calibration = (
            _load_calibration(request.calibration)
            if request.quantile is not None
            else None
        )
        result = []
        total = int(self.model.config.num_hidden_layers)
        for layer in request.layers:
            if not 0 <= layer < total:
                raise IndexError(f"layer {layer} out of range for {total} layers")
            if request.quantile is not None:
                assert calibration is not None
                stats = _layer_calibration(calibration, layer, materialized.recipe_hash)
                if isinstance(request.quantile, tuple):
                    threshold: float | tuple[float, float] = tuple(
                        _resolve_quantile(stats["direction"], value)
                        for value in request.quantile
                    )
                else:
                    threshold = _resolve_quantile(
                        stats["direction"], float(request.quantile)
                    )
            else:
                threshold = request.threshold_for(layer)
            result.append(
                ResolvedIntervention(
                    operation="cap",
                    layer=layer,
                    direction=materialized.tensor[layer],
                    positions=request.positions,
                    recipe_hash=materialized.recipe_hash,
                    bound=request.bound,
                    threshold=threshold,
                    metadata={
                        **materialized.metadata(),
                        "bound": request.bound,
                        "threshold": threshold,
                        "quantile": request.quantile,
                        "positions": request.positions,
                        "calibration": _calibration_provenance(
                            request.calibration, calibration
                        ),
                    },
                )
            )
        return result

    def resolve(
        self, steering: SteeringRequest | CappingRequest | SteeringPlan | None
    ) -> list[ResolvedIntervention]:
        if steering is None:
            return []
        if isinstance(steering, SteeringRequest):
            plan = SteeringPlan(additive=[steering])
        elif isinstance(steering, CappingRequest):
            plan = SteeringPlan(caps=[steering])
        elif isinstance(steering, SteeringPlan):
            plan = steering
        else:
            raise TypeError(
                f"unsupported steering configuration: {type(steering).__name__}"
            )
        additive = [op for item in plan.additive for op in self._resolve_additive(item)]
        caps = [op for item in plan.caps for op in self._resolve_cap(item)]
        return additive + caps if plan.ordering == "add_then_cap" else caps + additive

    def _tokenize(
        self, conversation: Sequence[Mapping[str, str]]
    ) -> Mapping[str, torch.Tensor]:
        template_kwargs: dict[str, Any] = {
            "tokenize": True,
            "add_generation_prompt": True,
            "return_tensors": "pt",
            "return_dict": True,
        }
        if self.model_id.lower().startswith("qwen/"):
            template_kwargs["enable_thinking"] = self.enable_thinking
        encoded = self.tokenizer.apply_chat_template(conversation, **template_kwargs)
        device = self.model.get_input_embeddings().weight.device
        return {key: value.to(device) for key, value in encoded.items()}

    def generate(
        self,
        conversation: Sequence[Mapping[str, str]],
        *,
        steering: SteeringRequest | CappingRequest | SteeringPlan | None = None,
        seed: int = 0,
        collect_diagnostics: bool = False,
        activation_capture: ActivationCaptureConfig | None = None,
        **generation_kwargs: Any,
    ) -> GenerationResponse:
        interventions = self.resolve(steering)
        set_seed(seed)
        inputs = self._tokenize(conversation)
        prompt_ids = inputs["input_ids"][0].detach().cpu().tolist()
        prompt_length = len(prompt_ids)
        defaults = {"max_new_tokens": 128}
        defaults.update(generation_kwargs)
        self.model.eval()
        context = SteeringContext(
            self.model,
            interventions,
            collect_diagnostics=collect_diagnostics,
            activation_capture=activation_capture,
        )
        with torch.inference_mode(), context:
            output_ids = self.model.generate(**inputs, **defaults)
        generated = output_ids[0, prompt_length:].detach().cpu().tolist()
        text = self.tokenizer.decode(generated, skip_special_tokens=True)

        def token_strings(token_ids: list[int]) -> list[str] | None:
            converter = getattr(self.tokenizer, "convert_ids_to_tokens", None)
            if not callable(converter):
                return None
            try:
                return list(converter(token_ids))
            except (TypeError, ValueError):
                return None

        metadata = {
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "tokenizer_revision": self.tokenizer_revision,
            "enable_thinking": self.enable_thinking,
            "quantization": self.quantization,
            "model_dtype": str(next(self.model.parameters()).dtype),
            "seed": seed,
            "generation": defaults,
            "conversation": list(conversation),
            "tokens": {
                "prompt_ids": prompt_ids,
                "prompt_tokens": token_strings(prompt_ids),
                "generated_ids": generated,
                "generated_tokens": token_strings(generated),
            },
            "activation_capture": (
                activation_capture.to_dict() if activation_capture is not None else None
            ),
            "interventions": [
                {
                    "operation": operation.operation,
                    "layer": operation.layer,
                    **dict(operation.metadata),
                }
                for operation in interventions
            ],
        }
        projections = [asdict(item) for item in context.diagnostics]
        return GenerationResponse(
            text, generated, metadata, projections, context.activation_events
        )
