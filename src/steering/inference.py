"""High-level Hugging Face generation with persona-vector interventions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
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
        local_files_only: bool = False,
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
            local_files_only=local_files_only,
        )
        load_kwargs = {
            "revision": model_revision,
            "device_map": device_map,
            "dtype": dtype,
            "trust_remote_code": trust_remote_code,
            "local_files_only": local_files_only,
            **model_kwargs,
        }
        # Do not pass an absent override: pre-quantized checkpoints may carry
        # their quantization config in config.json, and explicit None replaces it.
        # Transformers handles compatible merging and rejects conflicting configs.
        if quantization_config is not None:
            load_kwargs["quantization_config"] = quantization_config
        model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
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
        result = []
        for layer in self._layers(request):
            coefficient = request.coefficient_for(layer)
            if coefficient == 0.0:
                continue
            vector = materialized.tensor[layer].to(torch.float32)
            unit = unit_direction(vector)
            delta = coefficient * (vector if request.scaling == "raw" else unit)
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
                        "positions": request.positions,
                        "preserve_original_activation_norm": request.preserve_original_activation_norm,
                        "non_paper_norm_preservation": request.preserve_original_activation_norm,
                    },
                )
            )
        return result

    def _resolve_cap(self, request: CappingRequest) -> list[ResolvedIntervention]:
        materialized = self.vector_store.materialize(
            request.direction, model_config=self.model.config
        )
        result = []
        total = int(self.model.config.num_hidden_layers)
        for layer in request.layers:
            if not 0 <= layer < total:
                raise IndexError(f"layer {layer} out of range for {total} layers")
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
                        "positions": request.positions,
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

    def capture_complete_sequence(
        self,
        conversation: Sequence[Mapping[str, str]],
        generated_token_ids: Sequence[int],
        *,
        steering: SteeringRequest | CappingRequest | SteeringPlan | None,
        layers: Sequence[int],
    ) -> list[ActivationCaptureEvent]:
        """Capture one teacher-forced prompt+response pass for J-Lens.

        This second causal forward pass is equivalent to cached decoding at each
        position for ``positions='all'`` steering, but includes the final sampled
        token and avoids retaining every intermediate generation forward call.
        """

        inputs = self._tokenize(conversation)
        input_ids = inputs["input_ids"]
        generated = torch.tensor(
            [list(generated_token_ids)], device=input_ids.device, dtype=input_ids.dtype
        )
        full_input_ids = torch.cat((input_ids, generated), dim=1)
        full_inputs: dict[str, torch.Tensor] = {"input_ids": full_input_ids}
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            generated_mask = torch.ones(
                (attention_mask.shape[0], generated.shape[1]),
                device=attention_mask.device,
                dtype=attention_mask.dtype,
            )
            full_inputs["attention_mask"] = torch.cat(
                (attention_mask, generated_mask), dim=1
            )

        capture = ActivationCaptureConfig(
            layers=list(layers),
            points=["post_intervention"],
            positions="all",
            dtype="bfloat16",
        )
        context = SteeringContext(
            self.model,
            self.resolve(steering),
            collect_diagnostics=False,
            activation_capture=capture,
        )
        self.model.eval()
        with torch.inference_mode(), context:
            self.model(**full_inputs, use_cache=False)
        return context.activation_events

    def j_lens_top_k(
        self,
        jacobian: torch.Tensor,
        jacobian_scale: float,
        activations: torch.Tensor,
        top_k_count: int,
        allowed: Any,
    ) -> tuple[Any, Any, Any]:
        """Transport and unembed a layer batch on the model device."""

        base_model = getattr(self.model, "model", None)
        final_norm = getattr(base_model, "norm", None)
        lm_head = self.model.get_output_embeddings()
        if final_norm is None or lm_head is None:
            raise ValueError("J-Lens device readout requires model.norm and lm_head")
        try:
            norm_device = next(final_norm.parameters()).device
            head_device = next(lm_head.parameters()).device
        except StopIteration as error:
            raise ValueError("J-Lens norm and lm_head must have parameters") from error
        compute_dtype = torch.bfloat16 if norm_device.type == "cuda" else torch.float32
        vectors = activations.to(device=norm_device, dtype=compute_dtype)
        transport = jacobian.to(device=norm_device, dtype=compute_dtype)
        transported = (vectors @ transport.T) * float(jacobian_scale)
        normalized = final_norm(transported).to(head_device)
        logits = lm_head(normalized).to(torch.float32)
        allowed_mask = torch.as_tensor(allowed, device=head_device, dtype=torch.bool)
        ranked = logits.masked_fill(~allowed_mask.unsqueeze(0), -torch.inf)
        scores, ids = torch.topk(ranked, top_k_count, dim=1)
        probabilities = torch.exp(scores - torch.logsumexp(logits, dim=1, keepdim=True))
        return (
            ids.detach().cpu().numpy(),
            scores.detach().cpu().numpy(),
            probabilities.detach().cpu().numpy(),
        )
