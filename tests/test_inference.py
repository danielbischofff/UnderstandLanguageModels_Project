from types import SimpleNamespace

import numpy as np
import pytest
import torch

import steering.inference as inference_module
from steering.inference import InferenceRunner
from steering.interventions import CappingRequest, SteeringPlan, SteeringRequest
from steering.recipes import PublishedAssistantAxis, RoleDirection
from steering.vector_store import MaterializedVector


class Store:
    def materialize(self, recipe, model_config=None):
        tensor = torch.tensor([[1.0, 0.0]] * model_config.num_hidden_layers)
        return MaterializedVector(recipe, tensor, (), tuple([1.0] * len(tensor)))


class Model(torch.nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.config = SimpleNamespace(num_hidden_layers=layers, hidden_size=2)
        self.weight = torch.nn.Parameter(torch.ones(1))


def runner(layers=6):
    return InferenceRunner(
        Model(layers), object(), vector_store=Store(), model_id="test"
    )


@pytest.mark.parametrize(("layers", "expected"), [(6, 3), (5, 2)])
def test_middle_layer_default(layers, expected):
    resolved = runner(layers).resolve(
        SteeringRequest(RoleDirection("lawyer"), coefficient=1.0)
    )
    assert [item.layer for item in resolved] == [expected]


def test_zero_coefficient_produces_no_intervention():
    assert (
        runner().resolve(SteeringRequest(RoleDirection("lawyer"), coefficient=0.0))
        == []
    )


def test_raw_unit_and_per_layer_coefficients():
    resolved = runner().resolve(
        SteeringRequest(
            RoleDirection("lawyer"),
            coefficient={1: 2.0, 2: -1.0},
            layers=[1, 2],
            scaling="unit",
        )
    )
    assert torch.equal(resolved[0].delta, torch.tensor([2.0, 0.0]))
    assert torch.equal(resolved[1].delta, torch.tensor([-1.0, 0.0]))
    raw = runner().resolve(
        SteeringRequest(
            RoleDirection("lawyer"), coefficient=3.0, layers=[1], scaling="raw"
        )
    )
    assert torch.equal(raw[0].delta, torch.tensor([3.0, 0.0]))


def test_add_then_cap_order():
    recipe = PublishedAssistantAxis()
    plan = SteeringPlan(
        additive=[SteeringRequest(RoleDirection("lawyer"), layers=[4])],
        caps=[CappingRequest(recipe, "lower", [4], threshold=-2.0)],
    )
    resolved = runner().resolve(plan)
    assert [item.operation for item in resolved] == ["add", "cap"]
    assert resolved[1].threshold == -2.0


def test_loader_does_not_override_checkpoint_quantization_config(monkeypatch):
    observed = {}
    tokenizer_observed = {}
    model = Model(4)
    tokenizer = object()

    monkeypatch.setattr(
        inference_module.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: tokenizer_observed.update(kwargs) or tokenizer,
    )

    def load_model(*args, **kwargs):
        observed.update(kwargs)
        return model

    monkeypatch.setattr(
        inference_module.AutoModelForCausalLM, "from_pretrained", load_model
    )
    loaded = InferenceRunner.from_pretrained(
        "publisher/prequantized-model-with-arbitrary-name",
        dtype=torch.bfloat16,
        local_files_only=True,
        vector_store=Store(),
    )
    assert loaded.model is model
    assert observed["dtype"] is torch.bfloat16
    assert "torch_dtype" not in observed
    assert "quantization_config" not in observed
    assert observed["local_files_only"] is True
    assert tokenizer_observed["local_files_only"] is True


def test_explicit_quantization_is_forwarded_for_unquantized_checkpoint(monkeypatch):
    observed = {}
    quantization_config = object()

    monkeypatch.setattr(
        inference_module.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        inference_module,
        "BitsAndBytesConfig",
        lambda **kwargs: quantization_config,
    )

    def load_model(*args, **kwargs):
        observed.update(kwargs)
        return Model(4)

    monkeypatch.setattr(
        inference_module.AutoModelForCausalLM, "from_pretrained", load_model
    )
    InferenceRunner.from_pretrained(
        "publisher/ordinary-model", quantization="4bit", vector_store=Store()
    )
    assert observed["quantization_config"] is quantization_config


def test_complete_sequence_capture_includes_final_generated_token(monkeypatch):
    class Block(torch.nn.Module):
        def forward(self, hidden):
            return (hidden + 1,)

    class Inner(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([Block(), Block()])

    class CaptureModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Inner()
            self.config = SimpleNamespace(num_hidden_layers=2, hidden_size=2)
            self.weight = torch.nn.Parameter(torch.ones(1))

        def forward(self, input_ids, attention_mask=None, use_cache=False):
            hidden = torch.stack(
                [input_ids.float(), torch.zeros_like(input_ids).float()], dim=-1
            )
            for block in self.model.layers:
                hidden = block(hidden)[0]
            return hidden

    loaded = InferenceRunner(
        CaptureModel(), object(), vector_store=Store(), model_id="test"
    )
    monkeypatch.setattr(
        loaded,
        "_tokenize",
        lambda conversation: {
            "input_ids": torch.tensor([[10, 11]]),
            "attention_mask": torch.tensor([[1, 1]]),
        },
    )
    events = loaded.capture_complete_sequence(
        [{"role": "user", "content": "hello"}],
        [12, 13],
        steering=None,
        layers=[0, 1],
    )
    assert len(events) == 2
    assert events[0].input_token_ids == [[10, 11, 12, 13]]
    assert events[0].point == "post_intervention"
    assert events[0].tensor.dtype is torch.bfloat16


def test_j_lens_readout_uses_model_norm_and_head():
    class Base(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = torch.nn.RMSNorm(2)

    class ReadoutModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Base()
            self.lm_head = torch.nn.Linear(2, 4, bias=False)
            self.config = SimpleNamespace(num_hidden_layers=2, hidden_size=2)
            with torch.no_grad():
                self.lm_head.weight.copy_(
                    torch.tensor([[1.0, 0.0], [0.0, 1.0], [2.0, 0.0], [-1.0, 0.0]])
                )

        def get_output_embeddings(self):
            return self.lm_head

    loaded = InferenceRunner(
        ReadoutModel(), object(), vector_store=Store(), model_id="test"
    )
    ids, scores, probabilities = loaded.j_lens_top_k(
        torch.eye(2),
        1.0,
        torch.tensor([[1.0, 0.0]]),
        2,
        np.array([True, True, False, True]),
    )
    assert ids.tolist() == [[0, 1]]
    assert scores.shape == probabilities.shape == (1, 2)
    assert np.all(probabilities > 0)


@pytest.mark.parametrize("positions", ["all", "last", "generated_only"])
@pytest.mark.parametrize("method", ["additive", "projection_cap"])
def test_replay_matches_cached_causal_decoding(monkeypatch, positions, method):
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from steering.capture import ActivationCaptureConfig
    from steering.hooks import SteeringContext

    torch.manual_seed(0)
    model = Qwen3ForCausalLM(Qwen3Config(
        vocab_size=24, hidden_size=8, intermediate_size=16, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=2, head_dim=4,
    )).eval()

    class TinyStore:
        def materialize(self, recipe, model_config=None):
            tensor = torch.zeros(2, 8)
            tensor[:, 0] = 1
            return MaterializedVector(recipe, tensor, (), (1.0, 1.0))

    loaded = InferenceRunner(model, object(), vector_store=TinyStore(), model_id="test")
    prompt = torch.tensor([[1, 2, 3]])
    generated = [4, 5]
    monkeypatch.setattr(loaded, "_tokenize", lambda _: {"input_ids": prompt})
    recipe = RoleDirection("demon")
    request = (
        SteeringRequest(recipe, coefficient=0.5, layers=[0], positions=positions)
        if method == "additive" else
        CappingRequest(recipe, "lower", [0], threshold=0.5, positions=positions)
    )
    context = SteeringContext(model, loaded.resolve(request), activation_capture=ActivationCaptureConfig(
        layers=[0, 1], points=["post_intervention"], positions="all", dtype="bfloat16",
    ))
    with torch.inference_mode(), context:
        output = model(input_ids=prompt, use_cache=True)
        for token in generated:
            output = model(input_ids=torch.tensor([[token]]), past_key_values=output.past_key_values, use_cache=True)
    replay = loaded.capture_complete_sequence([], generated, steering=request, layers=[0, 1])
    for event in replay:
        cached = torch.cat([item.tensor for item in context.activation_events if item.layer == event.layer], dim=1)
        torch.testing.assert_close(event.tensor.float(), cached.float(), atol=0.002, rtol=0.01)
