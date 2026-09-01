from types import SimpleNamespace

import pytest
import torch
from transformers.modeling_outputs import BaseModelOutput

from steering.hooks import SteeringContext, locate_transformer_blocks
from steering.interventions import ResolvedIntervention


class Block(torch.nn.Module):
    def forward(self, hidden):
        return (hidden + 1, "cache")


class Inner(torch.nn.Module):
    def __init__(self, count=3):
        super().__init__()
        self.layers = torch.nn.ModuleList(Block() for _ in range(count))


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = Inner()
        self.config = SimpleNamespace(num_hidden_layers=3, hidden_size=2)

    def forward(self, hidden, *, cache_position=None, past_key_values=None):
        value = hidden
        for block in self.model.layers:
            value = block(value)[0]
        return value


class ModelOutputBlock(torch.nn.Module):
    def forward(self, hidden):
        return BaseModelOutput(last_hidden_state=hidden, hidden_states=(hidden,))


def add_op(layer=1, positions="all", delta=None):
    return ResolvedIntervention(
        operation="add",
        layer=layer,
        direction=torch.tensor([1.0, 0.0]),
        delta=torch.tensor(delta or [2.0, 0.0]),
        positions=positions,
        recipe_hash="recipe",
    )


def test_block_location_and_tuple_preservation():
    model = FakeModel()
    blocks, path = locate_transformer_blocks(model)
    assert path == "model.layers"
    observed = []
    handle = blocks[0].register_forward_hook(lambda m, a, out: observed.append(out))
    model(torch.zeros(1, 2, 2))
    handle.remove()
    assert observed[0][1] == "cache"


def test_model_output_is_reconstructed_without_dropping_fields():
    model = FakeModel()
    model.model.layers[1] = ModelOutputBlock()
    with SteeringContext(model, [add_op()]):
        output = model.model.layers[1](torch.zeros(1, 1, 2))
    assert isinstance(output, BaseModelOutput)
    assert output.hidden_states is not None
    assert torch.equal(output.last_hidden_state, torch.tensor([[[2.0, 0.0]]]))


def test_hook_is_active_and_always_removed():
    model = FakeModel()
    baseline = model(torch.zeros(1, 2, 2))
    with SteeringContext(model, [add_op()]):
        changed = model(torch.zeros(1, 2, 2))
        assert not torch.equal(changed, baseline)
    assert not model.model.layers[1]._forward_hooks
    assert not model._forward_pre_hooks
    with pytest.raises(RuntimeError), SteeringContext(model, [add_op()]):
        raise RuntimeError("boom")
    assert not model.model.layers[1]._forward_hooks


def test_generated_only_skips_prefill_and_changes_decode():
    model = FakeModel()
    baseline = model(torch.zeros(1, 2, 2), cache_position=torch.tensor([0, 1]))
    with SteeringContext(model, [add_op(positions="generated_only")]):
        prefill = model(torch.zeros(1, 2, 2), cache_position=torch.tensor([0, 1]))
        decode = model(torch.zeros(1, 1, 2), cache_position=torch.tensor([2]))
    assert torch.equal(prefill, baseline)
    assert torch.equal(decode[..., 0], torch.tensor([[5.0]]))


def test_add_then_cap_order_is_deterministic():
    model = FakeModel()
    cap = ResolvedIntervention(
        operation="cap",
        layer=1,
        direction=torch.tensor([1.0, 0.0]),
        positions="all",
        recipe_hash="cap",
        bound="upper",
        threshold=2.5,
    )
    with SteeringContext(model, [add_op(delta=[10.0, 0.0]), cap]):
        changed = model(torch.zeros(1, 1, 2))
    # Cap is applied at layer 1, then the final block adds one.
    assert changed[..., 0].item() == pytest.approx(3.5)


def test_hook_uses_activation_dtype():
    model = FakeModel().to(dtype=torch.bfloat16)
    with SteeringContext(model, [add_op()]):
        changed = model(torch.zeros(1, 1, 2, dtype=torch.bfloat16))
    assert changed.dtype == torch.bfloat16
