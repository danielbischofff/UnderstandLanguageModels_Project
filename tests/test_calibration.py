import torch

from steering.calibration import CalibrationRunner, Reservoir, RunningStats


class CalibrationBlock(torch.nn.Module):
    def forward(self, hidden):
        return (hidden + 1,)


class CalibrationInner(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([CalibrationBlock(), CalibrationBlock()])


class CalibrationModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = CalibrationInner()
        self.embedding = torch.nn.Embedding(20, 3)
        self.config = type("Config", (), {"num_hidden_layers": 2, "hidden_size": 3})()

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        del attention_mask, use_cache
        hidden = self.embedding(input_ids)
        for block in self.model.layers:
            hidden = block(hidden)[0]
        return hidden


class CalibrationTokenizer:
    def apply_chat_template(
        self, messages, *, return_assistant_tokens_mask=False, **kwargs
    ):
        del kwargs
        length = len(messages) + 2
        result = {
            "input_ids": torch.arange(length).unsqueeze(0),
            "attention_mask": torch.ones(1, length, dtype=torch.long),
        }
        if return_assistant_tokens_mask:
            mask = torch.zeros(1, length, dtype=torch.long)
            mask[:, -2:] = 1
            result["assistant_masks"] = mask
        return result


def test_running_statistics_match_torch():
    stats = RunningStats()
    stats.update(torch.tensor([1.0, 2.0]))
    stats.update(torch.tensor([3.0, 4.0]))
    values = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64)
    assert stats.count == 4
    assert stats.mean == values.mean().item()
    assert stats.std == values.std(correction=0).item()


def test_reservoir_is_bounded_and_deterministic():
    first = Reservoir(10, 42)
    second = Reservoir(10, 42)
    values = torch.arange(100)
    first.update(values)
    second.update(values)
    assert first.samples == second.samples
    assert len(first.samples) == 10
    assert set(first.quantiles([0.25, 0.5])) == {"0.25", "0.5"}


def test_calibration_collects_only_assistant_masked_tokens():
    model = CalibrationModel()
    runner = CalibrationRunner(
        model,
        CalibrationTokenizer(),
        vector_store=object(),  # no direction recipes in this test
        model_id="test/model",
    )
    record = runner.run(
        [[{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]],
        layers=[0, 1],
        dataset={"id": "local"},
        reservoir_capacity=10,
    )
    assert record["counts"] == {"conversations": 1, "responses": 1, "tokens": 2}
    assert record["layers"]["0"]["residual_norm"]["count"] == 2
    assert record["reproduction_approximation"] is True
    assert not model.model.layers[0]._forward_hooks
