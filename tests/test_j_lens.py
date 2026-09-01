import json
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import torch
from safetensors.torch import save_file

from steering import j_lens


def _artifact(tmp_path):
    directory = tmp_path / "activations"
    directory.mkdir()
    save_file(
        {
            "event_000000": torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]),
            "event_000001": torch.tensor([[[2.0, 4.0], [6.0, 8.0]]]),
        },
        directory / "activations.safetensors",
    )
    events = []
    for index, point in enumerate(["pre_intervention", "post_intervention"]):
        events.append(
            {
                "tensor_key": f"event_{index:06d}",
                "layer": 2,
                "point": point,
                "phase": "prefill",
                "forward_index": 0,
                "input_token_ids": [[10, 11]],
                "cache_positions": [0, 1],
                "position_policy": "all",
                "shape": [1, 2, 2],
                "dtype": "torch.float32",
            }
        )
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "tensor_file": "activations.safetensors",
                "events": events,
            }
        )
    )
    return directory


def test_quick_j_lens_reuses_backend_and_adds_pre_post_delta(tmp_path, monkeypatch):
    observed = {}

    class Lens:
        def __init__(self, path):
            self.source_layers = [2]
            self.n_prompts = 80
            observed["lens_path"] = path

        def transport(self, layer, matrix):
            observed["layer"] = layer
            observed["matrix"] = matrix.copy()
            return matrix

    class Unembedding:
        vocab_size = 4

        def __init__(self, data_dir):
            observed["data_dir"] = data_dir

        def logits(self, matrix):
            return np.column_stack([matrix, matrix.sum(axis=1), -matrix.sum(axis=1)])

    def top_k(logits, count, allowed=None):
        scores = logits if allowed is None else np.where(allowed, logits, -np.inf)
        ids = np.argsort(scores, axis=1)[:, ::-1][:, :count]
        selected = np.take_along_axis(logits, ids, axis=1)
        probabilities = np.full_like(selected, 0.25)
        return ids, selected, probabilities

    @contextmanager
    def backend(_source):
        yield (
            SimpleNamespace(LensTransport=Lens),
            SimpleNamespace(Unembedding=Unembedding, top_k=top_k),
            SimpleNamespace(
                decode_vocab=lambda tokenizer, size: (
                    ["zero", "one", "two", "three"],
                    np.array([True, True, True, True]),
                )
            ),
        )

    monkeypatch.setattr(j_lens, "_analysis_imports", backend)
    monkeypatch.setattr(
        j_lens.Tokenizer, "from_file", staticmethod(lambda path: object())
    )
    result = j_lens.quick_j_lens_tokens(
        _artifact(tmp_path),
        analysis_source=tmp_path / "source",
        data_dir=tmp_path / "data",
        lens_path=tmp_path / "lens.pt",
        layer=2,
        top_k_count=2,
    )
    assert [row["label"] for row in result["readouts"]] == [
        "pre_intervention",
        "post_intervention",
        "post_minus_pre",
    ]
    assert np.array_equal(
        observed["matrix"],
        np.array([[3.0, 4.0], [6.0, 8.0], [3.0, 4.0]], dtype=np.float32),
    )
    assert result["selection"]["input_token_id"] == 11
    assert result["selection"]["cache_position"] == 1
