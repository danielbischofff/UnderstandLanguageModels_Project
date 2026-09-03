import json

import pytest
import torch
from safetensors.torch import load_file

from steering.capture import ActivationCaptureEvent
from steering.experiments import (
    ExperimentConfig,
    ExperimentRecorder,
    canonical_hash,
    run_experiment,
)
from steering.inference import GenerationResponse


def config():
    return {
        "name": "point-of-view",
        "model": {
            "id": "Qwen/Qwen3-32B",
            "dtype": "bfloat16",
            "enable_thinking": False,
        },
        "questions": [
            {"id": "q1", "text": "first"},
            {"id": "q2", "text": "second"},
        ],
        "characters": ["unsteered", "demon"],
        "steering": {"coefficient": 1, "layers": [2]},
        "generation": {"max_new_tokens": 4, "do_sample": False},
        "seed": 7,
        "j_lens": {"top_k": 3},
    }


def test_config_expands_question_character_matrix_in_stable_order():
    parsed = ExperimentConfig.from_mapping(config())
    trials = parsed.trials()
    assert [(item["question"]["id"], item["character"]) for item in trials] == [
        ("q1", "unsteered"),
        ("q1", "demon"),
        ("q2", "unsteered"),
        ("q2", "demon"),
    ]
    assert trials[0]["steering"] is None
    assert trials[1]["steering"] == {"coefficient": 1.0, "layers": [2]}
    assert canonical_hash(trials[1]) == canonical_hash(parsed.trials()[1])
    assert parsed.model["id"] == "Qwen/Qwen3-32B"
    assert parsed.model["dtype"] == "bfloat16"


def test_config_rejects_unknown_fields_and_duplicate_ids():
    invalid = config()
    invalid["capture"] = {}
    with pytest.raises(ValueError, match="unknown experiment fields"):
        ExperimentConfig.from_mapping(invalid)
    invalid = config()
    invalid["questions"][1]["id"] = "q1"
    with pytest.raises(ValueError, match="question ids must be unique"):
        ExperimentConfig.from_mapping(invalid)


def test_config_requires_model_and_validates_output_options():
    invalid = config()
    del invalid["model"]
    with pytest.raises(TypeError, match="model must be a mapping"):
        ExperimentConfig.from_mapping(invalid)
    invalid = config()
    invalid["outputs"] = {"keep_activations": "yes"}
    with pytest.raises(TypeError, match="keep_activations must be a boolean"):
        ExperimentConfig.from_mapping(invalid)


class FakeRunner:
    def __init__(self):
        self.generated = []
        self.captured = []

    def generate(self, conversation, *, steering, **kwargs):
        self.generated.append((conversation, steering, kwargs))
        return GenerationResponse(
            text="answer",
            generated_token_ids=[12],
            metadata={"tokens": {"prompt_ids": [10, 11]}},
            projections=[],
        )

    def capture_complete_sequence(
        self, conversation, generated_token_ids, *, steering, layers
    ):
        self.captured.append((conversation, generated_token_ids, steering, layers))
        return ["temporary activation"]

    def j_lens_top_k(self, *args):
        raise AssertionError("the fake lens should not execute the readout")


class FakeLens:
    source_layers = (0, 1)

    def read(self, events, *, prompt_length, device_readout):
        assert events == ["temporary activation"]
        assert prompt_length == 2
        assert callable(device_readout)
        return {"tokens": [], "layers": []}


def test_run_records_compact_matrix_without_activation_artifacts(tmp_path):
    recorder = ExperimentRecorder.create(config(), root=tmp_path)
    runner = FakeRunner()
    run_experiment(runner, recorder, FakeLens())

    results = json.loads(
        (recorder.run_dir / "experiment-results.json").read_text(encoding="utf-8")
    )
    assert len(results["trials"]) == 4
    assert (recorder.run_dir / "report.html").is_file()
    assert [trial["character"] for trial in results["trials"]] == [
        "unsteered",
        "demon",
        "unsteered",
        "demon",
    ]
    assert not list(recorder.run_dir.rglob("*.safetensors"))
    assert not list(recorder.run_dir.rglob("activations"))
    assert runner.generated[0][1] is None
    assert runner.generated[1][1].vector.name == "demon"
    assert recorder.manifest()["status"] == "complete"


def test_optional_activation_retention_uses_safetensors(tmp_path):
    value = config()
    value["outputs"] = {"keep_activations": True}
    recorder = ExperimentRecorder.create(value, root=tmp_path)
    response = GenerationResponse(
        "answer", [12], {"tokens": {"prompt_ids": [10, 11]}}, []
    )
    event = ActivationCaptureEvent(
        tensor=torch.tensor([[[1.0, 2.0]]]),
        layer=0,
        point="post_intervention",
        phase="prefill",
        forward_index=0,
        input_token_ids=[[10]],
        cache_positions=None,
        position_policy="all",
    )
    recorder.record_result(
        0,
        recorder.config.trials()[0],
        response,
        {"tokens": [], "layers": []},
        [event],
    )
    activation_dir = next((recorder.run_dir / "trials").iterdir()) / "activations"
    assert torch.equal(
        load_file(activation_dir / "activations.safetensors")["event_000000"],
        event.tensor,
    )
def test_resume_uses_resolved_experiment(tmp_path):
    recorder = ExperimentRecorder.create(config(), root=tmp_path)
    resumed = ExperimentRecorder.resume(recorder.run_dir)
    assert resumed.config == recorder.config
    with resumed:
        assert "pid=" in resumed.lock_path.read_text(encoding="utf-8")


def test_failed_cells_are_recorded_and_reported(tmp_path):
    class FailingRunner(FakeRunner):
        def generate(self, *args, **kwargs):
            raise RuntimeError("generation failed")

    value = config()
    value["questions"] = [value["questions"][0]]
    value["characters"] = ["unsteered"]
    recorder = ExperimentRecorder.create(value, root=tmp_path)
    with pytest.raises(RuntimeError, match="1 of 1 experiment trials failed"):
        run_experiment(FailingRunner(), recorder, FakeLens())
    assert recorder.manifest()["status"] == "complete_with_failures"
    failure = next(iter(recorder.manifest()["trials"].values()))
    assert failure["error"] == "RuntimeError: generation failed"
