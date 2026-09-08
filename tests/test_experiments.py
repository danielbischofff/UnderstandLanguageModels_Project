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
from steering.interventions import CappingRequest, CharacterBandRequest


def test_band_yaml_expands_thirteen_trials_and_round_trips():
    from pathlib import Path
    import yaml
    path = Path(__file__).resolve().parents[1] / "configs/experiments/tylenol-demon-8000mg-band-sweep.yaml"
    parsed = ExperimentConfig.from_mapping(yaml.safe_load(path.read_text()))
    assert len(parsed.trials()) == 13
    assert parsed.coefficients == (0.0, 0.1, 0.25, 0.5)
    assert parsed.schemes[2].layers == tuple(range(28, 54))
    assert ExperimentConfig.from_mapping(parsed.to_mapping()) == parsed
    request = parsed.schemes[2].request("demon", 0.0)
    assert isinstance(request, CharacterBandRequest)
    assert request.fraction == 0.0
    invalid = parsed.to_mapping()
    invalid["steering"]["coefficients"] = [-0.1]
    with pytest.raises(ValueError, match="nonnegative"):
        ExperimentConfig.from_mapping(invalid)


@pytest.mark.parametrize("fraction", [-0.1, float("inf"), float("nan")])
def test_invalid_band_fraction(fraction):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        CharacterBandRequest("demon", fraction, (0,))


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
    assert not (activation_dir.parent / "response.txt").exists()
    assert not (activation_dir.parent / "status.json").exists()
def test_resume_uses_resolved_experiment(tmp_path):
    recorder = ExperimentRecorder.create(config(), root=tmp_path)
    resumed = ExperimentRecorder.resume(recorder.run_dir)
    assert resumed.config == recorder.config
    with resumed:
        assert "pid=" in resumed.lock_path.read_text(encoding="utf-8")


def sweep_config():
    value = config()
    value["steering"] = {
        "coefficients": [-1, 0, 2],
        "schemes": [
            {"name": "middle", "layers": [2]},
            {"name": "late", "layers": [3, 4], "positions": "generated_only", "scaling": "unit"},
            {"name": "floor", "method": "projection_cap", "layers": [2], "bound": "lower"},
        ],
    }
    return value


def test_sweep_roundtrip_and_unique_cells():
    parsed = ExperimentConfig.from_mapping(sweep_config())
    assert ExperimentConfig.from_mapping(parsed.to_mapping()) == parsed
    trials = parsed.trials()
    assert len(trials) == 2 * (1 + 3 * 3)
    assert len({canonical_hash(trial) for trial in trials}) == len(trials)
    assert sum(trial["steering"] is None for trial in trials) == 2
    assert [(trial["steering"]["coefficient"], trial["steering"]["name"])
            for trial in trials[1:10]] == [
        (alpha, scheme) for alpha in [-1, 0, 2] for scheme in ["middle", "late", "floor"]
    ]


def test_sweep_execution_and_resume_preserve_all_axes(tmp_path):
    recorder = ExperimentRecorder.create(sweep_config(), root=tmp_path)
    runner = FakeRunner()
    run_experiment(runner, recorder, FakeLens())
    assert len(runner.generated) == 20
    for generated, captured in zip(runner.generated, runner.captured):
        assert generated[1] == captured[2]
    late = runner.generated[2][1]
    assert late.layers == (3, 4)
    assert late.positions == "generated_only"
    assert late.scaling == "unit"
    cap = runner.generated[3][1]
    assert isinstance(cap, CappingRequest)
    assert cap.threshold == -1
    assert cap.bound == "lower"
    results = json.loads((recorder.run_dir / "experiment-results.json").read_text())
    assert [item["steering"] for item in results["trials"]] == [
        item["steering"] for item in recorder.config.trials()
    ]
    resumed = ExperimentRecorder.resume(recorder.run_dir)
    run_experiment(runner, resumed, FakeLens())
    assert len(runner.generated) == 20  # Every completed cell is skipped.


def test_legacy_hash_shape_is_preserved():
    value = config()
    parsed = ExperimentConfig.from_mapping(value)
    assert parsed.to_mapping()["steering"] == value["steering"]
    assert parsed.trials()[1] == {
        "question": {"id": "q1", "text": "first"}, "character": "demon",
        "steering": {"coefficient": 1.0, "layers": [2]},
        "generation": value["generation"], "seed": 7,
    }


def test_position_restricted_sweeps_require_cached_generation():
    value = sweep_config()
    value["generation"]["use_cache"] = False
    with pytest.raises(ValueError, match="require generation.use_cache"):
        ExperimentConfig.from_mapping(value)


@pytest.mark.parametrize("steering, message", [
    ({"coefficients": []}, "nonempty"),
    ({"coefficients": [float("nan")]}, "finite"),
    ({"coefficients": [1, 1.0]}, "unique"),
    ({"coefficients": [True]}, "booleans"),
    ({"coefficient": 1, "coefficients": [1]}, "not both"),
    ({"schemes": []}, "nonempty"),
    ({"schemes": [{"name": "a"}, {"name": "a"}]}, "names must be unique"),
    ({"schemes": [{"name": "a", "layers": [-1]}]}, "nonnegative"),
    ({"schemes": [{"name": "a", "layers": [1.5]}]}, "integers"),
    ({"layers": [1], "schemes": [{"name": "a"}]}, "inside each scheme"),
    ({"schemes": [{"name": "a", "method": "oops"}]}, "method"),
    ({"schemes": [{"name": "a", "bound": "upper"}]}, "only to projection_cap"),
    ({"schemes": [{"name": "a", "method": "projection_cap", "scaling": "unit"}]}, "only to additive"),
    ({"schemes": [{"name": "a", "method": "projection_cap", "bound": "interval"}]}, "lower or upper"),
])
def test_invalid_sweeps_fail_before_running(steering, message):
    value = config()
    value["steering"] = steering
    with pytest.raises(ValueError, match=message):
        ExperimentConfig.from_mapping(value)


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
    assert not list((recorder.run_dir / "trials").iterdir())
    assert not (recorder.run_dir / "summary.csv").exists()
