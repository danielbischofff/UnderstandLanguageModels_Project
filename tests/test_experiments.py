import json

import pytest
import torch
from safetensors.torch import load_file

from steering.capture import ActivationArtifact, ActivationCaptureEvent
from steering.experiments import (
    ExperimentRecorder,
    activation_capture_from_config,
    canonical_hash,
    expand_sweep,
    run_experiment,
)
from steering.inference import GenerationResponse


def config():
    return {
        "study": "study",
        "experiment": "experiment",
        "conversation": [{"role": "user", "content": "hello"}],
        "steering": {"coefficient": 0.0},
        "sweep": {"seed": [1, 2], "steering.coefficient": [0.0, 0.1]},
    }


def test_cartesian_sweep_is_deterministic():
    trials = expand_sweep(config())
    assert len(trials) == 4
    assert trials == expand_sweep(config())
    assert len({canonical_hash(trial) for trial in trials}) == 4


def test_record_and_resume_skips_complete_trial(tmp_path):
    recorder = ExperimentRecorder.create(config(), root=tmp_path)
    trial = expand_sweep(config())[0]
    response = GenerationResponse("answer", [1, 2], {"seed": 1}, [])
    with recorder:
        recorder.record_result(0, trial, response)
    assert recorder.is_complete(canonical_hash(trial))
    resumed = ExperimentRecorder.resume(recorder.run_dir)
    assert resumed.is_complete(canonical_hash(trial))
    result_path = next((recorder.run_dir / "trials").iterdir()) / "result.json"
    assert json.loads(result_path.read_text())["text"] == "answer"


def test_activation_capture_config_and_safe_recording(tmp_path):
    capture = activation_capture_from_config(
        {
            "layers": [2],
            "points": ["post_intervention"],
            "positions": "last",
            "dtype": "float32",
        }
    )
    assert capture is not None
    assert capture.layers == (2,)
    recorder = ExperimentRecorder.create(config(), root=tmp_path)
    trial = expand_sweep(config())[0]
    event = ActivationCaptureEvent(
        tensor=torch.tensor([[[1.0, 2.0]]]),
        layer=2,
        point="post_intervention",
        phase="decode",
        forward_index=3,
        input_token_ids=[[17]],
        cache_positions=[9],
        position_policy="last",
    )
    response = GenerationResponse(
        "answer",
        [1, 2],
        {"activation_capture": capture.to_dict()},
        [],
        [event],
    )
    with recorder:
        recorder.record_result(0, trial, response)
    trial_dir = next((recorder.run_dir / "trials").iterdir())
    tensors = load_file(trial_dir / "activations" / "activations.safetensors")
    assert torch.equal(tensors["event_000000"], event.tensor)
    manifest = json.loads((trial_dir / "activations" / "manifest.json").read_text())
    assert manifest["coordinate_system"]["tensor_axes"] == [
        "batch",
        "selected_sequence_position",
        "hidden",
    ]
    assert manifest["events"][0]["input_token_ids"] == [[17]]
    assert manifest["events"][0]["cache_positions"] == [9]
    artifact = ActivationArtifact.open(trial_dir / "activations")
    metadata, loaded = next(artifact.events())
    assert metadata["tensor_key"] == "event_000000"
    assert torch.equal(loaded, event.tensor)


def test_failed_trials_are_recorded_and_raise_nonzero_failure(tmp_path):
    class FailingRunner:
        def generate(self, *args, **kwargs):
            raise RuntimeError("generation failed")

    failed_config = {
        "study": "study",
        "experiment": "failure",
        "conversation": [{"role": "user", "content": "hello"}],
    }
    recorder = ExperimentRecorder.create(failed_config, root=tmp_path)
    with pytest.raises(RuntimeError, match="1 of 1 experiment trials failed"):
        run_experiment(FailingRunner(), recorder)
    assert recorder.manifest()["status"] == "complete_with_failures"
    status_path = next((recorder.run_dir / "trials").iterdir()) / "status.json"
    assert json.loads(status_path.read_text())["status"] == "failed"


def test_resume_ignores_a_stale_unlocked_writer_file(tmp_path):
    recorder = ExperimentRecorder.create(config(), root=tmp_path)
    recorder.lock_path.write_text("host=dead-node pid=999999\n")
    resumed = ExperimentRecorder.resume(recorder.run_dir)
    with resumed:
        assert "pid=" in resumed.lock_path.read_text()
