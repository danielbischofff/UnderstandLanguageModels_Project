import json

from steering.experiments import ExperimentRecorder, canonical_hash, expand_sweep
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
