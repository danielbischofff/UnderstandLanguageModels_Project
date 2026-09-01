import json

import pytest

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


def test_config_rejects_unknown_fields_and_duplicate_ids():
    invalid = config()
    invalid["capture"] = {}
    with pytest.raises(ValueError, match="unknown experiment fields"):
        ExperimentConfig.from_mapping(invalid)
    invalid = config()
    invalid["questions"][1]["id"] = "q1"
    with pytest.raises(ValueError, match="question ids must be unique"):
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


class FakeLens:
    source_layers = (0, 1)

    def read(self, events, *, prompt_length):
        assert events == ["temporary activation"]
        assert prompt_length == 2
        return {"tokens": [], "layers": []}


def test_run_records_compact_matrix_without_activation_artifacts(tmp_path):
    recorder = ExperimentRecorder.create(config(), root=tmp_path)
    runner = FakeRunner()
    run_experiment(runner, recorder, FakeLens())

    results = json.loads(
        (recorder.run_dir / "experiment-results.json").read_text(encoding="utf-8")
    )
    assert len(results["trials"]) == 4
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
